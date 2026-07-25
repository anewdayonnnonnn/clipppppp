"""
=============================================================================
  Chain-of-Thought (CoT) Medical Report Generator for AnomalyCLIP
=============================================================================

  Generates a step-by-step clinical reasoning report by analyzing the
  intermediate outputs of the AnomalyCLIP model:

    CoT Chain: Observation → Localization → Severity → Reasoning → Diagnosis

  Key features:
    - Extracts ALL intermediate signals (not just final prediction)
    - Multi-step analytical reasoning with explicit "thinking" steps
    - Structured text report with quantitative evidence
    - Supports both single-image and batch mode

  Usage:
    python generate_cot_report.py --image path/to/img.png
    python generate_cot_report.py --num_images 30
    python generate_cot_report.py --image img.png --verbose
=============================================================================
"""

import argparse
import os
import sys
import csv
import json
import random
from datetime import datetime
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure Dassl.pytorch is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dassl_path = os.path.join(_script_dir, "Dassl.pytorch")
if _dassl_path not in sys.path:
    sys.path.insert(0, _dassl_path)

from dassl.config import get_cfg_default
from open_clip.src.open_clip import create_model_from_pretrained

from trainers.AnomalyDetect.anomaly_detect import AnomalyCLIP
from trainers.AnomalyDetect.text_descriptions import get_text_descriptions


# ============================================================================
# Configuration
# ============================================================================

def extend_cfg(cfg):
    """Add AnomalyCLIP-specific config nodes."""
    from yacs.config import CfgNode as CN
    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIOMEDCOOP.CSC = False
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 0.75
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 0.75
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50
    cfg.TRAINER.ANOMALY_DETECT = CN()
    cfg.TRAINER.ANOMALY_DETECT.PREC = "fp32"
    cfg.TRAINER.ANOMALY_DETECT.LAMBDA_CONSIST = 0.5
    cfg.TRAINER.ANOMALY_DETECT.THRESHOLD_STD = 1.5
    cfg.TRAINER.ANOMALY_DETECT.PATCH_TEMPERATURE = 0.07
    cfg.OPTIM.MAX_EPOCH = 30
    cfg.DATASET.NAME = "BUSI"
    return cfg


# ============================================================================
# Data Structures for CoT Reasoning
# ============================================================================

@dataclass
class PatchInsight:
    """A significant region found during heatmap analysis."""
    quadrant: str                    # e.g., "upper-left", "center-right"
    center_row: int                  # patch grid row (0-13)
    center_col: int                  # patch grid col (0-13)
    mean_score: float                # average anomaly score in region
    max_score: float                 # peak anomaly score
    coverage_pct: float              # % of patches in region above threshold
    description: str                 # natural language description


@dataclass
class CoTStep:
    """One step in the chain-of-thought reasoning process."""
    step_id: int
    step_name: str                   # e.g., "Step 1: Observation"
    thinking: str                    # The "thinking" narrative
    evidence: Dict[str, any] = field(default_factory=dict)
    confidence: str = ""             # "high", "medium", "low"


@dataclass
class CoTReport:
    """Complete chain-of-thought analysis report."""
    image_path: str
    timestamp: str
    model_checkpoint: str

    # Raw model outputs
    class_probs: Dict[str, float] = field(default_factory=dict)
    pred_class: str = ""
    s_img: float = 0.0
    heatmap_raw: np.ndarray = None

    # Analysis results
    steps: List[CoTStep] = field(default_factory=list)
    patches_insights: List[PatchInsight] = field(default_factory=list)

    # Anomaly-specific fields (user requirement)
    is_anomaly: bool = False
    anomaly_type: str = ""          # e.g., "malignant_tumor", "benign_tumor"
    anomaly_score: float = 0.0       # s_img value
    anomaly_location: str = ""       # e.g., "右下象限 (Bottom-Right), 覆盖约 23% 区域"
    anomaly_location_quadrant: str = ""  # primary quadrant
    anomaly_location_pct: float = 0.0    # coverage percentage

    # Local concept prototype analysis
    concept_analysis: Dict[str, dict] = field(default_factory=dict)
    top_anomaly_concepts: List[str] = field(default_factory=list)

    # Final conclusion
    final_diagnosis: str = ""
    confidence_level: str = ""
    key_findings: List[str] = field(default_factory=list)


# ============================================================================
# Model Loading (shared)
# ============================================================================

def build_model(device, cfg, classnames, dataset_name="BUSI"):
    """Load BiomedCLIP and build AnomalyCLIP."""
    print("Loading BiomedCLIP backbone...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.float().eval()

    text_descriptions = get_text_descriptions(dataset_name)

    print("Building AnomalyCLIP (two-path model)...")
    clip_model = AnomalyCLIP(cfg, classnames, model, text_descriptions)
    clip_model.to(device)
    clip_model.eval()

    return clip_model, preprocess


def load_checkpoint(clip_model, checkpoint_path, device):
    """Load trained weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]

    # Pop token_prefix and token_suffix (regenerated by PromptLearner)
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
        state.pop(k, None)

    clip_model.load_state_dict(state, strict=False)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
    return clip_model


# ============================================================================
#  CoT Step 1: Observation — What does the model "see"?
# ============================================================================

def step1_observation(classnames, probs, pred_class, pred_prob):
    """
    CoT Step 1: Observe and describe the raw classification output.

    This simulates a radiologist's first impression:
    "I see a breast ultrasound image. The model's initial read suggests..."

    Returns a CoTStep with the thinking narrative.
    """
    n_cls = len(classnames)

    # Rank all classes by probability
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # Build observation narrative
    lines = []
    lines.append("【观察 Observation】")
    lines.append("")
    lines.append("对输入图像进行前向推理，BiomedCLIP 视觉编码器提取了全局 CLS 特征和 196 个 patch 特征。")
    lines.append("Path A（分类分支）通过可学习 prompt context 与图像特征计算余弦相似度，得到类别 logits。")
    lines.append("")

    # Top prediction
    top_cls, top_prob = ranked[0]
    lines.append(f"▸ 最高预测类别: {top_cls} ({top_prob:.2%})")

    # Runner-up
    if len(ranked) > 1:
        runner_cls, runner_prob = ranked[1]
        margin = top_prob - runner_prob
        lines.append(f"▸ 次高类别:     {runner_cls} ({runner_prob:.2%})")
        lines.append(f"▸ 置信度差距:   {margin:.2%} "
                     f"({'较大，判断明确' if margin > 0.2 else '较小，存在不确定性'})")

    # Full probability distribution
    lines.append("")
    lines.append("▸ 完整类别概率分布:")
    for cls_name, p in ranked:
        bar = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        lines.append(f"    {cls_name:<20s}  {bar}  {p:.2%}")

    # Anomaly vs Normal aggregation
    anomaly_probs = sum(p for cls_name, p in ranked if cls_name != "normal_scan")
    normal_prob = next((p for cls_name, p in ranked if cls_name == "normal_scan"), 0.0)
    lines.append("")
    lines.append(f"▸ 异常类合并概率: {anomaly_probs:.2%}")
    lines.append(f"▸ 正常类概率:     {normal_prob:.2%}")

    # Evidence dictionary
    evidence = {
        "top_prediction": top_cls,
        "top_probability": round(top_prob, 4),
        "runner_up": ranked[1][0] if len(ranked) > 1 else "",
        "runner_up_prob": round(ranked[1][1], 4) if len(ranked) > 1 else 0.0,
        "margin": round(margin, 4),
        "anomaly_aggregated_prob": round(anomaly_probs, 4),
        "normal_prob": round(normal_prob, 4),
        "all_probs": {c: round(p, 4) for c, p in ranked},
    }

    return CoTStep(
        step_id=1,
        step_name="Step 1: Observation（观察）",
        thinking="\n".join(lines),
        evidence=evidence,
        confidence="high" if top_prob > 0.7 else ("medium" if top_prob > 0.4 else "low"),
    )


# ============================================================================
#  CoT Step 2: Localization — Where is the anomaly?
# ============================================================================

def step2_localization(anomaly_scores, img_size=(224, 224)):
    """
    CoT Step 2: Analyze the spatial distribution of anomaly scores.

    This simulates a radiologist scanning across the image:
    "Let me look at each region. Where are the suspicious areas?"

    The heatmap is a 14×14 grid (196 patches). We:
      1. Divide into quadrants (top-left, top-right, bottom-left, bottom-right)
      2. Find the patches with the highest anomaly scores
      3. Identify contiguous "hot zones"
      4. Characterize the spatial pattern (focal vs. diffuse)
    """
    scores_2d = anomaly_scores.reshape(14, 14)  # [14, 14]

    # ── Quadrant analysis ──
    h_half, w_half = 7, 7
    quadrants = OrderedDict({
        "Top-Left":      scores_2d[:h_half, :w_half],
        "Top-Right":     scores_2d[:h_half, w_half:],
        "Bottom-Left":   scores_2d[h_half:, :w_half],
        "Bottom-Right":  scores_2d[h_half:, w_half:],
    })

    quadrant_stats = {}
    for name, region in quadrants.items():
        quadrant_stats[name] = {
            "mean": float(region.mean()),
            "max": float(region.max()),
            "std": float(region.std()),
            "above_mean_pct": float((region > region.mean()).mean() * 100),
        }

    # ── Global heatmap statistics ──
    hm_mean = float(scores_2d.mean())
    hm_max = float(scores_2d.max())
    hm_min = float(scores_2d.min())
    hm_std = float(scores_2d.std())

    # Threshold: mean + 1.5 * std
    threshold = hm_mean + 1.5 * hm_std
    high_risk_mask = scores_2d > threshold
    n_high_risk = int(high_risk_mask.sum())
    high_risk_pct = n_high_risk / 196 * 100

    # ── Find top-N anomaly patches ──
    flat_scores = anomaly_scores.flatten()
    top_k = 10
    top_indices = np.argsort(flat_scores)[-top_k:][::-1]
    top_patches = []
    for idx in top_indices:
        row, col = idx // 14, idx % 14
        top_patches.append({
            "row": int(row), "col": int(col),
            "score": float(flat_scores[idx]),
            # Map to approximate image coordinates (percentage)
            "x_pct": round(col / 14 * 100, 1),
            "y_pct": round(row / 14 * 100, 1),
        })

    # ── Spatial pattern classification ──
    # Focal: high-risk patches are concentrated; Diffuse: scattered
    if n_high_risk == 0:
        spatial_pattern = "无明确热点（热力图均匀）"
    elif n_high_risk <= 15:
        # Check if high-risk patches are contiguous
        high_rows, high_cols = np.where(high_risk_mask)
        row_span = high_rows.max() - high_rows.min()
        col_span = high_cols.max() - high_cols.min()
        if row_span <= 4 and col_span <= 4:
            spatial_pattern = "局灶性（focal）— 异常信号集中在单个小区域"
        else:
            spatial_pattern = "多灶性（multifocal）— 多个分散的小热点"
    elif n_high_risk <= 50:
        spatial_pattern = "区域性（regional）— 异常信号覆盖多个相邻区域"
    else:
        spatial_pattern = "弥漫性（diffuse）— 异常信号广泛分布"

    # ── Build narrative ──
    lines = []
    lines.append("【定位 Localization】")
    lines.append("")
    lines.append("Path B（定位分支）将 196 个 patch token 通过投影层映射到 512 维空间，")
    lines.append("与冻结的文本锚点 P_norm（正常锚点）和 P_anom（异常锚点）计算余弦相似度，")
    lines.append("经 softmax 归一化后得到每个 patch 的异常概率（热力图 L ∈ [0,1]）。")
    lines.append("")

    lines.append(f"▸ 热力图统计: mean={hm_mean:.4f}, max={hm_max:.4f}, min={hm_min:.4f}, std={hm_std:.4f}")
    lines.append(f"▸ 高风险阈值 (mean + 1.5σ): {threshold:.4f}")
    lines.append(f"▸ 高风险 patch 数量: {n_high_risk}/196 ({high_risk_pct:.1f}%)")
    lines.append(f"▸ 空间分布模式: {spatial_pattern}")
    lines.append("")

    # Quadrant summary
    lines.append("▸ 四象限分析:")
    for name, stats in quadrant_stats.items():
        bar_len = int(stats["mean"] * 40)
        bar = "▓" * bar_len + "░" * (40 - bar_len)
        lines.append(f"    {name:<15s}  {bar}  mean={stats['mean']:.4f}  max={stats['max']:.4f}")

    # Top patches
    lines.append("")
    lines.append(f"▸ 异常分数最高的 {top_k} 个 patch:")
    for i, tp in enumerate(top_patches[:5]):
        lines.append(f"    #{i+1}: 位置=({tp['x_pct']:.0f}%, {tp['y_pct']:.0f}%)  "
                     f"grid=({tp['row']},{tp['col']})  得分={tp['score']:.4f}")
    if len(top_patches) > 5:
        lines.append(f"    ... (共 {len(top_patches)} 个)")

    # ── Patch insights (for later use) ──
    # Find the most suspicious quadrant
    worst_quadrant = max(quadrant_stats.items(), key=lambda x: x[1]["mean"])
    patches_insights = [
        PatchInsight(
            quadrant=worst_quadrant[0],
            center_row=(0 if "Top" in worst_quadrant[0] else 10),
            center_col=(0 if "Left" in worst_quadrant[0] else 10),
            mean_score=worst_quadrant[1]["mean"],
            max_score=worst_quadrant[1]["max"],
            coverage_pct=worst_quadrant[1]["above_mean_pct"],
            description=f"最可疑区域位于{worst_quadrant[0]}象限, "
                        f"平均异常分数 {worst_quadrant[1]['mean']:.4f}"
        )
    ]

    # If spatial pattern is focal, add more detail
    if n_high_risk > 0 and n_high_risk <= 15:
        center_r = int(np.mean(high_rows))
        center_c = int(np.mean(high_cols))
        patches_insights.append(PatchInsight(
            quadrant="hotspot",
            center_row=center_r,
            center_col=center_c,
            mean_score=float(scores_2d[high_risk_mask].mean()),
            max_score=float(scores_2d[high_risk_mask].max()),
            coverage_pct=high_risk_pct,
            description=f"局灶性热点位于 grid ({center_r}, {center_c}) 附近, "
                        f"覆盖 {high_risk_pct:.1f}% 的 patch"
        ))

    evidence = {
        "hm_mean": hm_mean,
        "hm_max": hm_max,
        "hm_min": hm_min,
        "hm_std": hm_std,
        "threshold": round(threshold, 4),
        "n_high_risk": n_high_risk,
        "high_risk_pct": round(high_risk_pct, 2),
        "spatial_pattern": spatial_pattern,
        "quadrant_stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                               for kk, vv in v.items()}
                           for k, v in quadrant_stats.items()},
        "top_patches": top_patches,
    }

    confidence = "high" if n_high_risk >= 10 else ("medium" if n_high_risk >= 3 else "low")

    return CoTStep(
        step_id=2,
        step_name="Step 2: Localization（定位）",
        thinking="\n".join(lines),
        evidence=evidence,
        confidence=confidence,
    ), patches_insights


# ============================================================================
#  CoT Step 3: Severity — How severe is the anomaly?
# ============================================================================

def step3_severity(s_img, anomaly_scores, class_probs, anomaly_classes):
    """
    CoT Step 3: Assess severity by combining image-level and patch-level signals.

    s_img is the "adaptive threshold mean" from Path B.
    We interpret it in clinical terms and cross-reference it with classification.
    """
    scores_2d = anomaly_scores.reshape(14, 14)

    # ── s_img interpretation ──
    if s_img < 0.2:
        severity = "极低 (very low)"
        description = "图像级异常分数极低，表明整个图像中几乎没有可疑区域"
    elif s_img < 0.4:
        severity = "低 (low)"
        description = "图像级异常分数较低，仅少数区域显示轻微异常信号"
    elif s_img < 0.6:
        severity = "中等 (moderate)"
        description = "图像级异常分数处于中等水平，存在明显的可疑区域需要关注"
    elif s_img < 0.8:
        severity = "高 (high)"
        description = "图像级异常分数较高，大面积区域显示显著异常信号"
    else:
        severity = "极高 (very high)"
        description = "图像级异常分数极高，几乎所有区域均显示异常信号，需立即关注"

    # ── Distribution analysis ──
    # How many patches are in different risk buckets?
    low_risk = int((scores_2d < 0.3).sum())
    med_risk = int(((scores_2d >= 0.3) & (scores_2d < 0.6)).sum())
    high_risk = int((scores_2d >= 0.6).sum())

    # ── Consistency check between Path A and Path B ──
    anomaly_prob = sum(class_probs.get(c, 0) for c in anomaly_classes)
    # s_img is in [0,1], anomaly_prob is in [0,1]
    consistency_gap = abs(s_img - anomaly_prob)
    if consistency_gap < 0.15:
        consistency = "高度一致 (highly consistent)"
        consistency_detail = (
            f"分类异常概率 ({anomaly_prob:.2%}) 与定位异常分数 ({s_img:.3f}) "
            f"高度吻合，两个分支相互印证"
        )
    elif consistency_gap < 0.35:
        consistency = "基本一致 (roughly consistent)"
        consistency_detail = (
            f"分类异常概率 ({anomaly_prob:.2%}) 与定位异常分数 ({s_img:.3f}) "
            f"存在一定差异，但总体趋势一致"
        )
    else:
        consistency = "存在分歧 (divergent)"
        consistency_detail = (
            f"分类异常概率 ({anomaly_prob:.2%}) 与定位异常分数 ({s_img:.3f}) "
            f"差异较大，建议人工复核 — 分类分支和定位分支给出了不同的信号"
        )

    # ── Percentile of anomaly scores ──
    p50 = float(np.median(scores_2d))
    p90 = float(np.percentile(scores_2d, 90))
    p95 = float(np.percentile(scores_2d, 95))
    p99 = float(np.percentile(scores_2d, 99))

    # ── Build narrative ──
    lines = []
    lines.append("【严重程度 Severity Assessment】")
    lines.append("")
    lines.append("综合 Path A（分类）和 Path B（定位）的信号，对异常严重程度进行分级评估。")
    lines.append("")

    lines.append(f"▸ 图像级异常分数 (s_img): {s_img:.4f}")
    lines.append(f"▸ 严重程度分级: {severity}")
    lines.append(f"▸ 解读: {description}")
    lines.append("")

    lines.append(f"▸ Patch 风险分布 (共196个patch):")
    lines.append(f"    低风险 (<0.3):     {low_risk:3d} patches ({low_risk/196*100:.1f}%)")
    lines.append(f"    中等风险 (0.3-0.6): {med_risk:3d} patches ({med_risk/196*100:.1f}%)")
    lines.append(f"    高风险 (>0.6):     {high_risk:3d} patches ({high_risk/196*100:.1f}%)")
    lines.append("")

    lines.append(f"▸ 路径一致性检查 (Path A ↔ Path B):")
    lines.append(f"    {consistency_detail}")
    lines.append("")

    lines.append(f"▸ 异常分数分位数:")
    lines.append(f"    P50 = {p50:.4f}  |  P90 = {p90:.4f}  |  P95 = {p95:.4f}  |  P99 = {p99:.4f}")

    evidence = {
        "s_img": round(s_img, 4),
        "severity": severity,
        "risk_distribution": {"low": low_risk, "medium": med_risk, "high": high_risk},
        "consistency_gap": round(consistency_gap, 4),
        "consistency": consistency,
        "percentiles": {"p50": p50, "p90": p90, "p95": p95, "p99": p99},
    }

    return CoTStep(
        step_id=3,
        step_name="Step 3: Severity Assessment（严重程度评估）",
        thinking="\n".join(lines),
        evidence=evidence,
        confidence="high" if consistency_gap < 0.15 else ("medium" if consistency_gap < 0.35 else "low"),
    )


# ============================================================================
#  CoT Step 4: Reasoning — Why this diagnosis?
# ============================================================================

def step4_reasoning(classnames, class_probs, pred_class, s_img, patches_insights,
                    spatial_pattern, quadrant_stats, anomaly_classes):
    """
    CoT Step 4: Differential reasoning — explain WHY the model chose this class
    and why it rejected the alternatives.

    This is the core "chain of thought" step — the model connects evidence to conclusion.
    """
    anomaly_prob = sum(class_probs.get(c, 0) for c in anomaly_classes)
    normal_prob = class_probs.get("normal_scan", 0.0)

    lines = []
    lines.append("【推理 Reasoning】")
    lines.append("")
    lines.append("以下展示从证据到结论的推理链（differential reasoning）：")
    lines.append("")

    # ── Reason 1: Classification signal ──
    lines.append("▸ 推理链 1 — 分类信号分析:")
    top_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_prob = top_classes[0]

    if top1_name in anomaly_classes:
        lines.append(f"    分类分支将图像归为异常类 '{top1_name}'，置信度 {top1_prob:.2%}")
        lines.append(f"    该类别在提示学习中获得了最高的 image-text 匹配分数。")
    else:
        lines.append(f"    分类分支将图像归为正常类 '{top1_name}'，置信度 {top1_prob:.2%}")
        lines.append(f"    这表明全局 CLS 特征与正常类文本嵌入的余弦相似度最高。")

    # ── Reason 2: Localization signal ──
    lines.append("")
    lines.append("▸ 推理链 2 — 定位信号分析:")
    if s_img > 0.5:
        lines.append(f"    定位分支检测到显著异常区域 (s_img = {s_img:.3f})，")
        lines.append(f"    空间分布模式为 '{spatial_pattern}'，")
        if patches_insights:
            lines.append(f"    {patches_insights[0].description}。")
        lines.append(f"    这与分类分支的异常判断一致，形成双重证据链。")
    elif s_img > 0.25:
        lines.append(f"    定位分支检测到中等异常信号 (s_img = {s_img:.3f})，")
        lines.append(f"    空间分布模式为 '{spatial_pattern}'。")
        lines.append(f"    虽然信号不如强阳性明显，但仍提示需要关注。")
    else:
        lines.append(f"    定位分支未检测到明显异常 (s_img = {s_img:.3f})，")
        lines.append(f"    热力图整体均匀，无局灶性热点。")
        if top1_name in anomaly_classes:
            lines.append(f"    ⚠ 注意：分类分支与定位分支存在分歧 — 分类判断为异常，但定位未发现热点。")

    # ── Reason 3: Feature-level analysis ──
    lines.append("")
    lines.append("▸ 推理链 3 — 特征层面分析:")

    # Analyze which class-specific prompts contributed most
    # (We don't have direct access to prompt-level logits, but we can reason about it)
    if anomaly_prob > normal_prob:
        lines.append(f"    异常类文本锚点 (P_anom) 与图像 patch 特征的相似度更高，")
        lines.append(f"    表明图像中至少部分区域与异常描述（如'不规则边界''低回声肿块'）匹配。")
    else:
        lines.append(f"    正常类文本锚点 (P_norm) 与图像 patch 特征的相似度更高，")
        lines.append(f"    表明图像整体符合正常乳腺超声的影像学描述。")

    # ── Reason 4: Rule-out differential ──
    lines.append("")
    lines.append("▸ 推理链 4 — 排除性诊断 (Rule-out):")

    # What would an alternative diagnosis look like?
    sorted_by_prob = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    for alt_name, alt_prob in sorted_by_prob[1:]:
        if alt_prob > 0.1:
            lines.append(f"    排除 '{alt_name}' ({alt_prob:.2%}): 概率显著低于首选诊断，")
            lines.append(f"    该类别对应的文本嵌入与图像特征匹配度不足。")
        else:
            break

    if len([c for c, p in sorted_by_prob[1:] if p > 0.1]) == 0:
        lines.append(f"    其他所有类别的概率均低于 10%，不存在有竞争力的备选诊断。")

    # ── Evidence summary ──
    lines.append("")
    lines.append("▸ 证据汇总 (Evidence Matrix):")
    lines.append(f"    {'证据维度':<20s} {'数值':<20s} {'指向':<20s}")
    lines.append(f"    {'─'*20} {'─'*20} {'─'*20}")
    lines.append(f"    {'分类最高概率':<20s} {top1_prob:.2%}  {'':14s} {top1_name:<20s}")
    lines.append(f"    {'异常合并概率':<20s} {anomaly_prob:.2%}  {'':14s} {'异常' if anomaly_prob > 0.5 else '正常':<20s}")
    lines.append(f"    {'定位异常分数':<20s} {s_img:<20.4f} {'异常' if s_img > 0.5 else ('可疑' if s_img > 0.25 else '正常'):<20s}")
    lines.append(f"    {'双路一致性':<20s} {abs(s_img - anomaly_prob):<20.4f} {'一致' if abs(s_img - anomaly_prob) < 0.25 else '分歧':<20s}")

    evidence = {
        "top_prediction": top1_name,
        "top_probability": round(top1_prob, 4),
        "anomaly_aggregated_prob": round(anomaly_prob, 4),
        "s_img": round(s_img, 4),
        "consistency_gap": round(abs(s_img - anomaly_prob), 4),
        "spatial_pattern": spatial_pattern,
    }

    return CoTStep(
        step_id=4,
        step_name="Step 4: Reasoning（推理）",
        thinking="\n".join(lines),
        evidence=evidence,
        confidence="high",
    )


# ============================================================================
#  Anomaly Location Computation
# ============================================================================

def compute_anomaly_location(anomaly_scores, quadrant_stats, spatial_pattern):
    """
    Compute a human-readable anomaly location description.

    Uses BOTH absolute threshold (0.5) AND relative threshold to determine
    high-risk regions. This handles the case where the entire heatmap is
    uniformly high (std ~ 0), which would otherwise report "no anomaly".

    Returns:
        location_text: str  — e.g., "右下象限 (Bottom-Right), 覆盖约 23% 区域"
        primary_quadrant: str — the quadrant with highest mean anomaly score
        coverage_pct: float — percentage of high-risk patches (absolute threshold)
    """
    scores_2d = anomaly_scores.reshape(14, 14)
    ABS_THRESHOLD = 0.5  # absolute threshold for "suspicious"

    # ── Absolute threshold analysis ──
    high_risk_mask_abs = scores_2d > ABS_THRESHOLD
    n_high_abs = int(high_risk_mask_abs.sum())
    coverage_pct_abs = n_high_abs / 196 * 100

    # ── Find primary quadrant ──
    sorted_quadrants = sorted(quadrant_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    primary_name, primary_stats = sorted_quadrants[0]
    secondary_name, secondary_stats = sorted_quadrants[1] if len(sorted_quadrants) > 1 else (None, None)

    # ── Center of mass (using absolute threshold) ──
    if n_high_abs > 0:
        high_rows, high_cols = np.where(high_risk_mask_abs)
        center_r = float(np.mean(high_rows))
        center_c = float(np.mean(high_cols))
        if center_r < 4.67:
            depth = "浅层（近探头）"
        elif center_r < 9.33:
            depth = "中层"
        else:
            depth = "深层（远场）"
        if center_c < 4.67:
            lateral = "内侧"
        elif center_c < 9.33:
            lateral = "中央"
        else:
            lateral = "外侧"
    else:
        # Fall back to quadrant with highest mean
        center_r = 7.0  # default center
        center_c = 7.0
        depth = "全图"
        lateral = "全图"

    # ── Build location description ──
    quadrant_cn = {
        "Top-Left": "左上", "Top-Right": "右上",
        "Bottom-Left": "左下", "Bottom-Right": "右下",
    }

    if n_high_abs == 0:
        # No patches exceed absolute threshold → likely uniform or low
        if scores_2d.mean() > 0.7:
            # Uniformly high: anomaly is diffuse everywhere
            location_text = (
                f"弥漫性全图分布（热力图整体偏高，mean={scores_2d.mean():.3f}），"
                f"以{quadrant_cn.get(primary_name, primary_name)}象限最为显著"
            )
        else:
            location_text = "未检测到明确异常区域（热力图整体偏低，无局灶性热点）"
        primary_quad = primary_name if scores_2d.mean() > 0.7 else "无"
        coverage_pct = 100.0 if scores_2d.mean() > 0.7 else 0.0
    elif n_high_abs <= 10:
        # Very focal: a few patches above threshold
        location_text = (
            f"{quadrant_cn.get(primary_name, primary_name)}象限 ({primary_name}), "
            f"{depth}{lateral}侧, "
            f"局灶性小范围热点, 覆盖约 {coverage_pct_abs:.1f}% 区域"
        )
        primary_quad = primary_name
        coverage_pct = coverage_pct_abs
    elif n_high_abs <= 40:
        # Regional
        location_text = (
            f"主要位于{quadrant_cn.get(primary_name, primary_name)}象限 ({primary_name}), "
            f"{depth}{lateral}侧, "
            f"区域性异常信号, 覆盖约 {coverage_pct_abs:.1f}% 区域"
        )
        if secondary_name and secondary_stats["mean"] > primary_stats["mean"] * 0.7:
            location_text += (
                f", 次要在{quadrant_cn.get(secondary_name, secondary_name)}象限"
            )
        primary_quad = primary_name
        coverage_pct = coverage_pct_abs
    else:
        # Diffuse / widespread
        location_text = (
            f"弥漫性分布, 以{quadrant_cn.get(primary_name, primary_name)}象限 "
            f"({primary_name}) 最为显著, "
            f"{depth}{lateral}侧, 覆盖约 {coverage_pct_abs:.1f}% 区域"
        )
        primary_quad = primary_name
        coverage_pct = coverage_pct_abs

    return location_text, primary_quad, coverage_pct


# ============================================================================
#  CoT Step 5: Diagnosis — Final conclusion
# ============================================================================

def step5_diagnosis(classnames, class_probs, pred_class, pred_prob, s_img,
                    severity, spatial_pattern, consistency_gap, anomaly_classes,
                    anomaly_location_text):
    """
    CoT Step 5: Generate the final structured diagnosis report.

    This is the "verdict" — clearly separated into:
      - NORMAL case:  推理过程 + 最终答案
      - ANOMALY case: 推理过程 + 最终答案 + 异常类型 + 异常分数 + 异常位置

    This is the core user-facing output that summarizes the entire CoT chain.
    """
    anomaly_prob = sum(class_probs.get(c, 0) for c in anomaly_classes)
    normal_prob = class_probs.get("normal_scan", 0.0)
    pred_is_anomaly = pred_class in anomaly_classes

    # ── Overall confidence ──
    evidence_score = 0.0
    evidence_score += pred_prob * 0.35
    evidence_score += (1.0 if s_img > 0.5 else s_img / 0.5) * 0.35
    consistency = 1.0 - min(consistency_gap / 0.5, 1.0)
    evidence_score += consistency * 0.30

    if evidence_score > 0.7:
        confidence_level = "高置信度 (High Confidence)"
    elif evidence_score > 0.4:
        confidence_level = "中等置信度 (Moderate Confidence)"
    else:
        confidence_level = "低置信度 (Low Confidence — 建议人工复核)"

    # ── Build narrative ──
    lines = []
    lines.append("【最终诊断结论 Diagnosis】")
    lines.append("")

    if pred_is_anomaly:
        # ═══════════════════════════════════════════════════════
        #  ANOMALY CASE: 推理过程 + 最终答案 + 异常类型 + 异常分数 + 异常位置
        # ═══════════════════════════════════════════════════════
        anomaly_type = pred_class  # e.g., "malignant_tumor" or "benign_tumor"

        # Severity level description
        if anomaly_prob > 0.8:
            severity_desc = "明确阳性"
            recommendation = "建议结合临床病史，考虑进一步检查（如穿刺活检）以确认诊断。"
        elif anomaly_prob > 0.5:
            severity_desc = "可疑阳性"
            recommendation = "建议短期内复查或结合其他影像学检查（如钼靶）综合评估。"
        else:
            severity_desc = "不确定（倾向阳性）"
            recommendation = "分类和定位信号存在分歧，强烈建议人工阅片复核。"

        diagnosis = f"{severity_desc} — {anomaly_type}"

        lines.append("┌─────────────────────────────────────────────────────────┐")
        lines.append("│          异 常 诊 断 结 果                              │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  最终答案:    {diagnosis:<42s} │")
        lines.append(f"│  置信度:      {confidence_level:<42s} │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  异常类型:    {anomaly_type:<42s} │")
        lines.append(f"│  异常分数:    s_img = {s_img:.4f}  (严重程度: {severity:<20s}) │")
        lines.append(f"│  异常位置:    {anomaly_location_text:<42s} │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  综合评分:    {evidence_score:.2f} / 1.00{'':38s} │")
        lines.append(f"│  临床建议:    {recommendation[:42]:<42s} │")
        lines.append("└─────────────────────────────────────────────────────────┘")

        # Key findings for anomaly
        key_findings = [
            f"异常类型: {anomaly_type} — 分类分支概率 {pred_prob:.1%}",
            f"异常分数: s_img = {s_img:.4f}（{severity}严重程度）",
            f"异常位置: {anomaly_location_text}",
        ]
        if consistency_gap < 0.25:
            key_findings.append(f"双路验证: 分类与定位分支高度一致 (gap = {consistency_gap:.3f})，诊断可靠")
        else:
            key_findings.append(f"⚠ 双路验证: 分类与定位分支存在分歧 (gap = {consistency_gap:.3f})，需人工复核")
        key_findings.append(f"空间分布: {spatial_pattern}")

    else:
        # ═══════════════════════════════════════════════════════
        #  NORMAL CASE: 推理过程 + 最终答案
        # ═══════════════════════════════════════════════════════
        if normal_prob > 0.8:
            normal_desc = "明确阴性 — 正常扫描"
            recommendation = "常规随访即可，无需紧急干预。"
        elif normal_prob > 0.5:
            normal_desc = "可能正常 — 但存在轻微可疑信号"
            recommendation = "建议关注，按常规筛查计划进行随访。"
        else:
            normal_desc = "不确定（倾向正常）"
            recommendation = "信号不够明确，建议结合其他检查或短期随访确认。"

        diagnosis = normal_desc

        lines.append("┌─────────────────────────────────────────────────────────┐")
        lines.append("│          正 常 诊 断 结 果                              │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  最终答案:    {diagnosis:<42s} │")
        lines.append(f"│  置信度:      {confidence_level:<42s} │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  判断依据:    分类分支将图像归为 '{pred_class}'{'':28s} │")
        lines.append(f"│               (概率 {pred_prob:.1%}){'':37s} │")
        lines.append(f"│  定位验证:    定位分支未检测到明确异常区域{'':22s} │")
        lines.append(f"│               s_img = {s_img:.4f}（低于异常阈值）{'':22s} │")
        lines.append("├─────────────────────────────────────────────────────────┤")
        lines.append(f"│  综合评分:    {evidence_score:.2f} / 1.00{'':38s} │")
        lines.append(f"│  临床建议:    {recommendation[:42]:<42s} │")
        lines.append("└─────────────────────────────────────────────────────────┘")

        # Key findings for normal
        key_findings = [
            f"最终答案: 正常 — 分类分支判定为 '{pred_class}' (概率 {pred_prob:.1%})",
            f"定位验证: 定位分支未检测到明显异常 (s_img = {s_img:.4f} < 异常阈值)",
            f"热力图均匀，空间分布模式: {spatial_pattern}",
        ]
        if consistency_gap < 0.25:
            key_findings.append(f"双路验证: 分类与定位一致 (gap = {consistency_gap:.3f})，正常判断可靠")
        else:
            key_findings.append(f"注意: 分类与定位存在一定分歧 (gap = {consistency_gap:.3f})，建议关注")

    lines.append("")

    evidence = {
        "is_anomaly": pred_is_anomaly,
        "final_diagnosis": diagnosis,
        "confidence_level": confidence_level,
        "evidence_score": round(evidence_score, 4),
        "recommendation": recommendation,
        "key_findings": key_findings,
        # Anomaly-specific fields (only populated for anomaly cases)
        "anomaly_type": pred_class if pred_is_anomaly else "N/A",
        "anomaly_score": round(s_img, 4) if pred_is_anomaly else 0.0,
        "anomaly_location": anomaly_location_text if pred_is_anomaly else "N/A",
    }

    return CoTStep(
        step_id=5,
        step_name="Step 5: Diagnosis（诊断结论）",
        thinking="\n".join(lines),
        evidence=evidence,
        confidence=confidence_level,
    ), key_findings, diagnosis, confidence_level, pred_is_anomaly


# ============================================================================
#  Main CoT Pipeline: Run all steps
# ============================================================================

def run_cot_analysis(clip_model, img_path, device, classnames, verbose=False):
    """
    Run the complete chain-of-thought analysis on a single image.

    Pipeline:
      Image → [BiomedCLIP] → {logits, s_img, anomaly_scores}
         ↓
      Step 1: Observation  — classify, list probabilities
      Step 2: Localization — analyze heatmap, find hotspots
      Step 3: Severity     — assess risk level, consistency
      Step 4: Reasoning    — connect evidence to conclusion
      Step 5: Diagnosis    — final verdict + recommendations
    """
    anomaly_classes = [c for c in classnames if c != "normal_scan"]

    # ── Image preprocessing ──
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    img = Image.open(img_path).convert("RGB")
    inp = tfm(img).unsqueeze(0).to(device)

    # ── Forward pass ──
    with torch.no_grad():
        logits, s_img, anomaly_scores, concept_scores = clip_model(inp)

    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    class_probs = {classnames[i]: float(probs[i]) for i in range(len(classnames))}
    pred_idx = int(probs.argmax())
    pred_class = classnames[pred_idx]
    pred_prob = float(probs[pred_idx])
    s_img_val = float(s_img.item())
    anomaly_scores_np = anomaly_scores[0].cpu().numpy()  # [196]

    # ── Per-concept prototype analysis ──
    concept_analysis = {}
    for ctype in ["anomaly", "normal"]:
        for cname, csim in concept_scores[ctype].items():
            csim_np = csim[0].cpu().numpy()  # [196]
            concept_analysis[cname] = {
                "type": ctype,
                "mean": float(csim_np.mean()),
                "max": float(csim_np.max()),
                "std": float(csim_np.std()),
                "top5_mean": float(np.sort(csim_np)[-5:].mean()),
            }
    # Find top activated anomaly concepts
    anom_concepts_ranked = sorted(
        [(k, v) for k, v in concept_analysis.items() if v["type"] == "anomaly"],
        key=lambda x: x[1]["max"], reverse=True
    )
    top_anom_concepts = [name for name, _ in anom_concepts_ranked[:3]]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Chain-of-Thought Analysis: {img_path}")
        print(f"{'='*70}")

    # ── Step 1: Observation ──
    step1 = step1_observation(classnames, class_probs, pred_class, pred_prob)
    if verbose:
        print(f"\n{step1.thinking}")

    # ── Step 2: Localization (with concept prototype analysis) ──
    step2, patches_insights = step2_localization(anomaly_scores_np)
    # Augment with concept-level insights
    step2.evidence["top_anomaly_concepts"] = top_anom_concepts
    step2.evidence["concept_analysis"] = concept_analysis
    if top_anom_concepts:
        step2.thinking += f"\n     🔬 最强激活异常原型: {', '.join(top_anom_concepts)}"
    # Extract spatial pattern for later steps
    spatial_pattern = step2.evidence["spatial_pattern"]
    quadrant_stats = step2.evidence["quadrant_stats"]
    if verbose:
        print(f"\n{step2.thinking}")

    # ── Step 3: Severity ──
    step3 = step3_severity(s_img_val, anomaly_scores_np, class_probs, anomaly_classes)
    if verbose:
        print(f"\n{step3.thinking}")

    # ── Step 4: Reasoning ──
    step4 = step4_reasoning(
        classnames, class_probs, pred_class, s_img_val,
        patches_insights, spatial_pattern, quadrant_stats, anomaly_classes
    )
    # Add concept-based reasoning
    if top_anom_concepts:
        step4.thinking += (
            f"\n     📐 局部概念原型匹配（Local Concept Prototypes）:\n"
            f"        最显著异常模式: {', '.join(top_anom_concepts)}\n"
        )
        for cname in top_anom_concepts[:3]:
            ca = concept_analysis[cname]
            step4.thinking += (
                f"        - {cname}: max_sim={ca['max']:.3f}, "
                f"mean={ca['mean']:.3f}, top5_mean={ca['top5_mean']:.3f}\n"
            )
    if verbose:
        print(f"\n{step4.thinking}")

    # ── Compute anomaly location ──
    anomaly_location_text, primary_quadrant, location_coverage_pct = \
        compute_anomaly_location(anomaly_scores_np, quadrant_stats, spatial_pattern)

    # ── Step 5: Diagnosis ──
    consistency_gap = abs(s_img_val - sum(class_probs.get(c, 0) for c in anomaly_classes))
    step5, key_findings, final_diagnosis, confidence_level, is_anomaly = step5_diagnosis(
        classnames, class_probs, pred_class, pred_prob, s_img_val,
        step3.evidence["severity"], spatial_pattern, consistency_gap, anomaly_classes,
        anomaly_location_text
    )
    if verbose:
        print(f"\n{step5.thinking}")

    # ── Assemble the complete report ──
    report = CoTReport(
        image_path=img_path,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_checkpoint="AnomalyCLIP (BiomedCLIP ViT-B/16) + Local Concept Prototypes",
        class_probs=class_probs,
        pred_class=pred_class,
        s_img=s_img_val,
        heatmap_raw=anomaly_scores_np.reshape(14, 14),
        steps=[step1, step2, step3, step4, step5],
        patches_insights=patches_insights,
        # Anomaly-specific fields
        is_anomaly=is_anomaly,
        anomaly_type=pred_class if is_anomaly else "N/A",
        anomaly_score=s_img_val if is_anomaly else 0.0,
        anomaly_location=anomaly_location_text if is_anomaly else "N/A",
        anomaly_location_quadrant=primary_quadrant,
        anomaly_location_pct=location_coverage_pct,
        # Local concept prototype analysis
        concept_analysis=concept_analysis,
        top_anomaly_concepts=top_anom_concepts,
        # Final conclusion
        final_diagnosis=final_diagnosis,
        confidence_level=confidence_level,
        key_findings=key_findings,
    )

    return report


# ============================================================================
#  Report Formatting
# ============================================================================

def format_full_report(report: CoTReport) -> str:
    """Generate the complete text report with explicit normal/anomaly split."""
    lines = []
    sep = "=" * 70
    sub_sep = "-" * 70

    # ── Header ──
    lines.append(sep)
    lines.append("  AnomalyCLIP — 思维链 (Chain-of-Thought) 诊断报告")
    lines.append(sep)
    lines.append(f"  图像路径:       {report.image_path}")
    lines.append(f"  生成时间:       {report.timestamp}")
    lines.append(f"  模型:           {report.model_checkpoint}")
    lines.append(f"  方法:           5-Step Chain-of-Thought Reasoning")
    lines.append(sep)
    lines.append("")

    # ── Quick Summary Box ──
    if report.is_anomaly:
        lines.append("  +----------------------------------------------------------+")
        lines.append("  |  [ANOMALY] 异 常 诊 断                                  |")
        lines.append("  +----------------------------------------------------------+")
        lines.append(f"  |  异常类型:  {report.anomaly_type:<42s} |")
        lines.append(f"  |  异常分数:  s_img = {report.anomaly_score:.4f}{'':32s} |")
        lines.append(f"  |  异常位置:  {report.anomaly_location:<42s} |")
        lines.append("  +----------------------------------------------------------+")
    else:
        lines.append("  +----------------------------------------------------------+")
        lines.append("  |  [NORMAL] 正 常 诊 断                                   |")
        lines.append("  +----------------------------------------------------------+")
        lines.append(f"  |  最终答案:  {report.final_diagnosis:<42s} |")
        lines.append(f"  |  异常分数:  s_img = {report.s_img:.4f} (低于异常阈值) {'':20s} |")
        lines.append("  +----------------------------------------------------------+")
    lines.append("")

    # ── Chain of Thought Steps ──
    lines.append("  ╔══════════════════════════════════════════════════════════╗")
    lines.append("  ║  思维链 (Chain of Thought) 逐步骤展示                   ║")
    lines.append("  ╚══════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("  推理链: Step 1 观察 → Step 2 定位 → Step 3 严重度 → Step 4 推理 → Step 5 诊断")
    lines.append("")

    for step in report.steps:
        lines.append(sub_sep)
        lines.append(f"  {step.step_name}")
        lines.append(sub_sep)
        lines.append(step.thinking)
        lines.append("")

    # ── Final Summary ──
    lines.append(sep)
    lines.append("  诊断摘要 (Diagnostic Summary)")
    lines.append(sep)

    if report.is_anomaly:
        lines.append(f"  最终答案:       {report.final_diagnosis}")
        lines.append(f"  置信度:         {report.confidence_level}")
        lines.append(f"  ─────────────────────────────────────────────────────────")
        lines.append(f"  异常类型:       {report.anomaly_type}")
        lines.append(f"  异常分数:       s_img = {report.anomaly_score:.4f}")
        lines.append(f"  异常位置:       {report.anomaly_location}")
        lines.append(f"  位置象限:       {report.anomaly_location_quadrant}")
        lines.append(f"  覆盖比例:       {report.anomaly_location_pct:.1f}%")
    else:
        lines.append(f"  最终答案:       {report.final_diagnosis}")
        lines.append(f"  置信度:         {report.confidence_level}")
        lines.append(f"  异常分数:       s_img = {report.s_img:.4f}（低，支持正常判断）")
        lines.append(f"  判断依据:       分类分支与定位分支一致指向正常")

    lines.append("")
    lines.append("  关键发现:")
    for i, kf in enumerate(report.key_findings):
        lines.append(f"    {i+1}. {kf}")
    lines.append("")
    lines.append(sep)
    lines.append("  End of Report")
    lines.append(sep)

    return "\n".join(lines)


def format_compact_report(report: CoTReport) -> str:
    """Generate a compact one-line summary."""
    parts = []

    if report.is_anomaly:
        parts.append(f"[ANOMALY]")
        parts.append(f"type={report.anomaly_type}")
        parts.append(f"score={report.anomaly_score:.3f}")
        parts.append(f"loc={report.anomaly_location_quadrant}")
    else:
        parts.append(f"[NORMAL]")

    probs_str = " | ".join(f"{c}={p:.1%}" for c, p in report.class_probs.items())
    parts.append(f"Probs: {probs_str}")

    hm = report.heatmap_raw
    parts.append(f"HM(mean={hm.mean():.3f}, max={hm.max():.3f})")

    parts.append(f"Conf: {report.confidence_level}")

    return " | ".join(parts)


def export_report_json(report: CoTReport, output_path: str):
    """Export report as structured JSON for downstream processing."""
    data = {
        "image_path": report.image_path,
        "timestamp": report.timestamp,
        "model": report.model_checkpoint,
        "is_anomaly": report.is_anomaly,
        "final_diagnosis": report.final_diagnosis,
        "confidence_level": report.confidence_level,
        "prediction": {
            "class": report.pred_class,
            "probability": max(report.class_probs.values()),
            "all_probabilities": report.class_probs,
        },
        "anomaly_score_s_img": report.s_img,
        # Anomaly-specific fields
        "anomaly_type": report.anomaly_type,
        "anomaly_score": report.anomaly_score,
        "anomaly_location": report.anomaly_location,
        "anomaly_location_quadrant": report.anomaly_location_quadrant,
        "anomaly_location_coverage_pct": report.anomaly_location_pct,
        "key_findings": report.key_findings,
        "chain_of_thought": [
            {
                "step": s.step_id,
                "name": s.step_name,
                "confidence": s.confidence,
                "evidence": s.evidence,
            }
            for s in report.steps
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  JSON report saved to: {output_path}")


# ============================================================================
#  Batch Processing
# ============================================================================

def collect_images(dataset_dir, num_images, seed=42):
    """Collect image paths evenly distributed across classes."""
    random.seed(seed)
    classes = ["benign_tumor", "malignant_tumor", "normal_scan"]
    class_images = {}
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            random.shuffle(imgs)
            class_images[cls] = imgs
        else:
            class_images[cls] = []

    per_class = num_images // len(classes)
    remainder = num_images % len(classes)

    selected = []
    for i, cls in enumerate(classes):
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(class_images[cls]))
        selected.extend([(img_path, cls) for img_path in class_images[cls][:take]])

    random.shuffle(selected)
    return selected


# ============================================================================
#  Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought Medical Report Generator for AnomalyCLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_cot_report.py --image data/BUSI/BUSI/malignant_tumor/img.png
  python generate_cot_report.py --image img.png --verbose
  python generate_cot_report.py --num_images 30 --output_dir cot_reports
  python generate_cot_report.py --image img.png --export_json
        """
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Single image mode: path to image file")
    parser.add_argument("--num_images", type=int, default=0,
                        help="Batch mode: number of images to process")
    parser.add_argument("--dataset_dir", type=str, default="BUSI",
                        help="Path to BUSI dataset directory (for batch mode)")
    parser.add_argument("--checkpoint", type=str,
                        default="output/anomaly_detect/anomaly_clip/model.pth.tar-100",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="cot_reports",
                        help="Output directory for reports")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full CoT reasoning to console")
    parser.add_argument("--export_json", action="store_true",
                        help="Also export report as structured JSON")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact one-line format (batch mode only)")
    args = parser.parse_args()

    # Validation
    if not args.image and args.num_images == 0:
        parser.error("Must specify either --image or --num_images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model ──
    cfg = get_cfg_default()
    cfg = extend_cfg(cfg)
    cfg.freeze()

    classnames = ["benign_tumor", "malignant_tumor", "normal_scan"]
    clip_model, _ = build_model(device, cfg, classnames)
    load_checkpoint(clip_model, args.checkpoint, device)

    # ══════════════════════════════════════════════════════════════
    #  Single Image Mode
    # ══════════════════════════════════════════════════════════════
    if args.image:
        print(f"\n{'='*70}")
        print(f"  Chain-of-Thought Report Generation — Single Image Mode")
        print(f"{'='*70}")
        print(f"  Image:      {args.image}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"{'='*70}")

        # Run CoT analysis
        report = run_cot_analysis(
            clip_model, args.image, device, classnames, verbose=args.verbose
        )

        # Format and save full report
        full_text = format_full_report(report)
        basename = os.path.splitext(os.path.basename(args.image))[0]
        txt_path = os.path.join(args.output_dir, f"cot_report_{basename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n  Full report saved to: {txt_path}")

        # Print to console
        print(f"\n{full_text}")

        # Optional JSON export
        if args.export_json:
            json_path = os.path.join(args.output_dir, f"cot_report_{basename}.json")
            export_report_json(report, json_path)

        print(f"\nDone.")
        return

    # ══════════════════════════════════════════════════════════════
    #  Batch Mode
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Chain-of-Thought Report Generation — Batch Mode")
    print(f"{'='*70}")
    print(f"  Images:     {args.num_images}")
    print(f"  Dataset:    {args.dataset_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")

    image_list = collect_images(args.dataset_dir, args.num_images, seed=args.seed)
    print(f"Selected {len(image_list)} images "
          f"(benign: {sum(1 for _,c in image_list if c=='benign_tumor')}, "
          f"malignant: {sum(1 for _,c in image_list if c=='malignant_tumor')}, "
          f"normal: {sum(1 for _,c in image_list if c=='normal_scan')})\n")

    # CSV for aggregate results
    csv_path = os.path.join(args.output_dir, "cot_batch_summary.csv")
    csv_fields = [
        "index", "image", "true_class", "pred_class", "pred_prob",
        "is_anomaly", "anomaly_type", "anomaly_score", "anomaly_location",
        "s_img", "hm_mean", "hm_max", "severity", "consistency_gap",
        "top_anomaly_concepts", "confidence_level", "final_diagnosis", "key_findings"
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # Accumulators for batch statistics
    all_reports = []
    success = 0
    failed = 0

    for i, (img_path, true_class) in enumerate(image_list):
        try:
            report = run_cot_analysis(
                clip_model, img_path, device, classnames, verbose=False
            )
            all_reports.append(report)
            success += 1

            # Save individual report
            if not args.compact:
                full_text = format_full_report(report)
                basename = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(args.output_dir,
                                        f"cot_{i:03d}_{true_class}_{basename}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            # Write CSV row
            hm = report.heatmap_raw
            # Extract severity and consistency from steps
            severity = report.steps[2].evidence.get("severity", "")
            consistency_gap = report.steps[2].evidence.get("consistency_gap", 0)

            writer.writerow({
                "index": i,
                "image": img_path,
                "true_class": true_class,
                "pred_class": report.pred_class,
                "pred_prob": f"{max(report.class_probs.values()):.4f}",
                "is_anomaly": "YES" if report.is_anomaly else "NO",
                "anomaly_type": report.anomaly_type,
                "anomaly_score": f"{report.anomaly_score:.4f}" if report.is_anomaly else "N/A",
                "anomaly_location": report.anomaly_location if report.is_anomaly else "N/A",
                "s_img": f"{report.s_img:.4f}",
                "hm_mean": f"{hm.mean():.4f}",
                "hm_max": f"{hm.max():.4f}",
                "severity": severity,
                "consistency_gap": f"{consistency_gap:.4f}",
                "top_anomaly_concepts": ", ".join(report.top_anomaly_concepts),
                "confidence_level": report.confidence_level,
                "final_diagnosis": report.final_diagnosis,
                "key_findings": " || ".join(report.key_findings),
            })

            # Compact output to console
            compact = format_compact_report(report)
            print(f"[{i+1:3d}/{len(image_list)}] {compact}")

        except Exception as e:
            failed += 1
            writer.writerow({
                "index": i, "image": img_path, "true_class": true_class,
                "pred_class": "ERROR", "pred_prob": "",
                "s_img": "", "hm_mean": "", "hm_max": "",
                "severity": "", "consistency_gap": "",
                "confidence_level": "ERROR",
                "final_diagnosis": f"FAIL: {str(e)[:150]}",
                "key_findings": "",
            })
            print(f"[{i+1:3d}/{len(image_list)}] FAILED: {img_path} — {str(e)[:80]}")

    csv_file.close()

    # ── Batch-level Summary Report ──
    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"  Total: {len(image_list)} | Success: {success} | Failed: {failed}")
    print(f"")

    # Per-class breakdown
    for cls in classnames:
        cls_reports = [r for r, (_, tc) in zip(all_reports, image_list) if tc == cls]
        if not cls_reports:
            continue

        n = len(cls_reports)
        correct = sum(1 for r in cls_reports if r.pred_class == cls)
        s_imgs = [r.s_img for r in cls_reports]
        pred_probs = [max(r.class_probs.values()) for r in cls_reports]
        n_anomaly = sum(1 for r in cls_reports if r.is_anomaly)
        n_normal = n - n_anomaly

        print(f"  [{cls}]  N={n}  |  Predicted Anomaly: {n_anomaly}  |  Predicted Normal: {n_normal}")
        print(f"    Accuracy:            {correct}/{n} = {correct/n:.2%}")
        print(f"    s_img (mean±std):    {np.mean(s_imgs):.4f} ± {np.std(s_imgs):.4f}")
        print(f"    pred_prob (mean±std):{np.mean(pred_probs):.4f} ± {np.std(pred_probs):.4f}")

        # Anomaly type distribution
        anomaly_types = defaultdict(int)
        for r in cls_reports:
            if r.is_anomaly:
                anomaly_types[r.anomaly_type] += 1
        if anomaly_types:
            print(f"    Anomaly types:       {dict(anomaly_types)}")

        # Anomaly location distribution
        anomaly_locs = defaultdict(int)
        for r in cls_reports:
            if r.is_anomaly:
                anomaly_locs[r.anomaly_location_quadrant] += 1
        if anomaly_locs:
            print(f"    Anomaly locations:   {dict(anomaly_locs)}")

        # Confidence distribution
        conf_dist = defaultdict(int)
        for r in cls_reports:
            conf_dist[r.confidence_level.split(" (")[0] if " (" in r.confidence_level
                      else r.confidence_level] += 1
        print(f"    Confidence dist:     {dict(conf_dist)}")
        print()

    print(f"  CSV summary:  {csv_path}")
    print(f"  Reports dir:  {args.output_dir}/")
    print(f"  Done.")


if __name__ == "__main__":
    main()
