# pylint: disable = C0302, E0611
import copy
import math
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from monai.losses.focal_loss import FocalLoss
from scipy.ndimage import convolve1d
from scipy.stats import norm
from skimage.exposure import equalize_adapthist
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss, Module, MSELoss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from src.model import MammoCNN

# Constraint: Diagnosis weight >= all auxiliary task weights to ensure primary task focus
# Reference: "MAMMO: A Deep Learning Solution for Facilitating Radiologist-Machine
#             Collaboration in Breast Cancer Diagnosis Trent Kyono, Fiona J. Gilbert,
#             Mihaela van der Schaar, Fellow, IEEE (30th October, 2018)"

LOSS_WEIGHTS: dict[str, float] = {
    "diagnosis": 0.50,  # Primary task
    "findings": 0.15,
    "suspicion": 0.15,
    "density": 0.10,
    "age": 0.10,
}

# BI-RADS is 1-5 and density 1-4 in the processed metadata; the ordinal heads
# need 0-based level indices, so the dataset subtracts these.
BIRADS_MIN: int = 1
DENSITY_MIN: int = 1

TASKS: tuple[str, ...] = ("diagnosis", "findings", "suspicion", "density", "age")

CLAHE_CLIP_SCALE: float = 0.005
CLAHE_NBINS: int = 256


def apply_clahe(
    image: np.ndarray | torch.Tensor,
    grid_size: int = 8,
    clip_limit: float = 2.0,
    is_training: bool = False,
) -> torch.Tensor:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and
    returns a 3-channel float32 tensor in CHW format with values in [0, 1].
    Accepts a NumPy array or a Torch tensor, 1- or 3-channel, int or float.

    `grid_size` and `clip_limit` are in the paper's units; see
    CLAHE_CLIP_SCALE for the conversion.
    """
    # Accept both NumPy arrays and torch tensors
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            # Convert CHW -> HWC
            image_np = image.permute(1, 2, 0).cpu().numpy()
        elif image.ndim == 2:
            image_np = image.cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor image shape: {image.shape}")
    else:
        image_np = image

    # Equalise a single channel.
    if image_np.ndim == 3:
        if image_np.shape[-1] == 3:
            assert np.array_equal(
                image_np[..., 0], image_np[..., 1]
            ) and np.array_equal(image_np[..., 0], image_np[..., 2]), (
                "apply_clahe got a 3-channel image whose channels differ"
            )
        elif image_np.shape[-1] != 1:
            raise ValueError(f"Unsupported image shape for CLAHE: {image_np.shape}")
        image_np = image_np[..., 0]
    elif image_np.ndim != 2:
        raise ValueError(f"Unsupported image shape for CLAHE: {image_np.shape}")

    # Scale by the dtype's own maximum so 8-bit and 16-bit sources both land in
    # [0, 1].
    if np.issubdtype(image_np.dtype, np.integer):
        image_np = image_np.astype(np.float32) / float(np.iinfo(image_np.dtype).max)
    else:
        image_np = image_np.astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)

    if is_training:
        grid_size = int(
            grid_size + np.random.uniform(-np.log2(grid_size), np.log2(grid_size))
        )
        clip_limit = clip_limit + np.random.uniform(
            -np.log2(clip_limit), np.log2(clip_limit)
        )

    grid_size = max(1, grid_size)
    height, width = image_np.shape

    equalized = equalize_adapthist(
        image_np,
        kernel_size=(max(1, height // grid_size), max(1, width // grid_size)),
        clip_limit=clip_limit * CLAHE_CLIP_SCALE,
        nbins=CLAHE_NBINS,
    ).astype(np.float32)

    # timm builds inception_resnet_v2 with the default in_chans=3
    image_tensor = torch.from_numpy(equalized).unsqueeze(0).repeat(3, 1, 1)

    return image_tensor


def preprocess_image(
    image_path: str,
    laterality: str,
    target_size: tuple = (1024, 1280),  # (W, H)
) -> torch.Tensor:
    """
    Preprocesses the image by loading, orienting, downsampling, and
    returning a normalized single-channel tensor with values in [0, 1].
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(image_path)

    if image.ndim == 3:
        image = image[..., 0]

    # Scale by the dtype's own maximum, so a 16-bit PNG and a legacy 8-bit one
    # both land in [0, 1]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)  # type: ignore
    else:
        image = image.astype(np.float32)

    image = cv2.flip(image, 1) if laterality.upper() == "R" else image

    # Resize in float and clip afterwards: Lanczos has negative lobes and rings
    # past the input range, which would wrap around if it were cast to uint16.
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    image = np.clip(image, 0.0, 1.0)

    return torch.from_numpy(image).unsqueeze(0)


def get_lds_weights(
    labels: list[int],
    label_min: int,
    label_max: int,
    kernel: str = "gaussian",  # "gaussian" or "laplacian"
    kernel_std: float = 1.0,  # σ for gaussian, scale for laplacian
    kernel_size: int = 5,  # must be odd
    reweight: str = "inverse",  # "inverse" or "sqrt_inverse"
) -> dict[int, float]:
    """
    Compute per-label LDS weights for an ordinal regression target.

    Args:
        labels: list of raw integer label values from the training set
        label_min: minimum label value
        label_max: maximum label value
        kernel: smoothing kernel type
        kernel_std: controls kernel width — higher = more smoothing
        start with 1.0 for 5-bin suspicion, tune if needed
        kernel_size: number of kernel points (odd number)
        reweight: "inverse"      → w = 1 / smoothed_density
        "sqrt_inverse" → w = 1 / sqrt(smoothed_density)
        sqrt_inverse is gentler — use if inverse overweights rare bins

    Returns:
        dict mapping each integer label → float weight
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert len(labels) > 0, "labels must be non-empty"
    assert label_min <= label_max, "label_min must be <= label_max"

    # ── Step 1: Empirical frequency histogram ─────────────────────────
    # Map label values to bin indices
    n_bins = label_max - label_min + 1  # Number of discrete label values

    max_kernel_size = n_bins if n_bins % 2 == 1 else n_bins - 1
    if kernel_size > max_kernel_size:
        kernel_size = max(1, max_kernel_size)

    counts = np.zeros(n_bins)
    label_to_bin = {val: idx for idx, val in enumerate(range(label_min, label_max + 1))}

    for label in labels:
        counts[label_to_bin[label]] += 1

    # ── Step 2: Build kernel ───────────────────────────────────────────
    half = kernel_size // 2
    x = np.arange(-half, half + 1)  # [-2, -1, 0, 1, 2] for size=5

    if kernel == "gaussian":
        k = norm.pdf(x, scale=kernel_std)
    elif kernel == "laplacian":
        k = np.exp(-np.abs(x) / kernel_std)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    k /= k.sum()  # normalize kernel to sum to 1

    # ── Step 3: Convolve → smoothed effective density ─────────────────
    # mode="reflect" mirrors the signal at edges — appropriate for bounded scales
    smoothed = convolve1d(counts, weights=k, mode="reflect")
    smoothed = np.maximum(smoothed, 1e-8)  # prevent division by zero

    # ── Step 4: Compute weights ────────────────────────────────────────
    if reweight == "inverse":
        weights = 1.0 / smoothed
    elif reweight == "sqrt_inverse":
        weights = 1.0 / np.sqrt(smoothed)
    else:
        raise ValueError(f"Unknown reweight: {reweight}")

    # Normalize: weights average to 1.0 across the training set
    # (preserves the overall scale of the loss)
    weights_per_sample = np.array([weights[label_to_bin[l]] for l in labels])
    weights = weights / weights_per_sample.mean()

    return {val: float(weights[idx]) for val, idx in label_to_bin.items()}


def compute_multi_task_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_weights: dict[str, float] | None = None,
    findings_weights: torch.Tensor | None = None,
    suspicion_weights: torch.Tensor | None = None,
    density_weights: torch.Tensor | None = None,
    loss_scales: dict[str, float] | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes the weighted multi-task loss for MAMMO CNN.

    Args:
        outputs: dict of model outputs for each task
        targets: dict of ground truth labels for each task
        loss_weights: dict of weights for each task's loss
        findings_weights: class weights for the findings head
        suspicion_weights: per-cutpoint pos_weight for suspicion
        density_weights: class weights for the density head
        loss_scales: Optional per-task multipliers from calibrate_loss_scales()
        alpha: Focal Loss alpha parameter. NOTE: the paper uses alpha=2 because
            its Eq. 8 defines alpha as an inverse-class-frequency multiplier.
            MONAI uses Lin et al.'s other parameterisation where alpha is a
            balancing fraction in [0, 1]. The faithful translation of the
            paper's alpha is the per-class `weight` argument, not this one.
        gamma: Focal Loss gamma parameter

    Returns:
        total_loss: The combined weighted loss for all tasks
        individual_losses: dict of raw (unscaled) loss values for each task
    """
    if loss_weights is None:
        loss_weights = LOSS_WEIGHTS

    assert loss_weights["diagnosis"] >= sum(
        v for k, v in loss_weights.items() if k != "diagnosis"
    ), (
        "Diagnosis loss weight must be greater than or equal to the sum of auxiliary task weights."
    )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha must lie in [0, 1] — MONAI applies it as "
            f"target * alpha + (1 - target) * (1 - alpha), so alpha={alpha} would "
            f"give negative samples a negative weight and reward wrong predictions."
        )

    # ── Move class weights onto the logits' device ────────────────────
    device = outputs["diagnosis"].device

    if findings_weights is not None:
        findings_weights = findings_weights.to(device)
    if suspicion_weights is not None:
        suspicion_weights = suspicion_weights.to(device)
    if density_weights is not None:
        density_weights = density_weights.to(device)

    # ── Diagnosis: binary focal loss ──────────────────────────────────
    # MONAI indexes input.shape[1], so the head must stay (B, 1) — never (B,).
    diagnosis_logits = outputs["diagnosis"]
    if diagnosis_logits.ndim == 1:
        diagnosis_logits = diagnosis_logits.unsqueeze(1)
    diagnosis_target = targets["diagnosis"].float().view_as(diagnosis_logits)

    diagnosis_loss_fn = FocalLoss(
        include_background=True,
        to_onehot_y=False,
        gamma=gamma,
        alpha=alpha,
        reduction="mean",
        use_softmax=False,
    )
    findings_loss_fn = FocalLoss(
        include_background=True,
        to_onehot_y=False,
        gamma=gamma,
        weight=findings_weights,  # per-class (C,) tensor, or None
        reduction="mean",
        use_softmax=False,
    )
    density_loss_fn = CrossEntropyLoss(weight=density_weights, reduction="mean")
    age_loss_fn = MSELoss(reduction="mean")

    diagnosis_loss: torch.Tensor = diagnosis_loss_fn(diagnosis_logits, diagnosis_target)
    findings_loss: torch.Tensor = findings_loss_fn(
        outputs["findings"], targets["findings"].float()
    )
    suspicion_loss: torch.Tensor = coral_loss(
        outputs["suspicion"], targets["suspicion"], cutpoint_weights=suspicion_weights
    )
    density_loss: torch.Tensor = density_loss_fn(outputs["density"], targets["density"])
    age_loss: torch.Tensor = age_loss_fn(outputs["age"], targets["age"])

    raw = {
        "diagnosis": diagnosis_loss,
        "findings": findings_loss,
        "suspicion": suspicion_loss,
        "density": density_loss,
        "age": age_loss,
    }

    # scale (put every task on a common magnitude) then weight (apply intent)
    total_loss = sum(
        loss_weights[task] * (loss_scales or {}).get(task, 1.0) * value
        for task, value in raw.items()
    )

    # Report raw values for logging and effective contribution calculation,
    # but don't backprop through them
    loss_components = {task: value.item() for task, value in raw.items()}

    return total_loss, loss_components  # type: ignore


def levels_from_labels(labels: torch.Tensor, n_cutpoints: int) -> torch.Tensor:
    """
    Expand 0-based ordinal labels into CORAL binary level targets.

    label y becomes [1 if y > k else 0 for k in range(n_cutpoints)], e.g. on a
    5-level scale y=0 -> [0,0,0,0], y=3 -> [1,1,1,0], y=4 -> [1,1,1,1].

    Args:
        labels: (B,) long tensor of levels in [0, n_cutpoints]
        n_cutpoints: number of cutpoints (n_classes - 1)

    Returns:
        (B, n_cutpoints) float tensor of cumulative binary targets
    """
    thresholds = torch.arange(n_cutpoints, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(
    cutpoint_logits: torch.Tensor,
    labels: torch.Tensor,
    cutpoint_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CORAL ordinal loss — the sum of the per-cutpoint binary cross-entropies.

    Args:
        cutpoint_logits: (B, K-1) raw logits from CoralHead
        labels: (B,) long tensor of 0-based levels
        cutpoint_weights: optional (K-1,) pos_weight per cutpoint, to counter
            the class imbalance at each threshold

    Returns:
        scalar mean loss
    """
    n_cutpoints = cutpoint_logits.shape[1]
    levels = levels_from_labels(labels, n_cutpoints)

    if cutpoint_weights is not None:
        cutpoint_weights = cutpoint_weights.to(cutpoint_logits.device)

    # reduction="none" then sum over cutpoints, mean over batch — so the loss
    # scale is "errors per sample", not "errors per sample per cutpoint"
    per_cutpoint = binary_cross_entropy_with_logits(
        cutpoint_logits, levels, pos_weight=cutpoint_weights, reduction="none"
    )
    return per_cutpoint.sum(dim=1).mean()


def effective_contributions(
    loss_components: dict[str, float],
    loss_weights: dict[str, float],
    loss_scales: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Share of the total loss each task actually contributes.

    Returns:
        dict of task -> fraction of total weighted loss
    """
    if loss_weights is None:
        loss_weights = LOSS_WEIGHTS

    scales = loss_scales or {}
    weighted = {
        task: loss_weights.get(task, 0.0) * scales.get(task, 1.0) * value
        for task, value in loss_components.items()
    }
    total = sum(weighted.values())

    if total == 0:
        return dict.fromkeys(weighted, 0.0)

    return {task: value / total for task, value in weighted.items()}


@torch.no_grad()
def calibrate_loss_scales(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict | None = None,
    n_batches: int = 10,
    max_scale: float = 100.0,
) -> dict[str, float]:
    """
    Measure each task's raw loss magnitude, return 1/magnitude per task.

    Feed the result back as `loss_scales` and LOSS_WEIGHTS becomes literal: a
    coefficient of 0.50 really is half the loss. Run this ONCE on the
    freshly-initialised model, before training.

    Args:
        model: MammoCNN, freshly initialised
        loader: training loader (uses the first n_batches only)
        device: torch device
        loss_kwargs: forwarded to compute_multi_task_loss; any `loss_scales`
            key present is ignored, since that is what is being computed
        n_batches: how many batches to average over
        max_scale: upper bound on any single multiplier

    Returns:
        dict of task -> scale multiplier
    """
    loss_kwargs = {k: v for k, v in (loss_kwargs or {}).items() if k != "loss_scales"}
    was_training = model.training
    model.eval()
    model.to(device)

    totals: dict[str, float] = {}
    seen = 0

    for batch in loader:
        images = batch["mammogram"].to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "mammogram"
        }

        _, components = compute_multi_task_loss(model(images), targets, **loss_kwargs)

        for task, value in components.items():
            totals[task] = totals.get(task, 0.0) + value

        seen += 1
        if seen >= n_batches:
            break

    if was_training:
        model.train()

    if seen == 0:
        raise ValueError("loader yielded no batches")

    scales = {}
    for task, total in totals.items():
        magnitude = total / seen
        scale = max_scale if magnitude <= 0 else min(1.0 / magnitude, max_scale)

        if scale == max_scale:
            warnings.warn(
                f"loss scale for '{task}' hit the cap of {max_scale} (raw magnitude "
                f"{magnitude:.2e}). That head starts near-solved, so its share will "
                f"fall below its LOSS_WEIGHTS coefficient — usually fine, but check "
                f"the targets are what you expect.",
                stacklevel=2,
            )

        scales[task] = scale

    return scales


def ordinal_class_probs(cutpoint_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL cutpoint logits into a per-level probability distribution.

    From the cumulative probabilities P(y > k):
        P(y = 0) = 1 - P(y > 0)
        P(y = k) = P(y > k-1) - P(y > k)
        P(y = K - 1) = P(y > K - 2)

    Args:
        cutpoint_logits: (B, K-1) raw cutpoint logits

    Returns:
        (B, K) probabilities summing to 1 along dim 1
    """
    cumulative = torch.sigmoid(cutpoint_logits)  # (B, K-1) = P(y > k)
    ones = torch.ones_like(cumulative[:, :1])
    zeros = torch.zeros_like(cumulative[:, :1])

    upper = torch.cat([ones, cumulative], dim=1)  # P(y > k-1), k = 0..K-1
    lower = torch.cat([cumulative, zeros], dim=1)  # P(y > k),   k = 0..K-1

    return (upper - lower).clamp_min(0.0)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, or NaN when no positives are present."""
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


@torch.no_grad()
def evaluate(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict | None = None,
    findings_names: list[str] | None = None,
    age_std: float | None = None,
) -> dict[str, Any]:
    """
    Run one validation pass and return metrics for every head.

    Args:
        model: MammoCNN (or compatible)
        loader: validation/test DataLoader
        device: torch device
        loss_kwargs: forwarded to compute_multi_task_loss
        findings_names: optional class names for per-class findings metrics
        age_std: training-set age std, to report age MAE in years rather than
            in standardised units (the paper reports 5.97 years)

    Returns:
        dict of metric name -> value; per-class entries are nested dicts
    """
    loss_kwargs = loss_kwargs or {}
    model.eval()
    model.to(device)

    losses: list[float] = []
    n_density_classes = 0
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "diagnosis_score",
            "ordinal_score",
            "diagnosis_label",
            "findings_score",
            "findings_label",
            "suspicion_cutpoint_scores",
            "suspicion_level_pred",
            "suspicion_level_true",
            "density_pred",
            "density_true",
            "age_pred",
            "age_true",
        )
    }

    for batch in loader:
        images = batch["mammogram"].to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "mammogram"
        }

        outputs = model(images)
        total_loss, _ = compute_multi_task_loss(outputs, targets, **loss_kwargs)
        losses.append(total_loss.item())

        def add(key: str, tensor: torch.Tensor) -> None:
            collected[key].append(tensor.cpu().numpy())

        add("diagnosis_score", torch.sigmoid(outputs["diagnosis"].flatten()))
        add("ordinal_score", torch.sigmoid(outputs["diagnosis_ordinal"].flatten()))
        add("diagnosis_label", targets["diagnosis"].flatten())

        add("findings_score", torch.sigmoid(outputs["findings"]))
        add("findings_label", targets["findings"])

        add("suspicion_cutpoint_scores", torch.sigmoid(outputs["suspicion"]))
        add("suspicion_level_pred", MammoCNN.suspicion_to_level(outputs["suspicion"]))
        add("suspicion_level_true", targets["suspicion"])

        n_density_classes = outputs["density"].shape[1]
        add("density_pred", outputs["density"].argmax(dim=1))
        add("density_true", targets["density"])

        add("age_pred", outputs["age"].flatten())
        add("age_true", targets["age"].flatten())

    joined = {key: np.concatenate(values) for key, values in collected.items()}

    metrics: dict[str, Any] = {"val_loss": float(np.mean(losses))}

    # ── Diagnosis: dedicated head vs the CORAL cutpoint ───────────────
    labels = joined["diagnosis_label"]
    metrics["diagnosis_auc"] = _safe_auc(labels, joined["diagnosis_score"])
    metrics["diagnosis_ap"] = _safe_ap(labels, joined["diagnosis_score"])
    metrics["ordinal_auc"] = _safe_auc(labels, joined["ordinal_score"])
    metrics["ordinal_ap"] = _safe_ap(labels, joined["ordinal_score"])

    # ── Findings: multilabel, per class ───────────────────────────────
    findings_score = joined["findings_score"]
    findings_label = joined["findings_label"]
    n_findings = findings_score.shape[1]
    names = findings_names or [f"finding_{i}" for i in range(n_findings)]

    per_class_auc, per_class_ap, support = {}, {}, {}
    for index in range(n_findings):
        name = names[index] if index < len(names) else f"finding_{index}"
        column = findings_label[:, index]
        per_class_auc[name] = _safe_auc(column, findings_score[:, index])
        per_class_ap[name] = _safe_ap(column, findings_score[:, index])
        support[name] = int(column.sum())

    metrics["findings_auc_per_class"] = per_class_auc
    metrics["findings_ap_per_class"] = per_class_ap
    metrics["findings_support"] = support
    metrics["n_images"] = int(findings_label.shape[0])

    # macro over classes that actually have positives in this split
    metrics["findings_macro_auc"] = float(np.nanmean(list(per_class_auc.values())))
    metrics["findings_macro_ap"] = float(np.nanmean(list(per_class_ap.values())))

    # micro: pool all class-instance pairs, dominated by the common classes
    metrics["findings_micro_ap"] = _safe_ap(
        findings_label.ravel(), findings_score.ravel()
    )

    # ── Suspicion: ordinal ───────────────
    true_levels = joined["suspicion_level_true"]
    pred_levels = joined["suspicion_level_pred"]
    n_levels = joined["suspicion_cutpoint_scores"].shape[1] + 1

    metrics["suspicion_qwk"] = float(
        cohen_kappa_score(
            true_levels, pred_levels, weights="quadratic", labels=list(range(n_levels))
        )
    )

    # Per-CUTPOINT AUROC — "is y > k?"
    metrics["suspicion_auc_per_cutpoint"] = {
        f"gt_{level + BIRADS_MIN}": _safe_auc(
            (true_levels > level).astype(int),
            joined["suspicion_cutpoint_scores"][:, level],
        )
        for level in range(n_levels - 1)
    }

    # ── Density: ordinal too (A < B < C < D) ─────────
    density_true = joined["density_true"]
    density_pred = joined["density_pred"]
    density_labels = list(range(n_density_classes))

    for name, fn in (
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
    ):
        metrics[f"density_{name}"] = float(
            fn(
                density_true,
                density_pred,
                average="macro",
                labels=density_labels,
                zero_division=0,
            )
        )

    # Per-class F1
    per_class_f1 = f1_score(
        density_true,
        density_pred,
        average=None,
        labels=density_labels,
        zero_division=0,
    )
    metrics["density_f1_per_class"] = {
        f"class_{chr(ord('A') + i)}": float(v)
        for i, v in enumerate(per_class_f1)  # type: ignore
    }
    metrics["density_support"] = {
        f"class_{chr(ord('A') + i)}": int((density_true == i).sum())
        for i in density_labels
    }
    metrics["density_qwk"] = float(
        cohen_kappa_score(
            density_true, density_pred, weights="quadratic", labels=density_labels
        )
    )

    age_mae = float(np.abs(joined["age_pred"] - joined["age_true"]).mean())
    metrics["age_mae_standardised"] = age_mae
    metrics["age_mae_years"] = age_mae * age_std if age_std else float("nan")

    return metrics


def _flatten_for_wandb(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten evaluate()'s nested per-class dicts into `val/...` scalars.

    Weights & Biases charts one scalar per key, so the per-class findings and
    per-level suspicion dicts have to be expanded. Slashes create the grouping
    in the W&B sidebar, so per-class curves land together under one panel.
    Class names are slugged because spaces in keys make panel filters awkward.
    """
    flat: dict[str, float] = {}
    scalar_keys = {k for k, v in metrics.items() if not isinstance(v, dict)}

    for key, value in metrics.items():
        if isinstance(value, dict):
            group = (
                key.removesuffix("_per_class")
                .removesuffix("_per_level")
                .removesuffix("_per_cutpoint")
            )
            if group in scalar_keys:
                group = key
            for name, item in value.items():
                slug = str(name).lower().replace(" ", "_")
                flat[f"val/{group}/{slug}"] = float(item)
        else:
            flat[f"val/{key}"] = float(value)

    return flat


def train(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    patience: int = 15,
    learning_rate: float = 1e-4,
    lr_decay_factor: float = 0.1,
    lr_patience: int = 5,
    min_lr: float = 1e-6,
    warmup_epochs: float = 1.0,
    backbone_lr_mult: float = 0.1,
    weight_decay: float = 1e-4,
    accumulation_steps: int = 2,
    max_grad_norm: float | None = 1.0,
    loss_kwargs: dict | None = None,
    checkpoint_path: str | Path | None = None,
    monitor: str = "diagnosis_ap",
    monitor_mode: str = "max",
    use_amp: bool = True,
    findings_names: list[str] | None = None,
    age_std: float | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """
    Train the MAMMO CNN with gradient accumulation.

    Args:
        model: MammoCNN instance
        train_loader: shuffled loader over the CNN training split
        val_loader: validation loader
        device: torch device
        epochs: number of passes over the training split
        patience: epochs without improvement before stopping
        learning_rate: AdamW learning rate
        lr_decay_factor: multiplier applied on plateau
        lr_patience: epochs without improvement before reducing the LR
        min_lr: floor for the learning rate
        warmup_epochs: linear LR ramp from ~0 to full over this many epochs.
            Matters most when the backbone is unfrozen from step 1: the heads
            start random, so their gradients are large and noisy, and at full
            LR they overwrite the pretrained ImageNet features before learning
            anything. Set 0 to disable.
        backbone_lr_mult: backbone LR as a fraction of `learning_rate`. The
            paper protects pretrained features by alternating frozen and
            unfrozen stages (Appendix C-B, after Shen 2017); discriminative
            rates achieve the same end while training everything jointly.
            Set 1.0 for a single uniform LR.
        weight_decay: AdamW weight decay
        accumulation_steps: micro-batches per optimizer step
        max_grad_norm: gradient clipping threshold, or None to disable
        loss_kwargs: forwarded to compute_multi_task_loss (class weights etc.)
        checkpoint_path: where to save the best checkpoint, or None
        monitor: validation metric driving checkpoints, early stop and LR decay
        monitor_mode: "max" for AP/AUROC style metrics, "min" for losses/MAE
        use_amp: enable mixed precision (CUDA only; ignored on CPU)
        findings_names: class names for per-class findings metrics
        age_std: training-set age std, so age MAE is reported in years
        wandb_run: an active `wandb.init()` run to log to, or None

    Returns:
        history dict of per-epoch metric lists
    """
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

    if monitor_mode not in ("max", "min"):
        raise ValueError(f"monitor_mode must be 'max' or 'min', got {monitor_mode!r}")

    if patience <= lr_patience:
        warnings.warn(
            f"patience={patience} <= lr_patience={lr_patience}: training will stop "
            f"before a decayed learning rate has any epochs to prove itself. "
            f"Use patience >= 2 * lr_patience + 1 (here, >= {2 * lr_patience + 1}).",
            stacklevel=2,
        )

    early_stop_counter = 0
    loss_kwargs = loss_kwargs or {}
    model.to(device)

    # Discriminative learning rates: the pretrained backbone moves slowly, the
    # randomly-initialised heads move at full speed. Falls back to one group if
    # the model has no `backbone` attribute or backbone_lr_mult == 1.
    backbone_params, head_params = [], []
    backbone_module = getattr(model, "backbone", None)
    backbone_ids = (
        {id(p) for p in backbone_module.parameters()} if backbone_module else set()
    )

    for param in model.parameters():
        if not param.requires_grad:
            continue
        (backbone_params if id(param) in backbone_ids else head_params).append(param)

    param_groups = [{"params": head_params, "lr": learning_rate, "name": "heads"}]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": learning_rate * backbone_lr_mult,
                "name": "backbone",
            }
        )

    optimizer = AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)

    # Remember each group's target LR so warmup can ramp towards it and the
    # plateau scheduler can decay from it independently per group.
    for group in optimizer.param_groups:
        group["target_lr"] = group["lr"]

    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    optimizer_step = 0

    # Same metric and direction as the checkpoint/early-stop criterion
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=monitor_mode,
        factor=lr_decay_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )

    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history: dict[str, Any] = {"train_loss": [], "components": []}
    best_score = -math.inf if monitor_mode == "max" else math.inf
    best_state = None
    best_epoch = 0
    n_batches = len(train_loader)

    def improved(current: float) -> bool:
        return current > best_score if monitor_mode == "max" else current < best_score

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_components: dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)

        for step, batch in enumerate(progress):
            images = batch["mammogram"].to(device, non_blocking=True)
            targets = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "mammogram"
            }

            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(images)
                total_loss, components = compute_multi_task_loss(
                    outputs, targets, **loss_kwargs
                )

            # Scale so accumulated gradients average rather than sum
            scaler.scale(total_loss / accumulation_steps).backward()

            is_step = (step + 1) % accumulation_steps == 0
            is_last = step + 1 == n_batches

            # `is_last` matters: without it a trailing partial group would be
            # computed, never stepped, and silently carried into the next epoch
            if is_step or is_last:
                # Warmup is applied per optimizer step
                if optimizer_step < warmup_steps:
                    ramp = (optimizer_step + 1) / warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = group["target_lr"] * ramp

                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            running_loss += total_loss.item()
            for key, value in components.items():
                running_components[key] = running_components.get(key, 0.0) + value

            progress.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        train_loss = running_loss / n_batches
        components_mean = {k: v / n_batches for k, v in running_components.items()}

        metrics = evaluate(
            model, val_loader, device, loss_kwargs, findings_names, age_std
        )

        score = metrics.get(monitor, float("nan"))

        # A NaN monitor means the metric was undefined this epoch (e.g. a
        # validation shard with a single class), NOT that the model failed to
        # improve. Stepping the scheduler or the early-stop counter on it would
        # decay the LR and burn patience for a measurement that never happened.
        score_valid = not math.isnan(score)
        warming_up = optimizer_step < warmup_steps

        # Do not let the plateau scheduler decay an LR that is still ramping —
        # it would read the warmup value as a plateau and cut it further.
        if score_valid and not warming_up:
            scheduler.step(score)
        elif score_valid:
            pass
        else:
            warnings.warn(
                f"monitor metric '{monitor}' was NaN at epoch {epoch}; skipping "
                f"LR schedule and early-stop accounting for this epoch.",
                stacklevel=2,
            )

        group_lrs = {
            group.get("name", f"group{i}"): group["lr"]
            for i, group in enumerate(optimizer.param_groups)
        }
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["components"].append(components_mean)
        history.setdefault("learning_rate", []).append(current_lr)
        history.setdefault("group_lrs", []).append(group_lrs)
        for key, value in metrics.items():
            history.setdefault(key, []).append(value)

        shares = effective_contributions(
            components_mean,
            loss_kwargs.get("loss_weights"),  # type: ignore
            loss_kwargs.get("loss_scales"),
        )
        history.setdefault("loss_shares", []).append(shares)

        lr_text = "  ".join(f"lr[{n}] {v:.2e}" for n, v in group_lrs.items())
        print(
            f"epoch {epoch:>3}  train {train_loss:.4f}  val {metrics['val_loss']:.4f}"
            f"  {lr_text}{'  (warmup)' if warming_up else ''}"
        )
        print(
            f"    diagnosis  AUC {metrics['diagnosis_auc']:.4f}  "
            f"AP {metrics['diagnosis_ap']:.4f}"
            f"   |  ordinal  AUC {metrics['ordinal_auc']:.4f}  "
            f"AP {metrics['ordinal_ap']:.4f}"
        )
        print(
            f"    findings   macroAP {metrics['findings_macro_ap']:.4f}  "
            f"microAP {metrics['findings_micro_ap']:.4f}  "
            f"macroAUC {metrics['findings_macro_auc']:.4f}"
        )

        findings_ap = metrics["findings_ap_per_class"]
        findings_auc = metrics["findings_auc_per_class"]
        findings_support = metrics["findings_support"]
        n_images = metrics.get("n_images", 0)

        print(
            f"      {'per class (by support)':<28}"
            f"{'AP':>8}{'AUC':>8}{'n':>6}{'AP/prev':>9}"
        )

        for name in sorted(findings_support, key=lambda k: -findings_support[k]):
            n_pos = findings_support[name]
            average_precision = findings_ap.get(name, float("nan"))
            area = findings_auc.get(name, float("nan"))
            if n_pos == 0 or not n_images:
                print(
                    f"      {str(name)[:28]:<28}{'--':>8}{'--':>8}"
                    f"{n_pos:>6}{'--':>9}   (no positives in split)"
                )
                continue
            prevalence = n_pos / n_images
            lift = average_precision / prevalence if prevalence > 0 else float("nan")
            print(
                f"      {str(name)[:28]:<28}{average_precision:>8.4f}"
                f"{area:>8.4f}{n_pos:>6}{lift:>9.2f}"
            )

        cutpoint_text = "  ".join(
            f"{name.replace('gt_', '>')} {value:.3f}"
            for name, value in metrics["suspicion_auc_per_cutpoint"].items()
        )
        print(
            f"    suspicion  QWK {metrics['suspicion_qwk']:.4f}"
            f"   cutpoint AUROC  {cutpoint_text}"
        )

        density_text = "  ".join(
            f"{name.replace('class_', '')} {value:.3f}"
            for name, value in metrics["density_f1_per_class"].items()
        )
        print(
            f"    density    QWK {metrics['density_qwk']:.4f}  "
            f"macroP {metrics['density_precision']:.3f}  "
            f"macroR {metrics['density_recall']:.3f}  "
            f"macroF1 {metrics['density_f1']:.3f}"
        )
        print(f"      {'per-class F1':<28}{density_text}")
        print(f"    age        MAE {metrics['age_mae_years']:.2f} yrs")
        print(
            "    loss share "
            + "  ".join(f"{task} {100 * share:.0f}%" for task, share in shares.items())
        )

        if wandb_run is not None:
            wandb_run.log(
                _flatten_for_wandb(metrics)
                | {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    **{f"train/lr/{n}": v for n, v in group_lrs.items()},
                    **{f"train/raw_loss/{k}": v for k, v in components_mean.items()},
                    **{f"train/loss_share/{k}": v for k, v in shares.items()},
                },
                step=epoch,
            )

        if score_valid and improved(score):
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0  # Reset early stopping counter on improvement

            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": best_state,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "score": best_score,
                        "monitor": monitor,
                        "loss_scales": loss_kwargs.get("loss_scales"),
                    },
                    checkpoint_path,
                )
        elif score_valid:
            early_stop_counter += 1

        # Early stopping mechanism to prevent overfitting
        if early_stop_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}: no improvement in {monitor} "
                f"for {patience} epochs (best {best_score:.4f} at epoch {best_epoch})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_score"] = best_score
    history["best_epoch"] = best_epoch
    history["monitor"] = monitor

    if wandb_run is not None:
        wandb_run.summary[f"best/{monitor}"] = best_score
        wandb_run.summary["best/epoch"] = best_epoch

    return history


def get_coral_cutpoint_weights(
    levels: np.ndarray | pd.Series,
    n_cutpoints: int,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    """
    Per-cutpoint `pos_weight` for `coral_loss`.

    Args:
        levels: 0-based ordinal training labels
        n_cutpoints: number of cutpoints (n_classes - 1)
        max_pos_weight: upper clip on neg/pos

    Returns:
        (n_cutpoints,) float tensor
    """
    levels = np.asarray(levels)
    total = len(levels)
    weights = []

    for k in range(n_cutpoints):
        positives = int((levels > k).sum())
        negatives = total - positives

        if positives == 0:
            weights.append(1.0)
            continue

        weights.append(min(negatives / positives, max_pos_weight))

    return torch.tensor(weights, dtype=torch.float32)


def get_findings_weights(
    findings_matrix: np.ndarray,
    reweight: str = "sqrt_inverse",
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Per-class weights for the multilabel findings head.

    Args:
        findings_matrix: (N, C) binary matrix of training labels
        reweight: "inverse" or "sqrt_inverse"
        max_weight: cap on the normalised weight, applied after normalisation

    Returns:
        (C,) float tensor, mean 1.0
    """
    prevalence = np.clip(findings_matrix.mean(axis=0), 1e-8, None)

    if reweight == "inverse":
        weights = 1.0 / prevalence
    elif reweight == "sqrt_inverse":
        weights = 1.0 / np.sqrt(prevalence)
    else:
        raise ValueError(f"Unknown reweight: {reweight}")

    weights = weights / weights.mean()
    weights = np.clip(weights, None, max_weight)
    weights = weights / weights.mean()  # renormalise after any clipping

    return torch.tensor(weights, dtype=torch.float32)


class MammoCNNDataset(Dataset):
    """
    Dataset class for first stage of MAMMO CNN training
    """

    def __init__(
        self,
        image_df: pd.DataFrame,
        transform: Compose | None = None,
        is_training: bool = False,
        findings_column: str = "finding_vector",
    ) -> None:
        self.image_df = image_df
        self.transform = transform
        self.is_training = is_training
        self.findings_column = findings_column

        required = {findings_column, "breast_birads", "breast_density", "age"}
        missing = required - set(image_df.columns)
        if missing:
            raise KeyError(f"missing column(s) {sorted(missing)}")

    def __len__(self) -> int:
        return self.image_df.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.image_df.iloc[index]
        image_path = row["image_path"]
        laterality = row["laterality"]
        diagnosis = row["diagnosis"]
        findings = row[self.findings_column]
        age = row["age"]

        # BI-RADS 1-5 and density 1-4 in the metadata; the ordinal heads index
        # from 0, so shift here. CORAL levels and CrossEntropyLoss both need
        # 0-based indices.
        suspicion = int(row["breast_birads"]) - BIRADS_MIN
        density = int(row["breast_density"]) - DENSITY_MIN

        image = preprocess_image(image_path, laterality)

        # Apply data augmentations
        if self.transform:
            image = self.transform(image)

        image = apply_clahe(image, is_training=self.is_training)

        # Apply random gaussian noise during training
        if self.is_training:
            image = image.float()
            noise = torch.randn_like(image) * 0.01
            image = torch.clamp(image + noise, 0.0, 1.0)

        # Standardize the image
        image = image.float()
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-8)

        return {
            "mammogram": image,
            "diagnosis": torch.tensor(diagnosis, dtype=torch.float32),  # [B,]
            "findings": torch.tensor(
                np.asarray(findings), dtype=torch.float32
            ),  # [B,10]
            "suspicion": torch.tensor(suspicion, dtype=torch.long),  # [B,] level 0..4
            "density": torch.tensor(density, dtype=torch.long),  # [B,] class 0..3
            "age": torch.tensor(age, dtype=torch.float32),  # [B,]
        }
