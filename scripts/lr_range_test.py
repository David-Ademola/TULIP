"""
Leslie Smith's LR range test — find the usable learning rate empirically.

Ramps the LR exponentially from `--min-lr` to `--max-lr` over a few hundred
optimizer steps and records the loss. The useful LR is roughly an order of
magnitude below where the loss starts diverging, or at the point of steepest
descent. Reference: Smith (2017), "Cyclical Learning Rates for Training Neural
Networks", section 3.3.

Worth running rather than extrapolating, because two changes pull the LR in
opposite directions at once:
  - a larger effective batch argues for a HIGHER LR (less gradient noise)
  - unfreezing a pretrained backbone argues for a LOWER one (large early
    gradients from random heads destroy ImageNet features)
Scaling rules cannot resolve that; a measurement can.

Usage
-----
    python -m scripts.lr_range_test
    python -m scripts.lr_range_test --backbone-lr-mult 1.0 --steps 300
"""

import argparse
import math

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import MammoCNN
from src.utils import (
    BIRADS_MIN,
    LOSS_WEIGHTS,
    MammoCNNDataset,
    calibrate_loss_scales,
    compute_multi_task_loss,
    get_coral_cutpoint_weights,
    get_findings_weights,
    get_lds_weights,
)

SEED = 42


def build_cnn_split(parquet: str) -> tuple[pd.DataFrame, float]:
    """Rebuild the CNN training split exactly as main.ipynb does."""
    metadata = pd.read_parquet(parquet)
    patients = (
        metadata[["study_id", "patient_cancer", "Patient's Age"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_val, _ = train_test_split(
        patients, test_size=0.12, random_state=SEED, stratify=patients.patient_cancer
    )
    rest, _ = train_test_split(
        train_val, test_size=0.10, random_state=SEED, stratify=train_val.patient_cancer
    )
    rest, _ = train_test_split(
        rest,
        test_size=round(15 / 90, 6),
        random_state=SEED,
        stratify=rest.patient_cancer,
    )
    cnn, _ = train_test_split(
        rest,
        test_size=round(15 / 75, 6),
        random_state=SEED,
        stratify=rest.patient_cancer,
    )

    frame = metadata[metadata.study_id.isin(cnn.study_id)].reset_index(drop=True)
    scaler = StandardScaler().fit(frame[["Patient's Age"]])
    frame["age"] = scaler.transform(frame[["Patient's Age"]])

    return frame, float(scaler.scale_[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="vindr-mammo/breast_metadata.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--max-lr", type=float, default=1e-1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--smooth", type=float, default=0.05, help="EMA factor")
    parser.add_argument("--out", default="lr_range_test.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cnn_df, _ = build_cnn_split(args.parquet)
    print(f"CNN split: {len(cnn_df)} images, {100 * cnn_df.diagnosis.mean():.2f}% positive")

    density_lds = get_lds_weights(
        (cnn_df.breast_density.to_numpy() - 1).tolist(),
        label_min=0,
        label_max=3,
        kernel_size=3,
        reweight="sqrt_inverse",
    )
    loss_kwargs = {
        "loss_weights": LOSS_WEIGHTS,
        "suspicion_weights": get_coral_cutpoint_weights(
            cnn_df.breast_birads.to_numpy() - BIRADS_MIN, n_cutpoints=4
        ),
        "findings_weights": get_findings_weights(
            np.stack(cnn_df.finding_vector.to_numpy())
        ),
        "density_weights": torch.tensor(
            [density_lds[i] for i in range(4)], dtype=torch.float32
        ),
    }

    loader = DataLoader(
        MammoCNNDataset(cnn_df, is_training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = MammoCNN(pretrained=True).to(device)
    loss_kwargs["loss_scales"] = calibrate_loss_scales(
        model, loader, device, loss_kwargs, n_batches=10
    )
    print("loss scales:", {k: round(v, 3) for k, v in loss_kwargs["loss_scales"].items()})

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids]
    backbone = [p for p in model.parameters() if id(p) in backbone_ids]

    optimizer = AdamW(
        [
            {"params": heads, "lr": args.min_lr},
            {"params": backbone, "lr": args.min_lr * args.backbone_lr_mult},
        ],
        weight_decay=1e-4,
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    gamma = (args.max_lr / args.min_lr) ** (1 / args.steps)
    records: list[dict] = []
    smoothed = None
    best = math.inf
    model.train()

    step = 0
    progress = tqdm(total=args.steps, desc="lr range test")

    while step < args.steps:
        for micro, batch in enumerate(loader):
            images = batch["mammogram"].to(device, non_blocking=True)
            targets = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k != "mammogram"
            }

            with torch.amp.autocast(device.type, enabled=amp):
                loss, _ = compute_multi_task_loss(model(images), targets, **loss_kwargs)

            scaler.scale(loss / args.accumulation_steps).backward()

            if (micro + 1) % args.accumulation_steps:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            lr = args.min_lr * (gamma**step)
            value = loss.item()
            smoothed = (
                value
                if smoothed is None
                else args.smooth * value + (1 - args.smooth) * smoothed
            )
            records.append({"step": step, "lr": lr, "loss": value, "smoothed": smoothed})
            best = min(best, smoothed)

            step += 1
            progress.update(1)
            progress.set_postfix(lr=f"{lr:.2e}", loss=f"{smoothed:.4f}")

            # Diverged — no information left in the tail
            if step >= args.steps or smoothed > 4 * best or math.isnan(smoothed):
                step = args.steps
                break

            for group, mult in zip(optimizer.param_groups, (1.0, args.backbone_lr_mult)):
                group["lr"] = args.min_lr * (gamma**step) * mult

    progress.close()

    frame = pd.DataFrame(records)
    frame.to_csv(args.out, index=False)

    # Steepest descent of the smoothed curve = fastest learning per unit log-LR
    frame["gradient"] = np.gradient(frame.smoothed, np.log10(frame.lr))
    steepest = frame.loc[frame.gradient.idxmin()]
    minimum = frame.loc[frame.smoothed.idxmin()]

    print()
    print(f"wrote {args.out} ({len(frame)} steps)")
    print(f"  steepest descent at lr = {steepest.lr:.2e}")
    print(f"  loss minimum at     lr = {minimum.lr:.2e}")
    print()
    print(f"  suggested head LR:     {steepest.lr:.1e}")
    print(f"  suggested backbone LR: {steepest.lr * args.backbone_lr_mult:.1e}")
    print()
    print("Rule of thumb: pick roughly an order of magnitude below the divergence")
    print("point, or the steepest-descent LR. Plot `loss` against `lr` on a log x")
    print("axis and sanity-check the shape before trusting either number.")


if __name__ == "__main__":
    main()
