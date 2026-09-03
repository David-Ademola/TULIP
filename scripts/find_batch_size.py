"""
Find the largest batch size that fits on this GPU, for a given image size.

Usage
-----
    python -m scripts.find_batch_size
    python -m scripts.find_batch_size --height 1280 --width 1024
    python -m scripts.find_batch_size --no-amp --max-batch 512

Runs a real forward + backward + optimizer step, because inference-only probing
badly underestimates memory: gradients and the two AdamW moment buffers are
absent, and activation storage for backward is not held.
"""

import argparse

import torch
from torch.optim import AdamW

from src.model import MammoCNN
from src.utils import compute_multi_task_loss


def make_batch(
    batch_size: int, height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Synthetic batch with the same shapes and dtypes the real loader emits."""
    images = torch.randn(batch_size, 3, height, width, device=device)
    targets = {
        "diagnosis": torch.randint(0, 2, (batch_size,), device=device).float(),
        "findings": torch.randint(0, 2, (batch_size, 10), device=device).float(),
        "suspicion": torch.randint(0, 5, (batch_size,), device=device),
        "density": torch.randint(0, 4, (batch_size,), device=device),
        "age": torch.randn(batch_size, device=device),
    }
    return images, targets


def try_batch_size(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    use_amp: bool,
    freeze_backbone: bool,
) -> tuple[bool, float]:
    """
    Attempt one full training step.

    Returns:
        (fitted, peak_memory_gb)
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = MammoCNN(pretrained=False).to(device)
    if freeze_backbone:
        model.freeze_backbone()
    model.train()

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    try:
        # Two steps: the first allocates, the second is where AdamW's exp_avg
        # and exp_avg_sq buffers exist, which is the real steady-state peak.
        for _ in range(2):
            images, targets = make_batch(batch_size, height, width, device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                loss, _ = compute_multi_task_loss(model(images), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated() / 1e9
        return True, peak

    except torch.OutOfMemoryError:
        return False, float("nan")

    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true", help="measure in fp32")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="measure the heads-only warmup stage instead of full fine-tuning",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — this script measures GPU memory")

    device = torch.device("cuda")
    use_amp = not args.no_amp
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"device:      {torch.cuda.get_device_name(0)} ({total_gb:.0f} GB)")
    print(f"image size:  {args.height} x {args.width}")
    print(f"precision:   {'AMP' if use_amp else 'fp32'}")
    print(f"backbone:    {'frozen' if args.freeze_backbone else 'trainable'}")
    print()

    # Double until failure, then bisect — O(log n) full training steps.
    largest_ok, smallest_bad = 0, None
    batch_size = 8

    while batch_size <= args.max_batch:
        fitted, peak = try_batch_size(
            batch_size, args.height, args.width, device, use_amp, args.freeze_backbone
        )
        status = f"OK   peak {peak:5.1f} GB" if fitted else "OOM"
        print(f"  batch {batch_size:>4}: {status}")

        if not fitted:
            smallest_bad = batch_size
            break

        largest_ok = batch_size
        batch_size *= 2

    if smallest_bad is not None:
        low, high = largest_ok, smallest_bad
        while high - low > 1:
            mid = (low + high) // 2
            fitted, peak = try_batch_size(
                mid, args.height, args.width, device, use_amp, args.freeze_backbone
            )
            status = f"OK   peak {peak:5.1f} GB" if fitted else "OOM"
            print(f"  batch {mid:>4}: {status}")
            if fitted:
                low = mid
            else:
                high = mid
        largest_ok = low

    print()
    print(f"largest batch that fits: {largest_ok}")
    # Fragmentation and a real dataloader's pinned buffers eat into the margin,
    # so do not train at the absolute ceiling.
    print(f"recommended for training: {int(largest_ok * 0.8)}  (20% headroom)")

    positive_rate = 0.0495  # VinDr-Mammo, CNN training split
    safe = int(largest_ok * 0.8)
    print()
    print(
        f"at {100 * positive_rate:.2f}% positives, P(no positive in a batch of {safe}) "
        f"= {(1 - positive_rate) ** safe:.4f}"
    )
    print("  -> ACCUMULATION_STEPS = 1 is fine once this is below ~0.05")


if __name__ == "__main__":
    main()
