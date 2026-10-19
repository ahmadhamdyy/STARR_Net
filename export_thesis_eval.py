"""Export test predictions through the same path used by the evaluation loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from starr_train import DEFAULT_DATA_ROOT, ROOT, make_dataset, seed_everything
from utils.starr_model import VARIANTS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--time-csv-root", default=str(ROOT / "time_data"))
    parser.add_argument(
        "--mscm-checkpoint",
        default=str(ROOT / "ckpts/pre-trained_weights/pre-trained_weights_mscm.pth"),
    )
    parser.add_argument(
        "--geoclip-weights",
        default=str(ROOT / "geoclip/model/weights/location_encoder_weights.pth"),
    )
    parser.add_argument("--runs-root", default=str(ROOT / "runs"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "thesis_eval"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-data-parallel", action="store_true")
    return parser.parse_args()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    probabilistic: bool,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "lat",
            "lon",
            "label",
            "day",
            "hour",
            "risk",
            "uncertainty",
            "alpha",
            "beta",
        )
    }
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = [
                image.to(device, non_blocking=True) for image in batch["normal"]
            ]
            location = batch["location"].to(device, non_blocking=True)
            temporal = batch["temporal"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(images, location=location, temporal=temporal)

            count = len(batch["label"])
            fields["lat"].append(location[:, 0].detach().cpu())
            fields["lon"].append(location[:, 1].detach().cpu())
            fields["label"].append(batch["label"].detach().cpu())
            fields["day"].append(temporal[:, 0].detach().cpu())
            fields["hour"].append(temporal[:, 1].detach().cpu())
            fields["risk"].append(output["risk_score"].detach().cpu())
            fields["uncertainty"].append(output["uncertainty"].detach().cpu())
            if probabilistic:
                fields["alpha"].append(output["alpha"].detach().cpu())
                fields["beta"].append(output["beta"].detach().cpu())
            else:
                nan = torch.full((count,), float("nan"))
                fields["alpha"].append(nan)
                fields["beta"].append(nan.clone())

            if batch_index % 100 == 0:
                print(f"  exported {batch_index * loader.batch_size:,} samples")

    return {
        "lat": torch.cat(fields["lat"]).numpy().astype(np.float32),
        "lon": torch.cat(fields["lon"]).numpy().astype(np.float32),
        "label": torch.cat(fields["label"]).numpy().astype(np.int64),
        "day": torch.cat(fields["day"]).numpy().astype(np.float32),
        "hour": torch.cat(fields["hour"]).numpy().astype(np.float32),
        "risk": torch.cat(fields["risk"]).numpy().astype(np.float32),
        "uncertainty": torch.cat(fields["uncertainty"]).numpy().astype(np.float32),
        "alpha": torch.cat(fields["alpha"]).numpy().astype(np.float32),
        "beta": torch.cat(fields["beta"]).numpy().astype(np.float32),
    }


def export_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
    reference: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    seed_everything(args.seed)
    run_dir = Path(args.runs_root) / f"{variant}_seed{args.seed}"
    with (run_dir / "test_metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    checkpoint_path = Path(metrics.get("checkpoint", run_dir / "best.pth"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = make_dataset("test", args, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(
        variant=variant,
        mscm_checkpoint=args.mscm_checkpoint,
        geoclip_weights=args.geoclip_weights,
        dropout=args.dropout,
        freeze_location_encoder=True,
    )
    if (
        device.type == "cuda"
        and torch.cuda.device_count() > 1
        and not args.no_data_parallel
    ):
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    (model.module if hasattr(model, "module") else model).load_state_dict(
        checkpoint["model"], strict=True
    )

    print(f"{variant}: {checkpoint_path} (epoch {checkpoint.get('epoch', -1)})")
    values = collect_predictions(model, loader, device, variant != "mscm_ms")
    values["checkpoint_epoch"] = np.asarray(
        int(checkpoint.get("epoch", -1)), dtype=np.int64
    )

    if len(values["label"]) != 15_981 or int(values["label"].sum()) != 3_216:
        raise RuntimeError(f"{variant}: unexpected test-set composition")
    if reference is not None:
        for field in ("lat", "lon", "label", "day", "hour"):
            if not np.array_equal(values[field], reference[field]):
                raise RuntimeError(f"{variant}: unaligned field {field}")

    brier = float(np.mean(np.square(values["risk"] - values["label"])))
    order = np.argsort(values["uncertainty"], kind="stable")
    error = np.square(values["risk"][order] - values["label"][order])
    ausc = float(np.mean(np.cumsum(error) / np.arange(1, len(error) + 1)))
    print(
        f"  brier={brier:.9f} (JSON {metrics['brier']:.9f}, "
        f"delta={brier - metrics['brier']:+.2e})"
    )
    print(
        f"  ausc={ausc:.9f} (JSON {metrics['selective_ausc']:.9f}, "
        f"delta={ausc - metrics['selective_ausc']:+.2e})"
    )
    if abs(brier - float(metrics["brier"])) > 5e-5:
        raise RuntimeError(f"{variant}: exported Brier does not match evaluation loop")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{variant}_test_preds.npz"
    np.savez_compressed(path, **values)
    print(f"  wrote {path}")
    return values


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    reference = None
    for variant in variants:
        values = export_variant(variant, args, device, reference)
        if reference is None:
            reference = values


if __name__ == "__main__":
    main()
