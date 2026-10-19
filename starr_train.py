"""Train any model from the paper's headline comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from mydatasets_temporal import MultiScaleProbabilisticTemporal
from starr_model import VARIANTS, build_model
from wasserstein_beta_loss import WassersteinBetaLoss


DEFAULT_DATA_ROOT = (
    "/scratch/group/p.cis230407.000/Major-Metro-Central-TX_1024x1024"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a STARR comparison model")
    parser.add_argument("--variant", choices=VARIANTS, default="starr_full")
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
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a saved checkpoint and write val/test metrics without training.",
    )
    parser.add_argument(
        "--eval-checkpoint",
        default="",
        help="Checkpoint for --eval-only (default: output-dir/best.pth, else last.pth).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_dict_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    return (model.module if hasattr(model, "module") else model).state_dict()


def make_dataset(
    split: str,
    args: argparse.Namespace,
    transform: transforms.Compose,
) -> MultiScaleProbabilisticTemporal:
    return MultiScaleProbabilisticTemporal(
        split=split,
        data_path=args.data_root,
        time_csv_root=args.time_csv_root,
        temporal_seed=args.seed,
        basic_transform=transform,
        pos_metadata=f"metadata/{split}_pos.csv",
        neg_metadata=f"metadata/{split}_neg.csv",
        test=split != "train",
        base_certainty_K=21.0,
        epsilon_beta=1e-5,
        min_positive_risk_mean=0.15,
        min_concentration_positives=18.0,
        large_crop_ratio=0.0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    distribution_loss: WassersteinBetaLoss,
    device: torch.device,
    probabilistic: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "distribution_loss": 0.0, "classification_loss": 0.0}
    labels_all, risks_all, uncertainties_all = [], [], []

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        images = [
            image.to(device, non_blocking=True)
            for image in batch["aug" if training else "normal"]
        ]
        location = batch["location"].to(device, non_blocking=True)
        temporal = batch["temporal"].to(device, non_blocking=True)
        alpha_target = batch["alpha_target"].to(device, non_blocking=True)
        beta_target = batch["beta_target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            output = model(images, location=location, temporal=temporal)
            if probabilistic:
                loss_distribution = distribution_loss(
                    output["alpha"],
                    output["beta"],
                    alpha_target,
                    beta_target,
                    labels,
                )
            else:
                loss_distribution = output["risk_score"].new_zeros(())
            loss_classification = nn.functional.binary_cross_entropy_with_logits(
                output["classification_logit"],
                labels.float(),
            )
            loss = (
                5.0 * loss_distribution + loss_classification
                if probabilistic
                else loss_classification
            )

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        count = labels.numel()
        totals["loss"] += loss.detach().item() * count
        totals["distribution_loss"] += loss_distribution.detach().item() * count
        totals["classification_loss"] += loss_classification.detach().item() * count
        labels_all.append(labels.detach().cpu())
        risks_all.append(output["risk_score"].detach().cpu())
        uncertainties_all.append(output["uncertainty"].detach().cpu())

    labels_tensor = torch.cat(labels_all).float()
    risks_tensor = torch.cat(risks_all).float()
    uncertainties_tensor = torch.cat(uncertainties_all).float()
    brier = torch.mean((risks_tensor - labels_tensor).square()).item()
    if labels_tensor.std() > 0 and risks_tensor.std() > 0:
        risk_correlation = torch.corrcoef(
            torch.stack((risks_tensor, labels_tensor))
        )[0, 1].item()
    else:
        risk_correlation = 0.0
    order = torch.argsort(risks_tensor)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(order), dtype=torch.float32)
    positives = labels_tensor.sum()
    negatives = len(labels_tensor) - positives
    auc = (
        (
            ranks[labels_tensor.bool()].sum()
            - positives * (positives - 1.0) / 2.0
        )
        / (positives * negatives)
    ).item() if positives > 0 and negatives > 0 else float("nan")
    predicted = risks_tensor >= 0.5
    true_positive = (predicted & labels_tensor.bool()).sum().float()
    false_positive = (predicted & ~labels_tensor.bool()).sum().float()
    false_negative = (~predicted & labels_tensor.bool()).sum().float()
    f1 = (
        2.0 * true_positive
        / (2.0 * true_positive + false_positive + false_negative).clamp_min(1.0)
    ).item()
    recall = (true_positive / positives.clamp_min(1.0)).item()
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, 16)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (risks_tensor >= lower) & (
            (risks_tensor < upper) if upper < 1.0 else (risks_tensor <= upper)
        )
        if in_bin.any():
            ece += (
                in_bin.float().mean()
                * (risks_tensor[in_bin].mean() - labels_tensor[in_bin].mean()).abs()
            ).item()
    confidence_order = torch.argsort(uncertainties_tensor)
    selective_errors = (
        risks_tensor[confidence_order] - labels_tensor[confidence_order]
    ).square()
    selective_curve = selective_errors.cumsum(0) / torch.arange(
        1, len(selective_errors) + 1, dtype=torch.float32
    )
    selective_ausc = selective_curve.mean().item()
    size = len(labels_tensor)
    return {
        **{key: value / size for key, value in totals.items()},
        "brier": brier,
        "auc": auc,
        "ece": ece,
        "selective_ausc": selective_ausc,
        "f1": f1,
        "recall": recall,
        "risk_correlation": risk_correlation,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_score: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": state_dict_for_save(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "args": vars(args),
        },
        path,
    )


def resolve_eval_checkpoint(output_dir: Path, args: argparse.Namespace) -> Path:
    candidates = []
    if args.eval_checkpoint:
        candidates.append(Path(args.eval_checkpoint))
    candidates.extend(
        [
            output_dir / "best.pth",
            output_dir / "last.pth",
            Path(args.resume) if args.resume else None,
        ]
    )
    for path in candidates:
        if path is not None and path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"No evaluation checkpoint found. Looked for: {searched}")


def write_metrics(
    path: Path,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(
        args.output_dir or ROOT / "runs" / f"{args.variant}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    splits = ("val", "test") if args.eval_only else ("train", "val", "test")
    datasets = {
        split: make_dataset(split, args, transform)
        for split in splits
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }

    model = build_model(
        variant=args.variant,
        mscm_checkpoint=args.mscm_checkpoint,
        geoclip_weights=args.geoclip_weights,
        dropout=args.dropout,
        freeze_location_encoder=True,
    )
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.no_data_parallel:
        model = nn.DataParallel(model)
    model.to(device)

    distribution_loss = WassersteinBetaLoss(
        reduction="mean", eps=1e-8, alpha_min=0.005, beta_min=0.0001
    )
    probabilistic = args.variant != "mscm_ms"

    if args.eval_only:
        checkpoint_path = resolve_eval_checkpoint(output_dir, args)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(
            checkpoint["model"], strict=True
        )
        extra = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "variant": args.variant,
        }
        print(f"Evaluating {args.variant} from {checkpoint_path} (epoch={extra['checkpoint_epoch']})")
        val_metrics = run_epoch(
            model, loaders["val"], distribution_loss, device, probabilistic
        )
        test_metrics = run_epoch(
            model, loaders["test"], distribution_loss, device, probabilistic
        )
        write_metrics(output_dir / "val_metrics.json", val_metrics, extra)
        write_metrics(output_dir / "test_metrics.json", test_metrics, extra)
        print("val:", json.dumps(val_metrics, sort_keys=True))
        print("test:", json.dumps(test_metrics, sort_keys=True))
        return

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    start_epoch = 0
    best_score = (-math.inf, -math.inf)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(
            checkpoint["model"], strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = tuple(checkpoint["best_score"])

    writer = SummaryWriter(output_dir / "tensorboard")
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            distribution_loss,
            device,
            probabilistic,
            optimizer,
            scaler,
        )
        val_metrics = run_epoch(
            model, loaders["val"], distribution_loss, device, probabilistic
        )
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                writer.add_scalar(f"{split}/{name}", value, epoch)

        score = (-val_metrics["brier"], val_metrics["risk_correlation"])
        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scaler,
            epoch,
            max(best_score, score),
            args,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pth",
                model,
                optimizer,
                scaler,
                epoch,
                best_score,
                args,
            )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
            f"val_brier={val_metrics['brier']:.5f} "
            f"val_risk_corr={val_metrics['risk_correlation']:.5f}"
        )

    best = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    (model.module if hasattr(model, "module") else model).load_state_dict(
        best["model"], strict=True
    )
    test_metrics = run_epoch(
        model, loaders["test"], distribution_loss, device, probabilistic
    )
    write_metrics(
        output_dir / "test_metrics.json",
        test_metrics,
        {
            "checkpoint": str(output_dir / "best.pth"),
            "checkpoint_epoch": int(best.get("epoch", -1)),
            "variant": args.variant,
        },
    )
    print("test:", json.dumps(test_metrics, sort_keys=True))
    writer.close()


if __name__ == "__main__":
    main()
