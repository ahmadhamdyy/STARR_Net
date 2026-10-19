"""Run a prediction from any trained comparison variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from starr_model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Query STARRFull crash risk")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-17", required=True)
    parser.add_argument("--image-18", required=True)
    parser.add_argument("--image-19", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--day-of-week", type=float, required=True, help="1=Sunday, 7=Saturday")
    parser.add_argument("--hour", type=float, required=True, help="Hour in [0, 24)")
    parser.add_argument("--mscm-checkpoint", default="")
    parser.add_argument("--geoclip-weights", default="")
    args = parser.parse_args()

    if not 1 <= args.day_of_week <= 7:
        parser.error("--day-of-week must be between 1 and 7")
    if not 0 <= args.hour < 24:
        parser.error("--hour must be in [0, 24)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_args = checkpoint.get("args", {})
    variant = training_args.get("variant", "starr_full")
    mscm_checkpoint = args.mscm_checkpoint or training_args.get("mscm_checkpoint")
    geoclip_weights = args.geoclip_weights or training_args.get("geoclip_weights")
    if not mscm_checkpoint:
        parser.error("Checkpoint has no MSCM path; pass --mscm-checkpoint.")
    if variant in ("starr_loc", "starr_full") and not geoclip_weights:
        parser.error(
            "This location-aware checkpoint requires --geoclip-weights."
        )

    model = build_model(
        variant=variant,
        mscm_checkpoint=mscm_checkpoint,
        geoclip_weights=geoclip_weights,
        dropout=float(training_args.get("dropout", 0.7)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    image_size = int(training_args.get("image_size", 768))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    images = [
        transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        for path in (args.image_17, args.image_18, args.image_19)
    ]
    location = torch.tensor(
        [[args.latitude, args.longitude]], dtype=torch.float32, device=device
    )
    temporal = torch.tensor(
        [[args.day_of_week, args.hour]], dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        output = model(images, location=location, temporal=temporal)

    result = {
        "variant": variant,
        "risk_mean": output["risk_score"].item(),
        "uncertainty": output["uncertainty"].item(),
    }
    if "alpha" in output:
        result.update(
            {
                "alpha": output["alpha"].item(),
                "beta": output["beta"].item(),
                "concentration": (output["alpha"] + output["beta"]).item(),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
