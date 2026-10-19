"""Models used in the paper's headline comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from geoclip.model.location_encoder import LocationEncoder


def _unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The MSCM checkpoint must contain a state dictionary.")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    return {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def load_mscm_resnet50(checkpoint_path: str | Path) -> nn.Sequential:
    """Load the ResNet-50 image trunk from an MSCM/BetaRisk checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"MSCM pretrained checkpoint not found: {path}")

    state = _unwrap_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    backbone = models.resnet50(weights=None)

    # Original pretraining checkpoints store the backbone below ``ft_img``.
    backbone_state = {}
    sequential_prefixes = {
        "0.": "conv1.",
        "1.": "bn1.",
        "4.": "layer1.",
        "5.": "layer2.",
        "6.": "layer3.",
        "7.": "layer4.",
    }
    for key, value in state.items():
        if key.startswith("ft_img."):
            image_key = key[len("ft_img.") :]
            for source, destination in sequential_prefixes.items():
                if image_key.startswith(source):
                    image_key = destination + image_key[len(source) :]
                    break
            backbone_state[image_key] = value
        elif key in backbone.state_dict() and not key.startswith("fc."):
            backbone_state[key] = value

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    required_missing = [
        key
        for key in missing
        if not key.startswith("fc.")
    ]
    if required_missing or unexpected:
        raise RuntimeError(
            "MSCM checkpoint is incompatible with a ResNet-50 trunk. "
            f"Missing={required_missing}, unexpected={unexpected}"
        )
    return nn.Sequential(*list(backbone.children())[:-2])


class CyclicTimeEncoder(nn.Module):
    """Encode day-of-week and hour-of-day with cyclic features and an MLP."""

    def __init__(self, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        temporal = temporal.float()
        day, hour = temporal.unbind(dim=-1)
        features = torch.stack(
            (
                torch.sin(2.0 * math.pi * day / 7.0),
                torch.cos(2.0 * math.pi * day / 7.0),
                torch.sin(2.0 * math.pi * hour / 24.0),
                torch.cos(2.0 * math.pi * hour / 24.0),
            ),
            dim=-1,
        )
        return self.mlp(features)


VARIANTS = ("mscm_ms", "betarisk", "starr_loc", "starr_time", "starr_full")


class STARRModel(nn.Module):
    """Shared image model with optional GeoCLIP, time, and probabilistic heads."""

    def __init__(
        self,
        mscm_checkpoint: str | Path,
        geoclip_weights: str | Path | None = None,
        use_location: bool = True,
        use_time: bool = True,
        probabilistic: bool = True,
        dropout: float = 0.7,
        temporal_hidden_dim: int = 64,
        temporal_embedding_dim: int = 32,
        fusion_hidden_dim: int = 512,
        freeze_location_encoder: bool = True,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.use_location = use_location
        self.use_time = use_time
        self.probabilistic = probabilistic
        self.image_encoder = load_mscm_resnet50(mscm_checkpoint)
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(2048 * 3, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.location_encoder: nn.Module | None = None
        if use_location:
            if geoclip_weights is None:
                raise ValueError("A location-aware variant requires GeoCLIP weights.")
            geoclip_path = Path(geoclip_weights)
            if not geoclip_path.is_file():
                raise FileNotFoundError(
                    f"GeoCLIP location weights not found: {geoclip_path}. "
                    "Pass --geoclip-weights to location_encoder_weights.pth."
                )
            self.location_encoder = LocationEncoder(from_pretrained=False)
            location_state = _unwrap_state_dict(
                torch.load(geoclip_path, map_location="cpu", weights_only=False)
            )
            self.location_encoder.load_state_dict(location_state, strict=True)
            if freeze_location_encoder:
                self.location_encoder.requires_grad_(False)

        self.time_encoder: nn.Module | None = None
        if use_time:
            self.time_encoder = CyclicTimeEncoder(
                hidden_dim=temporal_hidden_dim,
                output_dim=temporal_embedding_dim,
            )
        fused_dim = (
            2048
            + (512 if use_location else 0)
            + (temporal_embedding_dim if use_time else 0)
        )
        if probabilistic:
            self.fusion_head = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            output_dim = fusion_hidden_dim
            self.alpha_head: nn.Module | None = nn.Linear(output_dim, 1)
            self.beta_head: nn.Module | None = nn.Linear(output_dim, 1)
        else:
            # MSCM-MS classifies the pooled multi-scale image feature directly.
            self.fusion_head = nn.Identity()
            output_dim = fused_dim
            self.alpha_head = None
            self.beta_head = None
        self.classifier = nn.Linear(output_dim, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.location_encoder is not None and not any(
            parameter.requires_grad for parameter in self.location_encoder.parameters()
        ):
            self.location_encoder.eval()
        return self

    def forward(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        location: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(images) != 3:
            raise ValueError(f"STARR expects three image scales, received {len(images)}")
        image_maps = [self.image_encoder(image) for image in images]
        image_features = self.channel_fusion(torch.cat(image_maps, dim=1)).flatten(1)

        streams = [image_features]
        if self.use_location:
            if location is None or self.location_encoder is None:
                raise ValueError("This variant requires location coordinates.")
            location = location.to(device=image_features.device, dtype=torch.float32)
            if not torch.isfinite(location).all():
                raise ValueError("Location contains NaN or infinite values.")
            streams.append(self.location_encoder(location).to(image_features.dtype))
        if self.use_time:
            if temporal is None or self.time_encoder is None:
                raise ValueError("This variant requires temporal features.")
            streams.append(
                self.time_encoder(
                    temporal.to(device=image_features.device, dtype=torch.float32)
                ).to(image_features.dtype)
            )

        features = self.fusion_head(torch.cat(streams, dim=1))
        classification_logit = self.classifier(features).squeeze(-1)
        if not self.probabilistic:
            risk = torch.sigmoid(classification_logit)
            return {
                "features": features,
                "risk_score": risk,
                "uncertainty": (risk * (1.0 - risk)).clamp_min(0.0).sqrt(),
                "classification_logit": classification_logit,
            }

        if self.alpha_head is None or self.beta_head is None:
            raise RuntimeError("Probabilistic heads were not initialized.")
        alpha = F.softplus(self.alpha_head(features)) + self.epsilon
        beta = F.softplus(self.beta_head(features)) + self.epsilon
        risk = alpha / (alpha + beta)
        concentration = alpha + beta
        variance = alpha * beta / (
            concentration.square() * (concentration + 1.0)
        )
        return {
            "features": features,
            "alpha": alpha.squeeze(-1),
            "beta": beta.squeeze(-1),
            "risk_score": risk.squeeze(-1),
            "uncertainty": variance.clamp_min(0.0).sqrt().squeeze(-1),
            "classification_logit": classification_logit,
        }

    @torch.no_grad()
    def predict(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        location: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward(images, location, temporal)


class STARRFull(STARRModel):
    """Backward-compatible name for the paper's full model."""

    def __init__(
        self,
        mscm_checkpoint: str | Path,
        geoclip_weights: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(
            mscm_checkpoint=mscm_checkpoint,
            geoclip_weights=geoclip_weights,
            use_location=True,
            use_time=True,
            probabilistic=True,
            **kwargs,
        )


def build_model(
    variant: str,
    mscm_checkpoint: str | Path,
    geoclip_weights: str | Path | None,
    **kwargs,
) -> STARRModel:
    """Construct one model from Table 1 with a shared image backbone."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    settings = {
        "mscm_ms": (False, False, False),
        "betarisk": (False, False, True),
        "starr_loc": (True, False, True),
        "starr_time": (False, True, True),
        "starr_full": (True, True, True),
    }
    use_location, use_time, probabilistic = settings[variant]
    return STARRModel(
        mscm_checkpoint=mscm_checkpoint,
        geoclip_weights=geoclip_weights if use_location else None,
        use_location=use_location,
        use_time=use_time,
        probabilistic=probabilistic,
        **kwargs,
    )
