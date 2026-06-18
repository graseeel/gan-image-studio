from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FidResult:
    fid: float
    sample_count: int
    representative: bool
    warning: str | None = None


def fid_from_features(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    *,
    representative_threshold: int = 1000,
) -> FidResult:
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("FID features must be rank-2 tensors")
    if real_features.size(1) != fake_features.size(1):
        raise ValueError("real and fake features must have the same width")
    if real_features.size(0) < 2 or fake_features.size(0) < 2:
        raise ValueError("FID requires at least two real and fake samples")

    real_mean, real_cov = _mean_and_covariance(real_features.float())
    fake_mean, fake_cov = _mean_and_covariance(fake_features.float())
    mean_distance = torch.sum((real_mean - fake_mean) ** 2)
    covmean = _matrix_sqrt_psd(real_cov @ fake_cov)
    fid = mean_distance + torch.trace(real_cov + fake_cov - 2.0 * covmean)
    sample_count = min(real_features.size(0), fake_features.size(0))
    representative = sample_count >= representative_threshold
    warning = None
    if not representative:
        warning = (
            f"FID computed with {sample_count} samples is useful for regression checks, "
            "but not representative of final image quality."
        )
    return FidResult(
        fid=float(torch.real(fid).clamp_min(0).item()),
        sample_count=sample_count,
        representative=representative,
        warning=warning,
    )


@torch.no_grad()
def extract_inception_features(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    images = (images.clamp(-1, 1) + 1.0) / 2.0
    resized = F.interpolate(
        images.to(device),
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )
    normalized = weights.transforms()(resized)
    features = model(normalized)
    if isinstance(features, tuple):
        features = features[0]
    return features.detach().cpu()


def _mean_and_covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.size(0) - 1)
    return mean, covariance


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    clipped = torch.clamp(eigenvalues, min=0)
    return (eigenvectors * torch.sqrt(clipped).unsqueeze(0)) @ eigenvectors.T
