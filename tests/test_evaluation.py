import pytest
import torch

from gan_image_studio.evaluation import fid_from_features


def test_fid_warns_when_sample_count_is_small() -> None:
    features = torch.randn(8, 16)

    result = fid_from_features(features, features.clone(), representative_threshold=100)

    assert result.fid == pytest.approx(0.0, abs=1e-4)
    assert not result.representative
    assert result.warning is not None


def test_fid_rejects_mismatched_feature_widths() -> None:
    with pytest.raises(ValueError):
        fid_from_features(torch.randn(8, 16), torch.randn(8, 8))
