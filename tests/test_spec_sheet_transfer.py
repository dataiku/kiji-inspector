from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def transfer():
    directory = Path(__file__).parents[1] / "demo" / "spec_sheet"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "evaluate_transfer", directory / "evaluate_transfer.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_explained_variance_bounds(transfer):
    x = torch.randn(50, 8)
    assert transfer.explained_variance(x, x.clone()) == pytest.approx(1.0)
    mean_prediction = x.mean(dim=0, keepdim=True).expand_as(x)
    assert transfer.explained_variance(x, mean_prediction) == pytest.approx(0.0, abs=1e-6)
    assert transfer.explained_variance(x, torch.zeros_like(x)) < 1.0


def test_affine_align_matches_target_statistics(transfer):
    torch.manual_seed(0)
    x = torch.randn(200, 6) * 3.0 + 5.0
    mean_to = torch.randn(1, 6)
    scale_to = 0.5
    aligned = transfer.affine_align(x, mean_to, scale_to)
    assert torch.allclose(aligned.mean(dim=0, keepdim=True), mean_to, atol=1e-5)
    assert transfer.rms_scale_of(aligned) == pytest.approx(scale_to, abs=1e-5)


def test_explained_variance_is_affine_invariant(transfer):
    torch.manual_seed(1)
    x = torch.randn(100, 5)
    reconstruction = x + 0.1 * torch.randn(100, 5)
    ev_raw = transfer.explained_variance(x, reconstruction)
    shift, scale = torch.randn(1, 5), 2.5
    ev_affine = transfer.explained_variance(x * scale + shift, reconstruction * scale + shift)
    assert ev_affine == pytest.approx(ev_raw, abs=1e-5)


def test_pca_recovers_low_rank_data(transfer):
    torch.manual_seed(2)
    basis = torch.randn(3, 10)
    x = torch.randn(300, 3) @ basis + torch.randn(1, 10)
    mean, components = transfer.pca_fit(x, k=3)
    assert components.shape == (3, 10)
    ev = transfer.explained_variance(x, transfer.pca_reconstruct(x, mean, components))
    assert ev > 0.999
    # rank-1 misses variance
    mean1, components1 = transfer.pca_fit(x, k=1)
    assert transfer.explained_variance(x, transfer.pca_reconstruct(x, mean1, components1)) < ev


def test_match_features_identity_and_null(transfer):
    torch.manual_seed(3)
    decoder = torch.randn(40, 64)
    identical = transfer.match_features(decoder, decoder)
    assert identical["meanMaxCosine"] == pytest.approx(1.0)
    assert identical["fracAtLeast09"] == 1.0
    noise = transfer.match_features(decoder, torch.randn(40, 64))
    assert noise["meanMaxCosine"] < 0.6
    assert noise["fracAtLeast09"] == 0.0
    permuted = transfer.match_features(decoder, decoder[torch.randperm(40)])
    assert permuted["meanMaxCosine"] == pytest.approx(1.0)


def test_functional_match_identity_and_permutation_null(transfer):
    torch.manual_seed(4)
    features = torch.rand(120, 30)
    identical = transfer.functional_match(features, features)
    assert identical["meanBestCorr"] > 0.999
    assert identical["fracAtLeast09"] == 1.0
    shuffled = transfer.functional_match(features, features[torch.randperm(120)])
    assert shuffled["meanBestCorr"] < 0.5
