"""
Unit tests for the MRMC core-set selection algorithm.
Uses synthetic data — no real datasets are downloaded.

Run with:  python -m pytest test_mrmc.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mrmc_original import (
    fit_mrmc_criterion,
    neg_exp_model,
    train_proxy_model,
    LinearProxyModel,
    select_coreset,
    evaluate,
)


DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_toy_dataset(N=200, num_classes=4, seed=0):
    """Tiny synthetic image-like dataset (no real images needed)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    images = torch.randn(N, 3, 8, 8, generator=rng)
    labels = torch.randint(0, num_classes, (N,), generator=rng)
    return TensorDataset(images, labels)


def make_tiny_resnet(num_classes=4):
    """A minimal conv-net that accepts 8x8 inputs."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )


# ---------------------------------------------------------------------------
# neg_exp_model
# ---------------------------------------------------------------------------

class TestNegExpModel:
    def test_exact_fit(self):
        """Known parameters should reconstruct the sequence exactly."""
        q, w = 2.5, 1.3
        epochs = np.arange(1, 11, dtype=float)
        expected = q * w ** (-epochs)
        result = neg_exp_model(epochs, q, w)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_decreasing(self):
        """Loss should decrease monotonically for w > 1."""
        q, w = 1.0, 1.5
        epochs = np.arange(1, 21, dtype=float)
        vals = neg_exp_model(epochs, q, w)
        assert np.all(np.diff(vals) < 0)

    def test_constant_for_w_equals_one(self):
        """w == 1 means no decay — all values equal q."""
        q, w = 0.8, 1.0
        epochs = np.arange(1, 6, dtype=float)
        vals = neg_exp_model(epochs, q, w)
        np.testing.assert_allclose(vals, q, rtol=1e-6)


# ---------------------------------------------------------------------------
# fit_mrmc_criterion
# ---------------------------------------------------------------------------

class TestFitMRMCCriterion:
    def _make_sequences(self, N=20, R=10, seed=1):
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.5, 2.0, size=N)
        w = rng.uniform(1.1, 2.0, size=N)
        epochs = np.arange(1, R + 1, dtype=float)
        sequences = q[:, None] * w[:, None] ** (-epochs[None, :])
        sequences += rng.normal(0, 0.01, size=sequences.shape)  # tiny noise
        sequences = np.clip(sequences, 0, None)
        return sequences, q, w, R

    def test_output_shape(self):
        N, R = 30, 8
        seqs = np.random.rand(N, R)
        scores = fit_mrmc_criterion(seqs)
        assert scores.shape == (N,)

    def test_scores_non_negative(self):
        N, R = 20, 10
        seqs, _, _, _ = self._make_sequences(N, R)
        scores = fit_mrmc_criterion(seqs)
        assert np.all(scores >= 0)

    def test_higher_drop_yields_higher_score(self):
        """
        A sample whose loss drops a lot should have a higher MRMC score
        than one whose loss barely moves.
        """
        R = 10
        epochs = np.arange(1, R + 1, dtype=float)
        # Large drop: q=2, w=2
        big_drop = (2.0 * 2.0 ** (-epochs))[None, :]
        # Small drop: q=0.1, w=1.01
        small_drop = (0.1 * 1.01 ** (-epochs))[None, :]
        seqs = np.vstack([big_drop, small_drop])
        scores = fit_mrmc_criterion(seqs)
        assert scores[0] > scores[1]

    def test_fallback_on_bad_data(self):
        """All-zero loss sequence should not raise and should return 0."""
        seqs = np.zeros((5, 8))
        scores = fit_mrmc_criterion(seqs)
        assert scores.shape == (5,)
        np.testing.assert_allclose(scores, 0.0, atol=1e-6)

    def test_single_sample(self):
        R = 10
        seqs = np.linspace(1.0, 0.1, R)[None, :]
        scores = fit_mrmc_criterion(seqs)
        assert scores.shape == (1,)
        assert scores[0] >= 0


# ---------------------------------------------------------------------------
# LinearProxyModel
# ---------------------------------------------------------------------------

class TestLinearProxyModel:
    def test_forward_shape(self):
        model = LinearProxyModel(feature_dim=32, num_classes=10)
        x = torch.randn(8, 32)
        out = model(x)
        assert out.shape == (8, 10)

    def test_parameter_count(self):
        model = LinearProxyModel(feature_dim=64, num_classes=5)
        params = sum(p.numel() for p in model.parameters())
        assert params == 64 * 5 + 5  # weight + bias


# ---------------------------------------------------------------------------
# train_proxy_model
# ---------------------------------------------------------------------------

class TestTrainProxyModel:
    def test_returns_linear_proxy(self):
        features = torch.randn(50, 16)
        labels = torch.randint(0, 4, (50,))
        proxy = train_proxy_model(features, labels, num_classes=4,
                                  device=DEVICE, epochs=2)
        assert isinstance(proxy, LinearProxyModel)

    def test_loss_decreases(self):
        """Proxy loss should be lower after training than a random init."""
        torch.manual_seed(42)
        features = torch.randn(100, 32)
        labels = torch.randint(0, 4, (100,))

        untrained = LinearProxyModel(32, 4)
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            loss_before = ce(untrained(features), labels).item()

        proxy = train_proxy_model(features, labels, num_classes=4,
                                  device=DEVICE, epochs=20, lr=0.05)
        with torch.no_grad():
            loss_after = ce(proxy(features), labels).item()

        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# select_coreset (end-to-end, no real data)
# ---------------------------------------------------------------------------

class TestSelectCoreset:
    """
    End-to-end tests using a tiny synthetic dataset and a minimal model.
    ToyResNet exposes an avgpool attribute so extract_features' hook works.
    """

    def _run_selection(self, N=80, num_classes=4, omega=0.5, rho=1.0,
                       gamma=2.0, R=2):
        dataset = make_toy_dataset(N=N, num_classes=num_classes)

        # Minimal model that has an avgpool attribute (needed for feature hook)
        class ToyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.relu = nn.ReLU()
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(8, num_classes)

            def forward(self, x):
                return self.fc(self.flatten(self.avgpool(self.relu(self.conv(x)))))

        model = ToyResNet().to(DEVICE)
        coreset_size = int(omega * N)

        indices = select_coreset(
            model=model,
            dataset=dataset,
            coreset_size=coreset_size,
            device=DEVICE,
            R=R,
            rho=rho,
            gamma=gamma,
            batch_size=32,
            num_workers=0,
            num_classes=num_classes,
        )
        return indices, N, coreset_size

    def test_coreset_size_no_reg(self):
        indices, N, target = self._run_selection(rho=1.0)
        assert len(indices) == target

    def test_coreset_size_with_reg(self):
        indices, N, target = self._run_selection(rho=0.5)
        assert len(indices) == target

    def test_indices_in_range(self):
        indices, N, _ = self._run_selection()
        assert all(0 <= i < N for i in indices)

    def test_no_duplicate_indices(self):
        indices, _, _ = self._run_selection(rho=0.5)
        assert len(indices) == len(set(indices))

    def test_coreset_is_subset(self):
        indices, N, target = self._run_selection(omega=0.3, rho=1.0)
        assert len(indices) == target


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_perfect_accuracy(self):
        """Model that always predicts label 0 should score 100% on all-0 labels."""
        class AlwaysZero(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                out = torch.zeros(B, 4)
                out[:, 0] = 1e6
                return out

        images = torch.randn(40, 3, 8, 8)
        labels = torch.zeros(40, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            TensorDataset(images, labels), batch_size=16)
        acc = evaluate(AlwaysZero(), loader, DEVICE)
        assert acc == pytest.approx(100.0)

    def test_zero_accuracy(self):
        """Model that always predicts label 0 should score 0% on all-1 labels."""
        class AlwaysZero(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                out = torch.zeros(B, 4)
                out[:, 0] = 1e6
                return out

        images = torch.randn(40, 3, 8, 8)
        labels = torch.ones(40, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            TensorDataset(images, labels), batch_size=16)
        acc = evaluate(AlwaysZero(), loader, DEVICE)
        assert acc == pytest.approx(0.0)

    def test_accuracy_range(self):
        model = make_tiny_resnet(num_classes=4)
        images = torch.randn(64, 3, 8, 8)
        labels = torch.randint(0, 4, (64,))
        loader = torch.utils.data.DataLoader(
            TensorDataset(images, labels), batch_size=16)
        acc = evaluate(model, loader, DEVICE)
        assert 0.0 <= acc <= 100.0
