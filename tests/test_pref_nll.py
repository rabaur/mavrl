"""Tests for `mavrl.feedback.pref`.

These tests cover:

* output shape / finiteness contracts of `log_probs` and `PreferenceModule.forward`,
* equivalence of `forward` to `F.binary_cross_entropy_with_logits`,
* the label convention (`prefs == 1` => first trajectory preferred),
* behaviour of the validity mask and `normalize_by_length` flag,
* limiting cases (`beta == 0`, `beta -> inf`, soft labels, gradient flow).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mavrl.feedback.pref import PreferenceModule, log_probs


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

DEFAULT_DTYPE = torch.float64  # double precision -> tighter tolerances


def _make_inputs(B: int = 4, T: int = 8, seed: int = 0):
    """Build a deterministic batch of inputs."""
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(B, 2, T, generator=g, dtype=DEFAULT_DTYPE)
    valid = torch.ones(B, 2, T, dtype=DEFAULT_DTYPE)
    prefs = torch.randint(0, 2, (B,), generator=g).to(DEFAULT_DTYPE)
    beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
    return rewards, valid, beta, prefs


@pytest.fixture
def module():
    return PreferenceModule(normalize_by_length=True)


@pytest.fixture
def module_sum():
    return PreferenceModule(normalize_by_length=False)


# --------------------------------------------------------------------------- #
# log_probs (free function)
# --------------------------------------------------------------------------- #

class TestLogProbs:

    def test_returns_correct_shape(self):
        rewards, valid, beta, _ = _make_inputs(B=5, T=7)
        lp = log_probs(rewards, valid, beta)
        assert lp.shape == (5, 2)

    def test_finite_for_typical_inputs(self):
        rewards, valid, beta, _ = _make_inputs(B=3, T=10)
        lp = log_probs(rewards, valid, beta)
        assert torch.isfinite(lp).all()

    def test_log_probs_sum_to_one(self):
        """The two outcomes ('first preferred' vs 'second preferred') form a
        Bernoulli, so exp(lp0) + exp(lp1) must equal 1 for every batch element.
        """
        rewards, valid, beta, _ = _make_inputs(B=8, T=6, seed=1)
        lp = log_probs(rewards, valid, beta)
        probs_sum = lp.exp().sum(dim=-1)
        torch.testing.assert_close(
            probs_sum, torch.ones_like(probs_sum), rtol=0, atol=1e-12,
        )

    def test_uniform_when_beta_is_zero(self):
        """beta = 0 -> indifferent: both outcomes have probability 0.5."""
        rewards, valid, _, _ = _make_inputs(B=4, T=5, seed=2)
        beta = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
        lp = log_probs(rewards, valid, beta)
        expected = torch.full_like(lp, math.log(0.5))
        torch.testing.assert_close(lp, expected)

    def test_first_outcome_dominates_when_seg1_better(self):
        """If seg1 is strictly better and beta is large, log P(seg1) ~ 0
        and log P(seg2) -> -inf.
        """
        T = 4
        rewards = torch.zeros(1, 2, T, dtype=DEFAULT_DTYPE)
        rewards[0, 0] = 1.0
        rewards[0, 1] = -1.0
        valid = torch.ones_like(rewards)
        beta = torch.tensor(50.0, dtype=DEFAULT_DTYPE)

        lp = log_probs(rewards, valid, beta)
        assert lp[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert lp[0, 1].item() < -10.0

    def test_matches_logsigmoid_formula(self):
        """Verify against an explicit hand-computed expression."""
        rewards = torch.tensor([[[2.0, 2.0], [0.0, 0.0]]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        # mean rewards: agg1 = 2.0, agg2 = 0.0, logits = 2.0
        expected = torch.stack([
            F.logsigmoid(torch.tensor(2.0, dtype=DEFAULT_DTYPE)),
            F.logsigmoid(torch.tensor(-2.0, dtype=DEFAULT_DTYPE)),
        ]).unsqueeze(0)

        lp = log_probs(rewards, valid, beta)
        torch.testing.assert_close(lp, expected)

    def test_normalize_by_length_flag_changes_aggregation(self):
        """The kwarg must actually toggle mean vs sum aggregation."""
        rewards = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]],
                               dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        lp_mean = log_probs(rewards, valid, beta, normalize_by_length=True)
        lp_sum = log_probs(rewards, valid, beta, normalize_by_length=False)
        # mean -> logits = 1.0; sum -> logits = 3.0
        assert lp_mean[0, 0].item() == pytest.approx(
            F.logsigmoid(torch.tensor(1.0, dtype=DEFAULT_DTYPE)).item(),
        )
        assert lp_sum[0, 0].item() == pytest.approx(
            F.logsigmoid(torch.tensor(3.0, dtype=DEFAULT_DTYPE)).item(),
        )


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #

class TestForward:

    def test_returns_scalar(self, module):
        rewards, valid, beta, prefs = _make_inputs()
        loss = module(rewards, valid, beta, prefs)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self, module):
        rewards, valid, beta, prefs = _make_inputs(seed=3)
        loss = module(rewards, valid, beta, prefs)
        assert loss.item() >= 0.0

    def test_matches_bce_with_logits(self, module):
        """`forward` must agree with `F.binary_cross_entropy_with_logits` on
        the logits implied by the (mean) reward difference.
        """
        rewards, valid, beta, prefs = _make_inputs(seed=4)
        loss = module(rewards, valid, beta, prefs)

        agg = (rewards * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        logits = beta * (agg[..., 0] - agg[..., 1])
        expected = F.binary_cross_entropy_with_logits(
            logits, prefs, reduction="mean"
        )
        torch.testing.assert_close(loss, expected)

    def test_label_convention_pref1_means_first_preferred(self, module):
        """Regression test for the convention bug.

        With seg1 strictly better than seg2, the NLL must be lower under
        `prefs == 1` (the dataset's "first preferred" label) than under
        `prefs == 0`.
        """
        T = 5
        rewards = torch.zeros(1, 2, T, dtype=DEFAULT_DTYPE)
        rewards[0, 0] = 1.0
        rewards[0, 1] = -1.0
        valid = torch.ones_like(rewards)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        loss_first = module(rewards, valid, beta,
                            torch.tensor([1.0], dtype=DEFAULT_DTYPE))
        loss_second = module(rewards, valid, beta,
                             torch.tensor([0.0], dtype=DEFAULT_DTYPE))
        assert loss_first.item() < loss_second.item()

    def test_swap_invariance(self, module):
        """Swapping the two trajectories and flipping the label
        leaves the loss unchanged.
        """
        rewards, valid, beta, prefs = _make_inputs(seed=5)
        loss = module(rewards, valid, beta, prefs)

        rewards_swapped = rewards.flip(dims=(1,))
        valid_swapped = valid.flip(dims=(1,))
        prefs_swapped = 1.0 - prefs
        loss_swapped = module(
            rewards_swapped, valid_swapped, beta, prefs_swapped,
        )
        torch.testing.assert_close(loss, loss_swapped)

    def test_uniform_loss_when_beta_zero(self, module):
        """At beta = 0 every prediction is 0.5, so the loss is log(2)."""
        rewards, valid, _, prefs = _make_inputs(seed=6)
        beta = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
        loss = module(rewards, valid, beta, prefs)
        torch.testing.assert_close(
            loss, torch.tensor(math.log(2.0), dtype=DEFAULT_DTYPE),
        )

    def test_known_value_single_pair(self, module):
        """Hand-computed loss for a single, simple pair."""
        rewards = torch.tensor([[[2.0], [0.0]]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
        prefs = torch.tensor([1.0], dtype=DEFAULT_DTYPE)

        loss = module(rewards, valid, beta, prefs)
        # NLL = softplus(-(r1 - r2)) = log(1 + exp(-2))
        expected = math.log1p(math.exp(-2.0))
        assert loss.item() == pytest.approx(expected, abs=1e-12)

    def test_soft_labels(self, module):
        """For soft `prefs` in [0, 1] the loss is the expected cross entropy."""
        rewards, valid, beta, _ = _make_inputs(seed=7)
        prefs = torch.full((rewards.shape[0],), 0.7, dtype=DEFAULT_DTYPE)

        loss = module(rewards, valid, beta, prefs)

        agg = (rewards * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        logits = beta * (agg[..., 0] - agg[..., 1])
        expected = F.binary_cross_entropy_with_logits(
            logits, prefs, reduction="mean",
        )
        torch.testing.assert_close(loss, expected)


# --------------------------------------------------------------------------- #
# Validity mask
# --------------------------------------------------------------------------- #

class TestValidityMask:

    def test_invalid_timesteps_are_ignored(self, module):
        """Padding the segment with garbage rewards behind a `valid=0` mask
        must not change the loss when `normalize_by_length=True`.
        """
        rewards, valid, beta, prefs = _make_inputs(B=3, T=4, seed=8)

        # Append two padding steps on each trajectory with "garbage" rewards.
        pad_rewards = torch.full(
            (rewards.shape[0], 2, 2), 1e3, dtype=DEFAULT_DTYPE,
        )
        pad_valid = torch.zeros_like(pad_rewards)

        rewards_padded = torch.cat([rewards, pad_rewards], dim=-1)
        valid_padded = torch.cat([valid, pad_valid], dim=-1)

        loss = module(rewards, valid, beta, prefs)
        loss_padded = module(rewards_padded, valid_padded, beta, prefs)
        torch.testing.assert_close(loss, loss_padded)

    def test_all_invalid_does_not_nan(self, module):
        """A pair with no valid timesteps must yield finite logits (the code
        clamps `n_valid` to `min=1`).
        """
        rewards = torch.randn(1, 2, 4, dtype=DEFAULT_DTYPE)
        valid = torch.zeros_like(rewards)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
        prefs = torch.tensor([1.0], dtype=DEFAULT_DTYPE)

        loss = module(rewards, valid, beta, prefs)
        assert torch.isfinite(loss)
        # All rewards are masked away, so logits = 0 and loss = log 2.
        torch.testing.assert_close(
            loss, torch.tensor(math.log(2.0), dtype=DEFAULT_DTYPE),
        )


# --------------------------------------------------------------------------- #
# normalize_by_length flag
# --------------------------------------------------------------------------- #

class TestLengthNormalization:

    def test_normalized_loss_invariant_to_repetition(self, module):
        """Repeating the same per-step rewards twice (doubling T) should not
        change the loss when rewards are mean-aggregated.
        """
        rewards, valid, beta, prefs = _make_inputs(B=2, T=3, seed=9)

        rewards_double = rewards.repeat(1, 1, 2)
        valid_double = valid.repeat(1, 1, 2)

        loss = module(rewards, valid, beta, prefs)
        loss_double = module(rewards_double, valid_double, beta, prefs)
        torch.testing.assert_close(loss, loss_double)

    def test_unnormalized_loss_changes_with_repetition(self, module_sum):
        """Without length normalization, doubling T scales the logits and
        therefore the loss differently.
        """
        rewards, valid, beta, prefs = _make_inputs(B=2, T=3, seed=10)

        rewards_double = rewards.repeat(1, 1, 2)
        valid_double = valid.repeat(1, 1, 2)

        loss = module_sum(rewards, valid, beta, prefs)
        loss_double = module_sum(rewards_double, valid_double, beta, prefs)
        assert not torch.allclose(loss, loss_double)


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #

class TestGradients:

    def test_gradient_flows_to_reward_samples(self, module):
        rewards, valid, beta, prefs = _make_inputs(seed=11)
        rewards = rewards.detach().requires_grad_(True)
        loss = module(rewards, valid, beta, prefs)
        loss.backward()
        assert rewards.grad is not None
        assert torch.isfinite(rewards.grad).all()
        assert rewards.grad.abs().sum().item() > 0.0

    def test_gradient_descent_reduces_loss(self, module):
        """A few SGD steps on the rewards should lower the loss."""
        rewards, valid, beta, prefs = _make_inputs(B=8, T=6, seed=12)
        rewards = rewards.detach().requires_grad_(True)

        initial_loss = module(rewards, valid, beta, prefs).item()
        opt = torch.optim.SGD([rewards], lr=0.5)
        for _ in range(20):
            opt.zero_grad()
            module(rewards, valid, beta, prefs).backward()
            opt.step()
        final_loss = module(rewards, valid, beta, prefs).item()

        assert final_loss < initial_loss
