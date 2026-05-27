"""Tests for `mavrl.feedback.rate`.

These tests cover:

* output shape / finiteness contracts of `log_probs` and `RatingDecoder.forward`,
* probabilistic invariants (sum-to-one, log-probs <= 0, agreement with the
  linear-space formula at moderate rewards, first-order stochastic dominance
  in the aggregated reward),
* the 0-indexed convention (rating == K-1 corresponds to the highest reward,
  rating == 0 to the lowest),
* numerical stability at extreme reward values: the motivation for the
  log-space implementation, where the previous `clamp(1e-8) -> log` pattern
  saturated wrong-tail log-probs around -18.4 and zeroed their gradient,
* behaviour of the validity mask and `normalize_by_length` flag,
* `RatingDecoder` cutpoint-ordering invariant and gradient flow to both
  rewards and cutpoint parameters.
"""

import pytest
import torch
import torch.nn.functional as F

from mavrl.feedback.rate import RatingModule, log_probs


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

DEFAULT_DTYPE = torch.float64  # double precision -> tighter tolerances


def _make_inputs(B: int = 4, T: int = 8, K: int = 5, seed: int = 0):
    """Build a deterministic batch of inputs."""
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(B, T, generator=g, dtype=DEFAULT_DTYPE)
    valid = torch.ones(B, T, dtype=DEFAULT_DTYPE)
    cutpoints = torch.linspace(-1.0, 1.0, K - 1, dtype=DEFAULT_DTYPE)
    ratings = torch.randint(0, K, (B,), generator=g)
    return rewards, valid, cutpoints, ratings


@pytest.fixture
def module():
    """Float64 module so its parameters match `DEFAULT_DTYPE` inputs."""
    return RatingModule(num_categories=5, normalize_by_length=True).double()


@pytest.fixture
def module_sum():
    return RatingModule(num_categories=5, normalize_by_length=False).double()


# --------------------------------------------------------------------------- #
# log_probs (free function)
# --------------------------------------------------------------------------- #


class TestLogProbs:

    def test_returns_correct_shape(self):
        rewards, valid, cutpoints, _ = _make_inputs(B=5, T=7, K=4)
        lp = log_probs(rewards, valid, cutpoints)
        assert lp.shape == (5, 4)

    def test_finite_for_typical_inputs(self):
        rewards, valid, cutpoints, _ = _make_inputs(B=3, T=10, K=5)
        lp = log_probs(rewards, valid, cutpoints)
        assert torch.isfinite(lp).all()

    def test_log_probs_are_non_positive(self):
        rewards, valid, cutpoints, _ = _make_inputs(seed=1)
        lp = log_probs(rewards, valid, cutpoints)
        # Allow a tiny FP slack; logs of probabilities must be <= 0.
        assert (lp <= 1e-12).all()

    def test_log_probs_sum_to_one(self):
        """exp(log_probs).sum(-1) == 1 by the telescoping identity."""
        rewards, valid, cutpoints, _ = _make_inputs(B=8, T=6, K=5, seed=2)
        lp = log_probs(rewards, valid, cutpoints)
        probs_sum = lp.exp().sum(dim=-1)
        torch.testing.assert_close(
            probs_sum, torch.ones_like(probs_sum), rtol=0, atol=1e-12,
        )

    def test_matches_linear_space_formula(self):
        """For moderate reward magnitudes the log-space identity must agree
        with a direct linear-space evaluation of `P(y=k) = sigma(b) - sigma(a)`.
        """
        cutpoints = torch.tensor([-1.0, 0.0, 1.0], dtype=DEFAULT_DTYPE)
        r = 0.5
        rewards = torch.tensor([[r]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)

        lp = log_probs(rewards, valid, cutpoints).squeeze(0)

        sig = torch.sigmoid(cutpoints - r)
        cum = torch.cat([
            torch.zeros(1, dtype=DEFAULT_DTYPE),
            sig,
            torch.ones(1, dtype=DEFAULT_DTYPE),
        ])
        expected = (cum[1:] - cum[:-1]).log()
        torch.testing.assert_close(lp, expected, atol=1e-12, rtol=0)

    def test_K_equals_2_reduces_to_bernoulli(self):
        """For K=2, the cumulative-logit model becomes a Bernoulli with
        logit `(r - theta_1)`.
        """
        cutpoints = torch.tensor([0.5], dtype=DEFAULT_DTYPE)
        rewards = torch.tensor([[1.0], [0.0], [-1.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)

        lp = log_probs(rewards, valid, cutpoints)  # (3, 2)
        agg = rewards.squeeze(-1)
        expected_lo = F.logsigmoid(cutpoints - agg)
        expected_hi = F.logsigmoid(agg - cutpoints)
        torch.testing.assert_close(lp[:, 0], expected_lo, atol=1e-12, rtol=0)
        torch.testing.assert_close(lp[:, 1], expected_hi, atol=1e-12, rtol=0)

    def test_high_reward_concentrates_on_top_category(self):
        """Aggregated reward -> +inf shifts all mass to the highest category."""
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)
        rewards = torch.tensor([[100.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)

        lp = log_probs(rewards, valid, cutpoints).squeeze(0)
        assert lp[-1].item() == pytest.approx(0.0, abs=1e-12)
        assert (lp[:-1] < -50).all()

    def test_low_reward_concentrates_on_bottom_category(self):
        """Aggregated reward -> -inf shifts all mass to the lowest category."""
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)
        rewards = torch.tensor([[-100.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)

        lp = log_probs(rewards, valid, cutpoints).squeeze(0)
        assert lp[0].item() == pytest.approx(0.0, abs=1e-12)
        assert (lp[1:] < -50).all()

    def test_extreme_reward_log_probs_remain_correct(self):
        """At extreme aggregated reward, the *wrong-tail* category must
        give the correct very-negative log-prob, not saturate at log(1e-8)
        ~ -18.4 as the previous linear-space implementation did.
        """
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)
        rewards = torch.tensor([[-50.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)

        lp = log_probs(rewards, valid, cutpoints).squeeze(0)
        assert torch.isfinite(lp).all()
        # Bottom category is the *correct* tail at r=-50, log P ~ 0.
        assert lp[0].item() == pytest.approx(0.0, abs=1e-12)
        # Top category: log P(y=K-1|r) = logsigmoid(r - theta_{K-1}).
        expected_top = F.logsigmoid(
            torch.tensor(-50.0 - 1.0, dtype=DEFAULT_DTYPE),
        )
        torch.testing.assert_close(lp[-1], expected_top, atol=1e-6, rtol=0)
        # And it sits well below the -18.4 cap of the old `clamp(1e-8) -> log`
        # implementation - this is the regression we are guarding against.
        assert lp[-1].item() < -40

    def test_first_order_stochastic_dominance_in_reward(self):
        """Higher reward must shift the rating CDF to the right:
        F(k | r_high) <= F(k | r_low) for every k.
        """
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)
        rewards_low = torch.tensor([[-1.0]], dtype=DEFAULT_DTYPE)
        rewards_high = torch.tensor([[+1.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards_low)

        cdf_low = log_probs(rewards_low, valid, cutpoints).exp().cumsum(-1)
        cdf_high = log_probs(rewards_high, valid, cutpoints).exp().cumsum(-1)
        assert (cdf_low >= cdf_high - 1e-12).all()

    def test_normalize_by_length_flag_changes_aggregation(self):
        """The kwarg must actually toggle mean vs sum aggregation."""
        rewards = torch.tensor([[1.0, 1.0, 1.0]], dtype=DEFAULT_DTYPE)
        valid = torch.ones_like(rewards)
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)

        lp_mean = log_probs(rewards, valid, cutpoints, normalize_by_length=True)
        lp_sum = log_probs(rewards, valid, cutpoints, normalize_by_length=False)

        # Mean -> agg = 1.0; sum -> agg = 3.0. Pass those scalars in via T=1
        # and a single valid step to bypass aggregation choice.
        unit_valid = torch.ones(1, 1, dtype=DEFAULT_DTYPE)
        ref_mean = log_probs(
            torch.tensor([[1.0]], dtype=DEFAULT_DTYPE), unit_valid, cutpoints,
        )
        ref_sum = log_probs(
            torch.tensor([[3.0]], dtype=DEFAULT_DTYPE), unit_valid, cutpoints,
        )
        torch.testing.assert_close(lp_mean, ref_mean)
        torch.testing.assert_close(lp_sum, ref_sum)


# --------------------------------------------------------------------------- #
# RatingDecoder.forward
# --------------------------------------------------------------------------- #


class TestForward:

    def test_returns_scalar(self, module):
        rewards, valid, _, ratings = _make_inputs(K=5)
        loss = module(rewards, valid, ratings)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self, module):
        rewards, valid, _, ratings = _make_inputs(seed=3, K=5)
        loss = module(rewards, valid, ratings)
        assert loss.item() >= 0.0

    def test_finite_for_typical_inputs(self, module):
        rewards, valid, _, ratings = _make_inputs(seed=4, K=5)
        loss = module(rewards, valid, ratings)
        assert torch.isfinite(loss)

    def test_matches_explicit_gather(self, module):
        """`forward` must equal `-log_probs(...).gather(...).mean()`."""
        rewards, valid, _, ratings = _make_inputs(seed=5, K=5)
        loss = module(rewards, valid, ratings)

        cutpoints = module.get_cutpoints()
        lps = log_probs(rewards, valid, cutpoints)
        expected = -lps.gather(1, ratings.unsqueeze(1)).squeeze(1).mean()
        torch.testing.assert_close(loss, expected)

    def test_lower_loss_when_ratings_match_rewards(self, module):
        """Pairings in which higher reward -> higher rating give a lower
        NLL than the inverted assignment.
        """
        rewards = torch.tensor(
            [[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=DEFAULT_DTYPE,
        )
        valid = torch.ones_like(rewards)
        ratings_correct = torch.tensor([0, 1, 2, 3, 4])
        ratings_inverted = torch.tensor([4, 3, 2, 1, 0])

        loss_correct = module(rewards, valid, ratings_correct)
        loss_inverted = module(rewards, valid, ratings_inverted)
        assert loss_correct.item() < loss_inverted.item()

    def test_shift_invariance_under_intercept_shift(self, module):
        """Adding a constant to both `r` and every cutpoint leaves the
        NLL unchanged (the model only depends on the differences
        `theta_k - r`).
        """
        rewards, valid, _, ratings = _make_inputs(seed=6, K=5)
        loss = module(rewards, valid, ratings)

        delta = 3.7
        rewards_shifted = rewards + delta
        with torch.no_grad():
            module.raw_theta_1.add_(delta)
        loss_shifted = module(rewards_shifted, valid, ratings)
        torch.testing.assert_close(loss, loss_shifted)


# --------------------------------------------------------------------------- #
# Cutpoint parameterization
# --------------------------------------------------------------------------- #


class TestCutpoints:

    @pytest.mark.parametrize("K", [3, 4, 5, 10])
    def test_strictly_increasing_after_random_perturbation(self, K):
        """The softplus-cumsum parameterization must stay strictly ordered
        for arbitrary `delta_increments` values.
        """
        m = RatingModule(num_categories=K)
        with torch.no_grad():
            m.delta_increments.copy_(torch.randn(K - 2))
            m.raw_theta_1.copy_(torch.tensor(0.5))
        cuts = m.get_cutpoints()
        assert cuts.shape == (K - 1,)
        assert (cuts[1:] > cuts[:-1]).all()

    def test_K2_returns_single_cutpoint(self):
        m = RatingModule(num_categories=2)
        cuts = m.get_cutpoints()
        assert cuts.shape == (1,)


# --------------------------------------------------------------------------- #
# Validity mask
# --------------------------------------------------------------------------- #


class TestValidityMask:

    def test_invalid_timesteps_are_ignored(self, module):
        """Padding the segment with garbage rewards behind a `valid=0`
        mask must not change the loss when `normalize_by_length=True`.
        """
        rewards, valid, _, ratings = _make_inputs(B=3, T=4, K=5, seed=8)

        pad_rewards = torch.full(
            (rewards.shape[0], 2), 1e3, dtype=DEFAULT_DTYPE,
        )
        pad_valid = torch.zeros_like(pad_rewards)
        rewards_padded = torch.cat([rewards, pad_rewards], dim=-1)
        valid_padded = torch.cat([valid, pad_valid], dim=-1)

        loss = module(rewards, valid, ratings)
        loss_padded = module(rewards_padded, valid_padded, ratings)
        torch.testing.assert_close(loss, loss_padded)

    def test_all_invalid_does_not_nan(self, module):
        """A segment with no valid timesteps must yield a finite loss
        (`aggregate_rewards` clamps `n_valid` at 1).
        """
        rewards = torch.randn(2, 4, dtype=DEFAULT_DTYPE)
        valid = torch.zeros_like(rewards)
        ratings = torch.tensor([0, 4])

        loss = module(rewards, valid, ratings)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# normalize_by_length flag
# --------------------------------------------------------------------------- #


class TestLengthNormalization:

    def test_normalized_loss_invariant_to_repetition(self, module):
        """Repeating the per-step rewards twice doesn't change the loss
        when rewards are mean-aggregated.
        """
        rewards, valid, _, ratings = _make_inputs(B=2, T=3, K=5, seed=9)
        rewards_double = rewards.repeat(1, 2)
        valid_double = valid.repeat(1, 2)

        loss = module(rewards, valid, ratings)
        loss_double = module(rewards_double, valid_double, ratings)
        torch.testing.assert_close(loss, loss_double)

    def test_unnormalized_loss_changes_with_repetition(self, module_sum):
        """Without length normalization, doubling T scales the aggregated
        reward and therefore the loss.
        """
        rewards, valid, _, ratings = _make_inputs(B=2, T=3, K=5, seed=10)
        rewards_double = rewards.repeat(1, 2)
        valid_double = valid.repeat(1, 2)

        loss = module_sum(rewards, valid, ratings)
        loss_double = module_sum(rewards_double, valid_double, ratings)
        assert not torch.allclose(loss, loss_double)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


class TestGradients:

    def test_gradient_flows_to_rewards(self, module):
        rewards, valid, _, ratings = _make_inputs(seed=11, K=5)
        rewards = rewards.detach().requires_grad_(True)
        loss = module(rewards, valid, ratings)
        loss.backward()
        assert rewards.grad is not None
        assert torch.isfinite(rewards.grad).all()
        assert rewards.grad.abs().sum().item() > 0.0

    def test_gradient_flows_to_cutpoint_parameters(self, module):
        rewards, valid, _, ratings = _make_inputs(seed=12, K=5)
        loss = module(rewards, valid, ratings)
        loss.backward()
        assert module.raw_theta_1.grad is not None
        assert torch.isfinite(module.raw_theta_1.grad)
        assert module.delta_increments.grad is not None
        assert torch.isfinite(module.delta_increments.grad).all()

    def test_gradient_descent_reduces_loss(self):
        """A few SGD steps over rewards and cutpoint parameters should
        reduce the loss given fixed labels.
        """
        m = RatingModule(num_categories=5).double()
        K = 5
        B, T = 8, 5
        g = torch.Generator().manual_seed(0)
        rewards = torch.randn(
            B, T, generator=g, dtype=DEFAULT_DTYPE,
        ).requires_grad_(True)
        valid = torch.ones(B, T, dtype=DEFAULT_DTYPE)
        ratings = torch.randint(0, K, (B,), generator=g)

        initial_loss = m(rewards, valid, ratings).item()
        opt = torch.optim.SGD([rewards] + list(m.parameters()), lr=0.5)
        for _ in range(20):
            opt.zero_grad()
            m(rewards, valid, ratings).backward()
            opt.step()
        final_loss = m(rewards, valid, ratings).item()
        assert final_loss < initial_loss

    def test_gradient_finite_at_extreme_reward_wrong_tail(self):
        """Log-space implementation gives a finite, non-zero gradient for
        the *wrong-tail* category at extreme reward magnitudes - exactly
        the case where the previous `clamp(1e-8) -> log` pattern silenced
        the gradient.
        """
        cutpoints = torch.linspace(-1.0, 1.0, 4, dtype=DEFAULT_DTYPE)
        rewards = torch.tensor(
            [[50.0]], dtype=DEFAULT_DTYPE, requires_grad=True,
        )
        valid = torch.ones_like(rewards.detach())

        # log P(y=0 | r=50): the lowest category at very high reward.
        lp = log_probs(rewards, valid, cutpoints)[0, 0]
        lp.backward()
        assert rewards.grad is not None
        assert torch.isfinite(rewards.grad).all()
        assert rewards.grad.abs().item() > 0.0
