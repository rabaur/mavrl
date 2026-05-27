"""Tests for `mavrl.feedback.demo`.

These tests cover:

* output shape / finiteness contracts of `log_probs` and
  `DemonstrationsDecoder.forward`,
* probabilistic invariants (sum-to-one, log-probs <= 0, uniform at beta=0,
  argmax-dominates at beta -> inf, agreement with `F.log_softmax`),
* equivalence of `forward` to the previous `F.cross_entropy(beta * q, a)`
  implementation (regression test for the refactor) and to an explicit
  `-log_probs(...).gather(...).mean()` decomposition,
* behaviour of the validity mask (1D and 2D shapes, all-invalid edge case),
* gradient flow and SGD convergence to the expert policy.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mavrl.feedback.demo import DemonstrationsDecoder, log_probs


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

DEFAULT_DTYPE = torch.float64  # double precision -> tighter tolerances


def _make_inputs(B: int = 4, A: int = 5, seed: int = 0):
    """Build a deterministic batch of inputs."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, A, generator=g, dtype=DEFAULT_DTYPE)
    acts = torch.randint(0, A, (B, 1), generator=g)
    valid = torch.ones(B, dtype=DEFAULT_DTYPE)
    beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
    return q, acts, beta, valid


@pytest.fixture
def module():
    return DemonstrationsDecoder()


# --------------------------------------------------------------------------- #
# log_probs (free function)
# --------------------------------------------------------------------------- #


class TestLogProbs:

    def test_returns_correct_shape(self):
        q, _, beta, _ = _make_inputs(B=5, A=4)
        lp = log_probs(q, beta)
        assert lp.shape == (5, 4)

    def test_finite_for_typical_inputs(self):
        q, _, beta, _ = _make_inputs(B=3, A=7)
        lp = log_probs(q, beta)
        assert torch.isfinite(lp).all()

    def test_log_probs_are_non_positive(self):
        q, _, beta, _ = _make_inputs(seed=1)
        lp = log_probs(q, beta)
        assert (lp <= 1e-12).all()

    def test_log_probs_sum_to_one(self):
        q, _, beta, _ = _make_inputs(B=8, A=6, seed=2)
        lp = log_probs(q, beta)
        probs_sum = lp.exp().sum(dim=-1)
        torch.testing.assert_close(
            probs_sum, torch.ones_like(probs_sum), rtol=0, atol=1e-12,
        )

    def test_uniform_when_beta_is_zero(self):
        """beta = 0 -> every action equally likely: log P = log(1 / A)."""
        q, _, _, _ = _make_inputs(B=4, A=5, seed=3)
        beta = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
        lp = log_probs(q, beta)
        expected = torch.full_like(lp, -math.log(5))
        torch.testing.assert_close(lp, expected)

    def test_argmax_dominates_when_beta_is_large(self):
        """At large beta the argmax action carries log P ~ 0 and the rest -> -inf."""
        q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=DEFAULT_DTYPE)
        beta = torch.tensor(50.0, dtype=DEFAULT_DTYPE)
        lp = log_probs(q, beta)
        assert lp[0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert (lp[0, [0, 2, 3]] < -40).all()

    def test_matches_log_softmax(self):
        """Hand-computed equivalence to `F.log_softmax(beta * q, dim=-1)`."""
        q, _, beta, _ = _make_inputs(seed=4)
        lp = log_probs(q, beta)
        expected = F.log_softmax(beta * q, dim=-1)
        torch.testing.assert_close(lp, expected, atol=1e-12, rtol=0)

    def test_supports_leading_batch_dimensions(self):
        """log_probs must broadcast over leading batch dims, not just (B, A)."""
        g = torch.Generator().manual_seed(5)
        q = torch.randn(2, 3, 4, generator=g, dtype=DEFAULT_DTYPE)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
        lp = log_probs(q, beta)
        assert lp.shape == (2, 3, 4)
        torch.testing.assert_close(
            lp.exp().sum(-1),
            torch.ones(2, 3, dtype=DEFAULT_DTYPE),
            rtol=0, atol=1e-12,
        )

    def test_shift_invariance_in_q(self):
        """Adding a constant to every Q at a state is a softmax shift; the
        resulting log_probs must be unchanged.
        """
        q, _, beta, _ = _make_inputs(seed=6)
        lp = log_probs(q, beta)
        delta = 7.3
        lp_shifted = log_probs(q + delta, beta)
        torch.testing.assert_close(lp, lp_shifted, atol=1e-12, rtol=0)


# --------------------------------------------------------------------------- #
# DemonstrationsDecoder.forward
# --------------------------------------------------------------------------- #


class TestForward:

    def test_returns_scalar(self, module):
        q, acts, beta, valid = _make_inputs()
        loss = module(acts, q, beta, valid)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self, module):
        q, acts, beta, valid = _make_inputs(seed=7)
        loss = module(acts, q, beta, valid)
        assert loss.item() >= 0.0

    def test_finite_for_typical_inputs(self, module):
        q, acts, beta, valid = _make_inputs(seed=8)
        loss = module(acts, q, beta, valid)
        assert torch.isfinite(loss)

    def test_matches_cross_entropy(self, module):
        """Regression test for the refactor: forward must equal the
        previous `F.cross_entropy(beta * q, acts.squeeze(-1))` formulation
        on fully valid inputs.
        """
        q, acts, beta, valid = _make_inputs(seed=9)
        loss = module(acts, q, beta, valid)
        expected = F.cross_entropy(
            beta * q, acts.squeeze(-1), reduction="mean",
        )
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=0)

    def test_matches_explicit_gather(self, module):
        """forward must equal `-log_probs(...).gather(...).mean()`."""
        q, acts, beta, valid = _make_inputs(seed=10)
        loss = module(acts, q, beta, valid)

        lps = log_probs(q, beta)
        expected = -lps.gather(1, acts.long().view(-1, 1)).squeeze(-1).mean()
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=0)

    def test_uniform_loss_when_beta_zero(self, module):
        """At beta = 0 every action is equiprobable, so the NLL is log(A)."""
        q, acts, _, valid = _make_inputs(B=6, A=4, seed=11)
        beta = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
        loss = module(acts, q, beta, valid)
        torch.testing.assert_close(
            loss, torch.tensor(math.log(4.0), dtype=DEFAULT_DTYPE),
        )

    def test_loss_vanishes_for_argmax_actions_at_high_beta(self, module):
        """When the demonstrated action equals argmax_a Q(s, a) and beta is
        large, the NLL collapses to ~0.
        """
        q = torch.tensor(
            [[0.0, 1.0, 0.0],
             [2.0, 0.0, 0.0]], dtype=DEFAULT_DTYPE,
        )
        acts = torch.tensor([[1], [0]])
        valid = torch.ones(2, dtype=DEFAULT_DTYPE)
        beta = torch.tensor(50.0, dtype=DEFAULT_DTYPE)
        loss = module(acts, q, beta, valid)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_argmax_actions_have_lower_loss_than_anti_argmax(self, module):
        """A batch demonstrating argmax actions must have strictly lower
        NLL than the same batch demonstrating argmin actions.
        """
        g = torch.Generator().manual_seed(12)
        q = torch.randn(8, 5, generator=g, dtype=DEFAULT_DTYPE)
        valid = torch.ones(8, dtype=DEFAULT_DTYPE)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        argmax_acts = q.argmax(dim=-1, keepdim=True)
        argmin_acts = q.argmin(dim=-1, keepdim=True)

        loss_argmax = module(argmax_acts, q, beta, valid)
        loss_argmin = module(argmin_acts, q, beta, valid)
        assert loss_argmax.item() < loss_argmin.item()

    def test_known_value_single_sample(self, module):
        """Hand-computed loss for a single (s, a) pair."""
        q = torch.tensor([[2.0, 0.0]], dtype=DEFAULT_DTYPE)
        acts = torch.tensor([[0]])
        valid = torch.ones(1, dtype=DEFAULT_DTYPE)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        loss = module(acts, q, beta, valid)
        # NLL = -log_softmax(beta*q)[0] = log(1 + exp(-2))
        expected = math.log1p(math.exp(-2.0))
        assert loss.item() == pytest.approx(expected, abs=1e-12)

    def test_accepts_1d_acts(self, module):
        """`forward` should accept actions of shape `(B,)` as well as `(B, 1)`."""
        q, acts_2d, beta, valid = _make_inputs(seed=13)
        acts_1d = acts_2d.squeeze(-1)

        loss_2d = module(acts_2d, q, beta, valid)
        loss_1d = module(acts_1d, q, beta, valid)
        torch.testing.assert_close(loss_1d, loss_2d, atol=1e-12, rtol=0)


# --------------------------------------------------------------------------- #
# Validity mask
# --------------------------------------------------------------------------- #


class TestValidityMask:

    def test_invalid_entries_are_ignored(self, module):
        """Replacing the Q-values at invalid rows with garbage must not
        change the loss."""
        q, acts, beta, valid = _make_inputs(B=4, A=3, seed=14)
        # Mark first two rows invalid.
        valid = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=DEFAULT_DTYPE)
        loss = module(acts, q, beta, valid)

        q_garbage = q.clone()
        q_garbage[:2] = 1e6
        loss_garbage = module(acts, q_garbage, beta, valid)
        torch.testing.assert_close(loss, loss_garbage, atol=1e-12, rtol=0)

    def test_all_invalid_does_not_nan(self, module):
        """A batch with no valid entries must yield a finite loss
        (the implementation clamps the denominator to >= 1).
        """
        q, acts, beta, _ = _make_inputs(B=3, A=4, seed=15)
        valid = torch.zeros(3, dtype=DEFAULT_DTYPE)
        loss = module(acts, q, beta, valid)
        assert torch.isfinite(loss)
        # All entries are masked, so the numerator is 0 and the loss is 0.
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_mask_works_with_2d_shape(self, module):
        """A `(B, 1)` mask must behave identically to a `(B,)` mask."""
        q, acts, beta, _ = _make_inputs(seed=16)
        valid_1d = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=DEFAULT_DTYPE)
        valid_2d = valid_1d.unsqueeze(-1)

        loss_1d = module(acts, q, beta, valid_1d)
        loss_2d = module(acts, q, beta, valid_2d)
        torch.testing.assert_close(loss_1d, loss_2d, atol=1e-12, rtol=0)

    def test_mask_averages_only_over_valid(self, module):
        """The denominator is the number of valid entries, not the batch size."""
        q = torch.tensor(
            [[2.0, 0.0],
             [0.0, 2.0]], dtype=DEFAULT_DTYPE,
        )
        acts = torch.tensor([[0], [1]])
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)
        valid = torch.tensor([1.0, 0.0], dtype=DEFAULT_DTYPE)

        loss = module(acts, q, beta, valid)
        # Only the first sample contributes; expected NLL = log(1 + exp(-2)).
        expected = math.log1p(math.exp(-2.0))
        assert loss.item() == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #


class TestGradients:

    def test_gradient_flows_to_q_values(self, module):
        q, acts, beta, valid = _make_inputs(seed=17)
        q = q.detach().requires_grad_(True)
        loss = module(acts, q, beta, valid)
        loss.backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()
        assert q.grad.abs().sum().item() > 0.0

    def test_gradient_zero_at_invalid_rows(self, module):
        """Invalid rows must contribute zero gradient to q_values."""
        q, acts, beta, _ = _make_inputs(B=4, A=3, seed=18)
        q = q.detach().requires_grad_(True)
        valid = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=DEFAULT_DTYPE)

        loss = module(acts, q, beta, valid)
        loss.backward()
        assert q.grad is not None
        # Rows where valid == 0 should have exactly zero gradient.
        torch.testing.assert_close(
            q.grad[1], torch.zeros_like(q.grad[1]), atol=0, rtol=0,
        )
        torch.testing.assert_close(
            q.grad[3], torch.zeros_like(q.grad[3]), atol=0, rtol=0,
        )
        # Valid rows must have a non-trivial gradient.
        assert q.grad[0].abs().sum().item() > 0.0
        assert q.grad[2].abs().sum().item() > 0.0

    def test_gradient_descent_reduces_loss(self, module):
        """A few SGD steps on the Q-values must reduce the demonstration NLL."""
        g = torch.Generator().manual_seed(19)
        q = torch.randn(16, 4, generator=g, dtype=DEFAULT_DTYPE).requires_grad_(True)
        acts = torch.randint(0, 4, (16, 1), generator=g)
        valid = torch.ones(16, dtype=DEFAULT_DTYPE)
        beta = torch.tensor(1.0, dtype=DEFAULT_DTYPE)

        initial_loss = module(acts, q, beta, valid).item()
        opt = torch.optim.SGD([q], lr=0.5)
        for _ in range(20):
            opt.zero_grad()
            module(acts, q, beta, valid).backward()
            opt.step()
        final_loss = module(acts, q, beta, valid).item()

        assert final_loss < initial_loss
