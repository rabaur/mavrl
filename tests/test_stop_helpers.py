"""Unit tests for shared building blocks introduced by the STOP IG fix.

Covers:
- :func:`mavrl.feedback.stop.to_sample_inputs`: shape/dtype/device contract.
- :meth:`mavrl.utils.policies.TabularQValueModel.from_reward_matrix`:
  alternative constructor produces the same ``Q_optimal`` as the env-based
  ``__init__`` when fed the env's own reward matrix.
"""

import numpy as np
import pytest
import torch

from mavrl.feedback.stop import to_sample_inputs
from mavrl.types import DataKey
from mavrl.utils.policies import TabularQValueModel


def _segment(states, actions, valid):
    return {
        DataKey.STATES: np.asarray(states),
        DataKey.OBS: np.asarray(states),
        DataKey.ACTS: np.asarray(actions),
        DataKey.VALID: np.asarray(valid),
    }


# --------------------------------------------------------------------------- #
# to_sample_inputs                                                            #
# --------------------------------------------------------------------------- #

class TestToSampleInputs:

    def _fake_q(self, n_actions: int = 3):
        # Q(s, a) = s * 0.1 + a -> deterministic, easy to inspect.
        def q_fn(states):
            states = np.asarray(states).reshape(-1, 1)
            actions = np.arange(n_actions).reshape(1, -1)
            return states * 0.1 + actions
        return q_fn

    def test_shapes_and_dtypes(self):
        seg_len, n_actions = 4, 3
        segs = [
            _segment([0, 1, 2, 3], [0, 1, 2, 0], [1, 1, 1, 1]),
            _segment([4, 5, 6, 7], [1, 1, 0, 2], [1, 1, 1, 0]),
        ]
        q_fn = self._fake_q(n_actions=n_actions)
        q_t, acts_t, valid_t = to_sample_inputs(segs, q_fn)

        assert q_t.shape == (2, seg_len, n_actions)
        assert acts_t.shape == (2, seg_len, 1)
        assert valid_t.shape == (2, seg_len)
        assert q_t.dtype == torch.float32
        assert acts_t.dtype == torch.int64
        assert valid_t.dtype == torch.float32

    def test_squeezes_trailing_singleton_state_dim(self):
        # Tabular grid envs store states as ``(T, 1)`` due to ``np.atleast_1d``.
        states = np.array([[0], [1], [2], [3]], dtype=np.int64)
        seg = _segment(states, [0, 1, 0, 1], [1, 1, 1, 1])
        q_t, _, _ = to_sample_inputs([seg], self._fake_q(n_actions=2))
        # If the singleton wasn't squeezed, q_fn would have produced an
        # extra trailing dim (B=1, T=4, 1, A=2) instead of (1, 4, 2).
        assert q_t.shape == (1, 4, 2)

    def test_q_values_match_q_fn_output(self):
        seg = _segment([2, 5, 7], [0, 1, 1], [1, 1, 1])
        q_fn = self._fake_q(n_actions=2)
        q_t, _, _ = to_sample_inputs([seg], q_fn)
        expected = q_fn(np.array([2, 5, 7]))
        np.testing.assert_allclose(q_t.cpu().numpy().squeeze(0), expected)

    def test_falls_back_to_obs_when_states_missing(self):
        seg = {
            DataKey.OBS: np.array([1, 3, 5, 7], dtype=np.int64),
            DataKey.ACTS: np.array([0, 0, 0, 0], dtype=np.int64),
            DataKey.VALID: np.array([1, 1, 1, 1], dtype=bool),
        }
        q_t, _, _ = to_sample_inputs([seg], self._fake_q(n_actions=2))
        assert q_t.shape == (1, 4, 2)

    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            to_sample_inputs([], self._fake_q())

    def test_device_kwarg_routes_tensors(self):
        seg = _segment([0, 1], [0, 1], [1, 1])
        q_t, acts_t, valid_t = to_sample_inputs(
            [seg], self._fake_q(n_actions=2), device="cpu",
        )
        assert q_t.device.type == "cpu"
        assert acts_t.device.type == "cpu"
        assert valid_t.device.type == "cpu"


# --------------------------------------------------------------------------- #
# TabularQValueModel.from_reward_matrix                                       #
# --------------------------------------------------------------------------- #

class TestFromRewardMatrix:

    def _chain_PR(self, n_states: int = 4, n_actions: int = 2):
        """Same chain MDP as `tests/test_simulate_response.py` for parity."""
        P = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
        for s in range(n_states):
            P[s, 0, s] = 1.0
            P[s, 1, min(s + 1, n_states - 1)] = 1.0
        # Reward of 1 at the goal state, 0 elsewhere; broadcast to (s, a, s').
        r_vec = np.zeros(n_states, dtype=np.float32)
        r_vec[-1] = 1.0
        R = np.broadcast_to(
            r_vec[:, None, None], (n_states, n_actions, n_states),
        ).astype(np.float32)
        return P, R, r_vec

    def test_matches_env_constructor_with_same_reward(self):
        """``from_reward_matrix(R, P, gamma)`` should reproduce ``__init__``."""
        # Build a minimal stand-in env exposing the two methods __init__ uses.
        P, R, _ = self._chain_PR()

        class _StubEnv:
            def get_reward_matrix(self_):
                return R

            def get_transition_matrix(self_):
                return P

        gamma = 0.95
        env_based = TabularQValueModel(_StubEnv(), gamma=gamma)
        explicit = TabularQValueModel.from_reward_matrix(R, P, gamma)
        np.testing.assert_allclose(env_based.Q_optimal, explicit.Q_optimal)

    def test_q_values_method_works_on_built_model(self):
        P, R, _ = self._chain_PR()
        m = TabularQValueModel.from_reward_matrix(R, P, gamma=0.9)
        q_single = m.q_values(0)
        q_batch = m.q_values(np.array([0, 1, 2]))
        assert q_single.shape == (2,)        # n_actions
        assert q_batch.shape == (3, 2)       # (T, n_actions)

    def test_optimal_action_in_chain_is_advance(self):
        """Sanity check: VI should pick action ``1`` (advance) at every state."""
        P, R, _ = self._chain_PR(n_states=5)
        m = TabularQValueModel.from_reward_matrix(R, P, gamma=0.9)
        # Last state is absorbing with reward 1, so both actions are tied
        # there; check non-terminal states only.
        for s in range(4):
            assert int(np.argmax(m.Q_optimal[s])) == 1, (
                f"Optimal action at non-terminal state {s} should be 'advance'"
            )

    def test_different_rewards_yield_different_qs(self):
        """Per-posterior-sample factory contract: different R ⇒ different Q*."""
        P, _, _ = self._chain_PR()
        n_states, n_actions = P.shape[0], P.shape[1]

        r1 = np.zeros(n_states, dtype=np.float32); r1[-1] = 1.0
        r2 = np.zeros(n_states, dtype=np.float32); r2[0] = 1.0  # reward at start

        R1 = np.broadcast_to(r1[:, None, None], (n_states, n_actions, n_states))
        R2 = np.broadcast_to(r2[:, None, None], (n_states, n_actions, n_states))

        Q1 = TabularQValueModel.from_reward_matrix(R1, P, 0.9).Q_optimal
        Q2 = TabularQValueModel.from_reward_matrix(R2, P, 0.9).Q_optimal
        assert not np.allclose(Q1, Q2)
