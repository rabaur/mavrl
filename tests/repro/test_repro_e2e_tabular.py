"""Layer 3: end-to-end reproducibility on tabular envs.

Runs ``run_experiment`` twice with the same args and asserts:

- The per-epoch metrics captured via ``db_callback`` are identical.
- The final ``result["final_metrics"]`` dict is identical.
- ``result["best_epoch"]`` and ``result["early_stop_epoch"]`` agree.
- The final ``fb_model.state_dict()`` (loaded from
  ``result["best_model_path"]``) is bitwise equal across runs.

This is the strongest pin: a regression in any of dataset gen, model init,
training, validation, model saving, or final eval will trip this test.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from train import run_experiment

from tests.repro._repro_helpers import metrics_equal, state_dict_equal


def _run_twice(args: argparse.Namespace):
    """Run ``run_experiment`` twice, return ``(result_a, callbacks_a, result_b, callbacks_b)``.

    ``callbacks_*`` is a list of ``(epoch, dict)`` tuples captured from the
    ``db_callback`` hook for per-epoch comparison.
    """
    callbacks_a: list[tuple[int, dict]] = []
    callbacks_b: list[tuple[int, dict]] = []

    def cb_a(epoch, metrics):
        callbacks_a.append((epoch, dict(metrics)))

    def cb_b(epoch, metrics):
        callbacks_b.append((epoch, dict(metrics)))

    result_a = run_experiment(args, db_callback=cb_a)
    result_b = run_experiment(args, db_callback=cb_b)
    return result_a, callbacks_a, result_b, callbacks_b


def _drop_volatile_keys(metrics: dict) -> dict:
    """Strip metric keys that may legitimately vary across runs (timing).

    Currently nothing is dropped from validation metrics, but
    ``training_time_sec`` in the result dict is wall-clock and must be
    excluded before comparing top-level results.
    """
    return {k: v for k, v in metrics.items() if k != "training_time_sec"}


def _drop_volatile_result_keys(result: dict) -> dict:
    """Strip result-dict keys that vary across runs (paths, timings)."""
    return {
        k: v for k, v in result.items()
        if k not in {"training_time_sec", "best_model_path", "wandb_run_id"}
    }


def _assert_e2e_reproducible(args: argparse.Namespace) -> None:
    result_a, cbs_a, result_b, cbs_b = _run_twice(args)

    # ── Per-epoch metrics ────────────────────────────────────────────────
    assert len(cbs_a) == len(cbs_b), (
        f"different number of validation callbacks: {len(cbs_a)} vs {len(cbs_b)}"
    )
    for (ep_a, m_a), (ep_b, m_b) in zip(cbs_a, cbs_b):
        assert ep_a == ep_b, f"callback epoch mismatch: {ep_a} vs {ep_b}"
        ok, reason = metrics_equal(m_a, m_b)
        assert ok, f"epoch {ep_a}: {reason}"

    # ── Final metrics ────────────────────────────────────────────────────
    ok, reason = metrics_equal(result_a["final_metrics"], result_b["final_metrics"])
    assert ok, f"final_metrics differ: {reason}"

    # ── Best epoch / early stop ──────────────────────────────────────────
    for k in ("best_epoch", "early_stop_epoch", "early_stop_reason"):
        assert result_a[k] == result_b[k], (
            f"{k} differs: {result_a[k]!r} vs {result_b[k]!r}"
        )

    # ── Saved model state dict ───────────────────────────────────────────
    if result_a["best_model_path"] is not None:
        assert result_b["best_model_path"] is not None
        ckpt_a = torch.load(result_a["best_model_path"], map_location="cpu",
                            weights_only=False)
        ckpt_b = torch.load(result_b["best_model_path"], map_location="cpu",
                            weights_only=False)
        ok, reason = state_dict_equal(
            ckpt_a["model_state_dict"], ckpt_b["model_state_dict"],
        )
        assert ok, f"best_model state_dict differs: {reason}"
        assert ckpt_a["epoch"] == ckpt_b["epoch"]
        assert ckpt_a["val_loss"] == ckpt_b["val_loss"]


# ── Test cases ───────────────────────────────────────────────────────────────


def test_e2e_grid_trap_reproducible(tabular_args):
    """grid_trap with all 4 modalities is bitwise reproducible end-to-end."""
    _assert_e2e_reproducible(tabular_args)


def test_e2e_chain_reproducible(chain_args):
    """chain env (different action space) is bitwise reproducible end-to-end."""
    _assert_e2e_reproducible(chain_args)


def test_e2e_tabular_single_modality(tabular_args):
    """Sanity: a single-modality run is reproducible too.

    Useful diagnostic when the multi-modality test fails: if this passes
    but ``test_e2e_grid_trap_reproducible`` fails, the issue is in the
    cross-modality interaction (most likely the shared dataloader generator
    in [make_dataset.py:34](mavrl/data/make_dataset.py)).
    """
    args = argparse.Namespace(**vars(tabular_args))
    args.n_demo_samples = 0
    args.n_rating_samples = 0
    args.n_stop_samples = 0
    _assert_e2e_reproducible(args)
