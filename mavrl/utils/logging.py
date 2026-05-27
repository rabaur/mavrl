from typing import Any
from tabulate import tabulate
from itertools import combinations
import numpy as np


def console_log_eval_metrics(eval_metrics: dict[str, Any]):
    print(f"  Evaluation Results:")
    if "eval/epic_distance" in eval_metrics and eval_metrics["eval/epic_distance"] is not None:
        print(f"    EPIC Distance: {eval_metrics['eval/epic_distance']:.6f}")
    if "eval/regret" in eval_metrics:
        print(f"    Regret: {eval_metrics['eval/regret']:.6f}")
    if "eval/negative_log_likelihood" in eval_metrics:
        print(f"    Eval NLL: {eval_metrics['eval/negative_log_likelihood']:.4f}")
    if "eval/kl_divergence" in eval_metrics:
        print(f"    Eval KL: {eval_metrics['eval/kl_divergence']:.4f}")
    if "eval/td_error" in eval_metrics:
        print(f"    Eval TD Error: {eval_metrics['eval/td_error']:.4f}")
    print()


BOLD_RED = "\033[1;31m"
RESET = "\033[0m"

def _fmt(v: float, highlight: bool = False) -> str:
    """Format value with optional bold red for extreme ratios."""
    s = f"{v:.3f}"
    if highlight and not np.isnan(v) and (v < 0.1 or v > 10):
        return f"{BOLD_RED}{s}{RESET}"
    return s

def console_log_batch_metrics(losses: dict[str, dict[str, float]], step: int, steps_per_epoch: int, total_loss: float):
    """Display loss components in a table.
    
    Args:
        losses: Dict of dicts {feedback_type: {"nll": x, "kl": y, "td": z}}
        step: Current step
        steps_per_epoch: Total steps per epoch
        total_loss: Total loss value
    """
    components = ["nll", "kl", "td"]
    fb_types = list(losses.keys())
    formatted = []

    # Per feedback type rows
    for fb in fb_types:
        formatted.append([fb] + [_fmt(losses[fb].get(c, 0.0)) for c in components])

    # Total row
    totals = [sum(losses[fb].get(c, 0.0) for fb in fb_types) for c in components]
    formatted.append(["TOTAL"] + [_fmt(v) for v in totals])

    # Ratio rows for each pair (with highlighting)
    for fb1, fb2 in combinations(fb_types, 2):
        ratios = []
        for c in components:
            v1, v2 = losses[fb1].get(c, 0.0), losses[fb2].get(c, 0.0)
            ratios.append(v1 / v2 if v2 != 0 else float('nan'))
        formatted.append([f"{fb1}/{fb2}"] + [_fmt(v, highlight=True) for v in ratios])

    print(f"  Step {step}/{steps_per_epoch-1} | Total Loss: {total_loss:.3f}")
    print(tabulate(formatted, headers=["", "NLL", "KL", "TD"], tablefmt="rounded_grid"))


def losses_to_flat_dict(losses: dict[str, dict[str, float]]) -> dict[str, float]:
    """Convert nested losses dict to flat dict for wandb logging."""
    flat = {}
    components = set()
    for fb, comps in losses.items():
        for c, v in comps.items():
            flat[f"{c}_{fb}"] = v
            components.add(c)
    for c in components:
        flat[c] = sum(losses[fb].get(c, 0.0) for fb in losses)
    return flat
