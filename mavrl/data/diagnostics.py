import numpy as np
from mavrl.utils.gym import get_undiscounted_return
from mavrl.types import DataKey, Trajectory


def print_preference_stats(prefs: list[Trajectory], name: str) -> None:
    """Print statistics about the preference dataset.
    
    Args:
        preferences: Array of preference probabilities (probability that traj1 > traj2)
        cum_rews: Array of cumulative rewards for each pair, shape (num_samples, 2)
        name: Dataset name for display
        threshold: Distance from 0.5 to consider a preference "meaningful" (default 0.3 means <0.2 or >0.8)
    """
    print("-"*80)
    print(f"DATA SUMMARY Preferences: {name}")
    print("-"*80)

    cum_rews = np.array([np.nansum(p[DataKey.REWS]) for p in prefs])

    mean_reward = np.nanmean(cum_rews)
    std_reward = np.nanstd(cum_rews)
    min_reward = np.nanmin(cum_rews)
    max_reward = np.nanmax(cum_rews)

    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} [{min_reward:.1f}, {max_reward:.1f}]")


def print_preference_pair_diagnostics(prefs: list[dict], beta: float, name: str) -> None:
    """Print detailed diagnostic statistics about preference pairs.
    
    Args:
        prefs: List of preference pair dictionaries containing rewards, preferences, and valid masks
        beta: Rationality parameter used for preference generation
        name: Dataset name for display
    """
    print("="*80)
    print(f"PREFERENCE PAIR DIAGNOSTICS: {name}")
    print("="*80)
    
    preference_probs = []
    return_pairs = []
    validity_ratios = []
    
    for pref_pair in prefs:
        p = pref_pair[DataKey.PREFERENCE]
        preference_probs.append(p)
        
        rewards = pref_pair[DataKey.REWS]
        r1 = np.nansum(rewards[0])
        r2 = np.nansum(rewards[1])
        return_pairs.append((r1, r2))
        
        valid = pref_pair[DataKey.VALID]
        validity_ratios.append(np.mean(valid))
    
    preference_probs = np.array(preference_probs)
    validity_ratios = np.array(validity_ratios)
    
    # 1. Preference probability distribution
    print("\n[1] PREFERENCE PROBABILITY DISTRIBUTION")
    print(f"    Beta (rationality): {beta}")
    print(f"    Mean p: {np.mean(preference_probs):.4f}")
    print(f"    Std p:  {np.std(preference_probs):.4f}")
    print(f"    Min p:  {np.min(preference_probs):.4f}")
    print(f"    Max p:  {np.max(preference_probs):.4f}")
    
    # Histogram bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(preference_probs, bins=bins)
    print(f"\n    Preference probability histogram:")
    for i in range(len(bins)-1):
        bar = "█" * (hist[i] * 50 // max(max(hist), 1))
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:5d} {bar}")
    
    # Check for weak signal (concentrated around 0.5)
    weak_signal_ratio = np.mean((preference_probs > 0.4) & (preference_probs < 0.6))
    strong_signal_ratio = np.mean((preference_probs < 0.2) | (preference_probs > 0.8))
    print(f"\n    ⚠️  Weak signal (p in [0.4, 0.6]): {weak_signal_ratio*100:.1f}%")
    print(f"    ✓  Strong signal (p < 0.2 or p > 0.8): {strong_signal_ratio*100:.1f}%")
    
    # 2. Mean return differences (normalized by valid steps)
    mean_return_diffs = []
    for pref_pair in prefs:
        rewards = pref_pair[DataKey.REWS]
        valid = pref_pair[DataKey.VALID]
        r1_mean = np.nansum(rewards[0]) / max(np.sum(valid[0]), 1)
        r2_mean = np.nansum(rewards[1]) / max(np.sum(valid[1]), 1)
        mean_return_diffs.append(r1_mean - r2_mean)
    mean_return_diffs = np.array(mean_return_diffs)
    
    print(f"\n[2] RETURN DIFFERENCES - MEAN (normalized by valid steps)")
    print(f"    Mean diff:  {np.mean(mean_return_diffs):.4f}")
    print(f"    Std diff:   {np.std(mean_return_diffs):.4f}")
    print(f"    Min diff:   {np.min(mean_return_diffs):.4f}")
    print(f"    Max diff:   {np.max(mean_return_diffs):.4f}")
    print(f"    Abs mean:   {np.mean(np.abs(mean_return_diffs)):.4f}")
    
    logits_mean = beta * mean_return_diffs
    print(f"\n    Logits (beta * mean_diff):")
    print(f"    Mean logit: {np.mean(logits_mean):.4f}")
    print(f"    Std logit:  {np.std(logits_mean):.4f}")
    print(f"    Min logit:  {np.min(logits_mean):.4f}")
    print(f"    Max logit:  {np.max(logits_mean):.4f}")
    
    saturated_mean = np.sum(np.abs(logits_mean) > 10)
    print(f"    ⚠️  Saturated (|logit| > 10): {saturated_mean} ({saturated_mean/len(logits_mean)*100:.1f}%)")
    
    # 3. Segment validity
    print("\n[3] SEGMENT VALIDITY")
    print(f"    Mean validity ratio: {np.mean(validity_ratios):.4f}")
    print(f"    Min validity ratio:  {np.min(validity_ratios):.4f}")
    print(f"    Max validity ratio:  {np.max(validity_ratios):.4f}")
    
    low_validity = np.sum(validity_ratios < 0.5)
    print(f"    ⚠️  Segments with <50% valid steps: {low_validity} ({low_validity/len(validity_ratios)*100:.1f}%)")
    
    # 4. Individual trajectory returns
    print("\n[4] TRAJECTORY RETURNS")
    r1_vals = np.array([p[0] for p in return_pairs])
    r2_vals = np.array([p[1] for p in return_pairs])
    print(f"    Traj 1: {np.mean(r1_vals):.2f} ± {np.std(r1_vals):.2f} [{np.min(r1_vals):.2f}, {np.max(r1_vals):.2f}]")
    print(f"    Traj 2: {np.mean(r2_vals):.2f} ± {np.std(r2_vals):.2f} [{np.min(r2_vals):.2f}, {np.max(r2_vals):.2f}]")
    
    print("="*80)


def print_demonstration_stats(episodes: list[Trajectory], name: str) -> None:
    """Print concise statistics about the demonstration dataset."""
    print("-"*80)
    print(f"DATA SUMMARY Demonstrations: {name}")
    print("-"*80)

    returns = [get_undiscounted_return(episode) for episode in episodes]
    lengths = [len(episode[DataKey.REWS]) for episode in episodes]
    returns = np.array(returns)
    lengths = np.array(lengths)
    print(f"Demos: {len(returns)} | Steps: {lengths.sum()}")
    print(f"Return: {returns.mean():.1f} ± {returns.std():.1f} [{returns.min():.1f}, {returns.max():.1f}]")
    print(f"Length: {lengths.mean():.1f} ± {lengths.std():.1f} [{lengths.min()}, {lengths.max()}]")


def print_demonstration_stats(episodes: list[Trajectory], name: str) -> None:
    """Print concise statistics about the demonstration dataset."""
    print("-"*80)
    print(f"DATA SUMMARY Demonstrations: {name}")
    print("-"*80)

    returns = [get_undiscounted_return(episode) for episode in episodes]
    lengths = [len(episode[DataKey.REWS]) for episode in episodes]
    returns = np.array(returns)
    lengths = np.array(lengths)
    print(f"Demos: {len(returns)} | Steps: {lengths.sum()}")
    print(f"Return: {returns.mean():.1f} ± {returns.std():.1f} [{returns.min():.1f}, {returns.max():.1f}]")
    print(f"Length: {lengths.mean():.1f} ± {lengths.std():.1f} [{lengths.min()}, {lengths.max()}]")


def print_rating_stats(segments: list[Trajectory], name: str) -> None:
    """Print statistics about the rating dataset segments.
    
    Args:
        segments: List of trajectory segments
        name: Dataset name for display
    """
    print("-"*80)
    print(f"DATA SUMMARY Ratings: {name}")
    print("-"*80)

    cum_rews = np.array([np.nansum(seg[DataKey.REWS]) for seg in segments])

    mean_reward = np.nanmean(cum_rews)
    std_reward = np.nanstd(cum_rews)
    min_reward = np.nanmin(cum_rews)
    max_reward = np.nanmax(cum_rews)

    print(f"Number of segments: {len(segments)}")
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} [{min_reward:.1f}, {max_reward:.1f}]")


def print_rating_diagnostics(
    segments: list[dict], 
    ratings: np.ndarray,
    cutpoints: np.ndarray,
    num_categories: int,
    name: str
) -> None:
    """Print detailed diagnostic statistics about ratings.
    
    Args:
        segments: List of segment dictionaries
        ratings: Array of assigned ordinal ratings
        cutpoints: Array of cutpoints used for rating assignment
        num_categories: Number of ordinal categories
        name: Dataset name for display
    """
    print("="*80)
    print(f"RATING DIAGNOSTICS: {name}")
    print("="*80)
    
    # Compute returns
    returns = np.array([np.nansum(seg[DataKey.REWS]) for seg in segments])
    
    # Validity ratios
    validity_ratios = np.array([np.mean(seg[DataKey.VALID]) for seg in segments])
    
    # 1. Return distribution
    print("\n[1] RETURN DISTRIBUTION")
    print(f"    Mean:   {np.mean(returns):.2f}")
    print(f"    Std:    {np.std(returns):.2f}")
    print(f"    Min:    {np.min(returns):.2f}")
    print(f"    Max:    {np.max(returns):.2f}")
    
    # 2. Cutpoints
    print(f"\n[2] CUTPOINTS (quantile-based)")
    print(f"    Number of categories: {num_categories}")
    print(f"    Cutpoint values: {cutpoints}")
    
    # Show quantile boundaries
    quantiles = [100 * (i+1) / num_categories for i in range(num_categories - 1)]
    for i, (q, c) in enumerate(zip(quantiles, cutpoints)):
        print(f"    θ_{i+1} ({q:.0f}th percentile): {c:.2f}")
    
    # 3. Rating distribution
    print(f"\n[3] RATING DISTRIBUTION")
    rating_counts = np.bincount(ratings, minlength=num_categories)
    for k in range(num_categories):
        pct = rating_counts[k] / len(ratings) * 100
        bar = "█" * int(pct / 2)
        print(f"    Category {k}: {rating_counts[k]:5d} ({pct:5.1f}%) {bar}")
    
    # 4. Returns per category
    print(f"\n[4] RETURNS PER CATEGORY")
    for k in range(num_categories):
        mask = ratings == k
        if np.any(mask):
            cat_returns = returns[mask]
            print(f"    Category {k}: {np.mean(cat_returns):.2f} ± {np.std(cat_returns):.2f} [{np.min(cat_returns):.2f}, {np.max(cat_returns):.2f}]")
    
    # 5. Segment validity
    print(f"\n[5] SEGMENT VALIDITY")
    print(f"    Mean validity ratio: {np.mean(validity_ratios):.4f}")
    print(f"    Min validity ratio:  {np.min(validity_ratios):.4f}")
    print(f"    Max validity ratio:  {np.max(validity_ratios):.4f}")
    
    low_validity = np.sum(validity_ratios < 0.5)
    print(f"    ⚠️  Segments with <50% valid steps: {low_validity} ({low_validity/len(validity_ratios)*100:.1f}%)")
    
    print("="*80)


def print_stop_stats(episodes: list[Trajectory], name: str) -> None:
    """Print statistics about the stop dataset episodes."""
    print("-" * 80)
    print(f"DATA SUMMARY Stops: {name}")
    print("-" * 80)
    
    cum_rews = np.array([np.nansum(ep[DataKey.REWS]) for ep in episodes])
    lengths = np.array([len(ep[DataKey.REWS]) for ep in episodes])
    
    print(f"Episodes: {len(episodes)}")
    print(f"Return: {np.mean(cum_rews):.1f} ± {np.std(cum_rews):.1f} [{np.min(cum_rews):.1f}, {np.max(cum_rews):.1f}]")
    print(f"Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} [{np.min(lengths)}, {np.max(lengths)}]")


def print_stop_diagnostics(
    stop_times: np.ndarray,
    segment_len: int,
    max_regrets: np.ndarray,
    lambd: float,
    c: float,
    regret_ref: float,
    regret_discount: float,
    name: str
) -> None:
    """Print diagnostic statistics about stop times."""
    print("=" * 80)
    print(f"STOP DIAGNOSTICS: {name}")
    print("=" * 80)
    
    # Regret distribution
    print(f"\n[1] MAX CUMULATIVE REGRET (per segment)")
    print(f"    Mean:   {np.mean(max_regrets):.2f}")
    print(f"    Std:    {np.std(max_regrets):.2f}")
    print(f"    25th:   {np.percentile(max_regrets, 25):.2f}")
    print(f"    50th:   {np.percentile(max_regrets, 50):.2f}")
    print(f"    75th:   {np.percentile(max_regrets, 75):.2f}")
    print(f"    Max:    {np.max(max_regrets):.2f}")
    
    # Lambda calibration
    print(f"\n[2] LAMBDA CALIBRATION")
    print(f"    c (scale):          {c}")
    print(f"    regret_ref:         {regret_ref:.2f}")
    print(f"    regret_discount:    {regret_discount:.2f}")
    print(f"    lambda:             {lambd:.6f}")
    
    # Stop time distribution
    valid_stops = stop_times[stop_times >= 0]
    censored = np.sum(stop_times < 0)
    
    print(f"\n[3] STOP TIME DISTRIBUTION (segment_len={segment_len})")
    print(f"    Stopped:    {len(valid_stops)} ({len(valid_stops)/len(stop_times)*100:.1f}%)")
    print(f"    Censored:   {censored} ({censored/len(stop_times)*100:.1f}%)")
    
    if len(valid_stops) > 0:
        print(f"    Mean stop:  {np.mean(valid_stops):.1f}")
        print(f"    Std stop:   {np.std(valid_stops):.1f}")
        print(f"    Min stop:   {np.min(valid_stops)}")
        print(f"    Max stop:   {np.max(valid_stops)}")
        
        # Relative stop position (stop_time / segment_len)
        rel_stops = valid_stops / segment_len
        print(f"    Relative (t/T): {np.mean(rel_stops):.2f} ± {np.std(rel_stops):.2f}")
    
    print("=" * 80)