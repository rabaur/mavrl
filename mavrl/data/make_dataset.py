from pathlib import Path
from typing import Optional
import torch
from mavrl.data.preference_dataset import PreferenceDataset
from mavrl.data.demonstration_dataset import DemonstrationDataset
from mavrl.data.rating_dataset import RatingDataset
from mavrl.data.stop_dataset import StopDataset
from mavrl.data.utils import derive_seed
from mavrl.data.cache import (
    compute_cache_key,
    save_dataset_to_cache,
    load_dataset_from_cache,
    _cache_path,
    _slice_data,
    slide_demo_data,
)
from mavrl.types import FeedbackType
from mavrl.utils.policies import QValueModel
from torch.utils.data import DataLoader

def make_dataset(
    active_feedback_types,
    args,
    make_env_fn,
    policies,
    device,
    obs_transform,
    action_transform,
    name: Optional[str] = "train",
    q_true: Optional[QValueModel] = None,
):
    datasets = {}
    dataloaders = {}
    g = torch.Generator()
    g.manual_seed(args.seed)

    cache_dir = Path(args.dataset_cache_dir).expanduser() if getattr(args, "dataset_cache_dir", None) else None
    cache_gen = getattr(args, "dataset_cache_gen_samples", None)

    # ── Preference ───────────────────────────────────────────────────────
    if FeedbackType.PREF in active_feedback_types:
        num_samples = args.n_pref_samples if name == "train" else max(int(args.n_pref_samples * 1.0), 1)
        pref_dataset = _try_load_cache(cache_dir, FeedbackType.PREF, args, name, num_samples, device)

        if pref_dataset is None:
            gen_samples = max(num_samples, cache_gen) if cache_gen else num_samples
            base_seed = derive_seed(args.seed, FeedbackType.PREF.value, name)
            pref_dataset = PreferenceDataset(
                base_seed=base_seed,
                num_episodes=args.n_pref_episodes,
                num_pref_pairs=gen_samples,
                policy=policies[FeedbackType.PREF],
                make_env_fn=make_env_fn,
                device=device,
                beta=args.pref_data_rationality,
                model_beta=args.pref_model_rationality,
                gamma=args.gamma,
                obs_transform=obs_transform,
                act_transform=action_transform,
                segment_len=args.pref_seg_len,
                name=name,
                min_reward_threshold=args.min_reward_pref,
                td_error_weight=args.td_error_weight,
            )
            _maybe_save_cache(cache_dir, FeedbackType.PREF, args, name, pref_dataset)
            if gen_samples > num_samples:
                pref_dataset.data = _slice_data(pref_dataset.data, num_samples)

        pref_dataset._rationality = torch.tensor(
            args.pref_model_rationality, dtype=torch.float32, device=device
        )

        datasets[FeedbackType.PREF] = pref_dataset
        dataloaders[FeedbackType.PREF] = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        print(f"Created preference dataset with {len(pref_dataset)} samples")

    # ── Demonstration ────────────────────────────────────────────────────
    if FeedbackType.DEMO in active_feedback_types:
        num_samples = args.n_demo_samples if name == "train" else max(int(args.n_demo_samples * 1.0), 1)
        demo_dataset = _try_load_cache(cache_dir, FeedbackType.DEMO, args, name, num_samples, device)

        if demo_dataset is None:
            gen_samples = max(num_samples, cache_gen) if cache_gen else num_samples
            base_seed = derive_seed(args.seed, FeedbackType.DEMO.value, name)
            demo_dataset = DemonstrationDataset(
                base_seed=base_seed,
                num_demonstrations=gen_samples,
                num_steps=None,
                make_env_fn=make_env_fn,
                policy=policies[FeedbackType.DEMO],
                device=device,
                beta=args.demo_rationality,
                model_beta=args.demo_model_rationality,
                gamma=args.gamma,
                td_error_weight=args.td_error_weight,
                obs_transform=obs_transform,
                act_transform=action_transform,
                name=name,
                step_offset=args.step_offset,
                min_reward_threshold=args.min_reward_demo,
                subsample_factor=args.subsample_factor
            )
            _maybe_save_cache(cache_dir, FeedbackType.DEMO, args, name, demo_dataset)
            if gen_samples > num_samples:
                demo_dataset.data, demo_dataset.episode_lengths = slide_demo_data(
                    demo_dataset.data, demo_dataset.episode_lengths, num_samples
                )

        _demo_model_beta = args.demo_model_rationality if args.demo_model_rationality is not None else args.demo_rationality
        demo_dataset._rationality = torch.tensor(
            _demo_model_beta, dtype=torch.float32, device=device
        )

        datasets[FeedbackType.DEMO] = demo_dataset
        dataloaders[FeedbackType.DEMO] = DataLoader(demo_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        print(f"Created demonstration dataset with {len(demo_dataset)} samples")

    # ── Rating ───────────────────────────────────────────────────────────
    if FeedbackType.RATE in active_feedback_types:
        num_samples = args.n_rating_samples if name == "train" else max(int(args.n_rating_samples * 1.0), 1)
        rating_dataset = _try_load_cache(cache_dir, FeedbackType.RATE, args, name, num_samples, device)

        if rating_dataset is None:
            gen_samples = max(num_samples, cache_gen) if cache_gen else num_samples
            base_seed = derive_seed(args.seed, FeedbackType.RATE.value, name)
            rating_dataset = RatingDataset(
                base_seed=base_seed,
                num_episodes=args.n_rating_episodes,
                num_samples=gen_samples,
                policy=policies[FeedbackType.RATE],
                make_env_fn=make_env_fn,
                device=device,
                num_categories=5,
                gamma=args.gamma,
                obs_transform=obs_transform,
                act_transform=action_transform,
                segment_len=args.rating_seg_len,
                name=name,
                min_reward_threshold=args.min_reward_rating,
                td_error_weight=args.td_error_weight,
                noise_std=getattr(args, "rating_noise_std", 0.0),
            )
            _maybe_save_cache(cache_dir, FeedbackType.RATE, args, name, rating_dataset)
            if gen_samples > num_samples:
                rating_dataset.data = _slice_data(rating_dataset.data, num_samples)

        datasets[FeedbackType.RATE] = rating_dataset
        dataloaders[FeedbackType.RATE] = DataLoader(rating_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        print(f"Created rating dataset with {len(rating_dataset)} samples")

    # ── Stop ─────────────────────────────────────────────────────────────
    if FeedbackType.STOP in active_feedback_types:
        if q_true is None:
            raise ValueError("q_model must be provided for stop feedback")
        num_samples = args.n_stop_samples if name == "train" else max(int(args.n_stop_samples * 1.0), 1)
        stop_dataset = _try_load_cache(cache_dir, FeedbackType.STOP, args, name, num_samples, device)

        if stop_dataset is None:
            gen_samples = max(num_samples, cache_gen) if cache_gen else num_samples
            base_seed = derive_seed(args.seed, FeedbackType.STOP.value, name)
            stop_dataset = StopDataset(
                base_seed=base_seed,
                num_episodes=args.n_stop_episodes,
                num_samples=gen_samples,
                segment_len=args.stop_seg_len,
                q_model=q_true,
                policy=policies[FeedbackType.STOP],
                make_env_fn=make_env_fn,
                device=device,
                c=args.stop_c,
                regret_percentile=args.stop_regret_percentile,
                regret_discount=args.stop_regret_discount,
                gamma=args.gamma,
                obs_transform=obs_transform,
                act_transform=action_transform,
                name=name,
                td_error_weight=args.td_error_weight,
            )
            _maybe_save_cache(cache_dir, FeedbackType.STOP, args, name, stop_dataset)
            if gen_samples > num_samples:
                stop_dataset.data = _slice_data(stop_dataset.data, num_samples)

        model_c = getattr(args, "stop_model_c", None) or args.stop_c
        if model_c != args.stop_c:
            scale = model_c / args.stop_c
            stop_dataset._lambda = torch.tensor(
                stop_dataset.lambd * scale, dtype=torch.float32, device=device
            )

        datasets[FeedbackType.STOP] = stop_dataset
        dataloaders[FeedbackType.STOP] = DataLoader(stop_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        print(f"Created stop dataset with {len(stop_dataset)} samples")

    return datasets, dataloaders


# ── Internal helpers ─────────────────────────────────────────────────────────

def _try_load_cache(cache_dir, feedback_type, args, name, n_samples, device):
    """Attempt to load a dataset from cache; returns None on miss."""
    if cache_dir is None:
        return None
    key = compute_cache_key(feedback_type, args, name)
    path = _cache_path(cache_dir, feedback_type, key)
    return load_dataset_from_cache(path, feedback_type, n_samples, device, args.td_error_weight)


def _maybe_save_cache(cache_dir, feedback_type, args, name, dataset):
    """Save dataset to cache if caching is enabled."""
    if cache_dir is None:
        return
    key = compute_cache_key(feedback_type, args, name)
    path = _cache_path(cache_dir, feedback_type, key)
    save_dataset_to_cache(path, dataset)
