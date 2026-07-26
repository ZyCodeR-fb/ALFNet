from data_provider.graph_data_loader import (
    AsyncLifecyclePEMS,
    LifecycleCoveragePEMS,
)
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import numpy as np


class LifecycleBalancedBatchSampler(Sampler):
    """Group-balanced batches backed by the dataset's epoch-specific schedule."""

    def __init__(self, dataset, batch_size, seed):
        if int(batch_size) < 4 or int(batch_size) % 4:
            raise ValueError("LCBT v3 batch_size must be divisible by four")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        groups = {name: [] for name in ("zero_history", "steady", "join", "fault")}
        for index, item in enumerate(self.dataset.schedule):
            groups[str(item["scenario"])].append(index)
        rng = np.random.default_rng(
            self.seed + int(getattr(self.dataset, "current_epoch", 0)) * 1_000_003
        )
        for name in groups:
            groups[name] = list(rng.permutation(groups[name]))
        per_group = self.batch_size // 4
        full_batches = min(len(indices) // per_group for indices in groups.values())
        cursor = {name: 0 for name in groups}
        for _ in range(full_batches):
            batch = []
            for name in groups:
                start = cursor[name]
                end = start + per_group
                batch.extend(groups[name][start:end])
                cursor[name] = end
            yield list(rng.permutation(batch))
        tail = []
        for name in groups:
            tail.extend(groups[name][cursor[name]:])
        for start in range(0, len(tail), self.batch_size):
            yield list(rng.permutation(tail[start:start + self.batch_size]))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def data_provider(args, flag):
    if args.test_for_async_lifecycle and flag == "test":
        Dataset = AsyncLifecyclePEMS
    elif (
        args.train_for_async_lifecycle
        and args.lifecycle_training_strategy == "lcbt_v3"
        and flag in {"train", "val"}
    ):
        Dataset = LifecycleCoveragePEMS
    else:
        raise ValueError(
            f"Unsupported data request flag={flag!r}; use LCBT train/val or LCE/PLES test_async"
        )
    if flag != 'train':
        shuffle_flag = False
        drop_last = False
        batch_size = 1 if args.test_for_async_lifecycle and flag == "test" else 256
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    dataset_kwargs = dict(
        data_path=args.data_path,
        size=[args.input_len, args.look_back, args.pred_len],
        flag=flag,
        scale=not args.no_scale,
        freq=freq,
        inverse=args.inverse,
        target_features=args.target_features,
        input_features=args.input_features,
        dloader_name=args.dloader_name,
        step=args.slide_step,
        splits=[args.train_ratio, args.valid_ratio],
        unknown_nodes_path=args.unknown_nodes_path,
        unknown_nodes_num=args.unknown_nodes_num,
        unknown_nodes_list=args.unknown_nodes_list,
    )
    if Dataset is AsyncLifecyclePEMS:
        dataset_kwargs.update(
            lifecycle_role_features=args.lifecycle_role_features,
            lifecycle_history_age_feature=args.lifecycle_history_age_feature,
            lifecycle_observation_mask_feature=args.lifecycle_observation_mask_feature,
            lifecycle_age_mode=args.lifecycle_age_mode,
            ples_pre_steps=args.ples_pre_steps,
            ples_post_steps=args.ples_post_steps,
            ples_seed=args.ples_seed,
            ples_bootstrap_samples=args.ples_bootstrap_samples,
            ples_bootstrap_seed=args.ples_bootstrap_seed,
            ples_protocol_version=args.ples_protocol_version,
            lce_protocol_version=args.lce_protocol_version,
            ples_manifest_root=args.ples_manifest_root,
            ples_overwrite_manifest=args.ples_overwrite_manifest,
            lifecycle_training_strategy=args.lifecycle_training_strategy,
        )
        dataset_kwargs.update(
            lcbt_manifest_root=args.lcbt_manifest_root,
            lcbt_scaler_end_ratio=args.lcbt_scaler_end_ratio,
            lcbt_test_start_ratio=args.lcbt_test_start_ratio,
            lcbt_cohort_fractions=args.lcbt_cohort_fractions,
            lcbt_cohort_weights=args.lcbt_cohort_weights,
        )
    elif Dataset is LifecycleCoveragePEMS:
        dataset_kwargs.update(
            lifecycle_role_features=args.lifecycle_role_features,
            lifecycle_history_age_feature=args.lifecycle_history_age_feature,
            lifecycle_observation_mask_feature=args.lifecycle_observation_mask_feature,
            lcbt_manifest_root=args.lcbt_manifest_root,
            lcbt_protocol_seed=args.lcbt_protocol_seed,
            lcbt_validation_seed=args.lcbt_validation_seed,
            lcbt_train_end_ratio=args.lcbt_train_end_ratio,
            lcbt_val_end_ratio=args.lcbt_val_end_ratio,
            lcbt_test_start_ratio=args.lcbt_test_start_ratio,
            lcbt_scaler_end_ratio=args.lcbt_scaler_end_ratio,
            lcbt_join_ratio=args.lcbt_join_ratio,
            lcbt_fault_ratio=args.lcbt_fault_ratio,
            lcbt_steady_ratio=args.lcbt_steady_ratio,
            lcbt_zero_history_ratio=args.lcbt_zero_history_ratio,
            lcbt_cohort_fractions=args.lcbt_cohort_fractions,
            lcbt_cohort_weights=args.lcbt_cohort_weights,
            lcbt_clustered_ratio=args.lcbt_clustered_ratio,
            lcbt_unseen_validation_ratio=args.lcbt_unseen_validation_ratio,
            lcbt_generated_unseen_fraction=args.lcbt_generated_unseen_fraction,
            lifecycle_age_mode=args.lifecycle_age_mode,
            lcbt_overwrite_manifest=args.lcbt_overwrite_manifest,
            lifecycle_training_strategy=args.lifecycle_training_strategy,
            lcbt_validation_origins_per_stratum=args.lcbt_validation_origins_per_stratum,
            lcbt_min_observed_fraction=args.lcbt_min_observed_fraction,
            lcbt_episode_hidden_fraction=args.lcbt_episode_hidden_fraction,
        )
    data_set = Dataset(**dataset_kwargs)
    if flag == "train" and args.lifecycle_training_strategy == "lcbt_v3":
        return DataLoader(
            data_set,
            batch_sampler=LifecycleBalancedBatchSampler(
                data_set,
                batch_size=batch_size,
                seed=args.lcbt_protocol_seed,
            ),
            num_workers=args.num_workers,
        )
    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
