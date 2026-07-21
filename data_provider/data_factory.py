from data_provider.graph_data_loader import (
    AsyncLifecyclePEMS,
    DynamicPEMS,
    LifecycleCoveragePEMS,
    LifecycleTrainPEMS,
    StaticPEMS,
)
from torch.utils.data import DataLoader


def data_provider(args, flag):
    if args.test_for_async_lifecycle and flag == "test":
        Dataset = AsyncLifecyclePEMS
    elif (
        args.train_for_async_lifecycle
        and args.lifecycle_training_strategy in {"lcbt_v1", "lcbt_v2"}
        and flag in {"train", "val"}
    ):
        Dataset = LifecycleCoveragePEMS
    elif args.train_for_async_lifecycle and flag in {"train", "val", "test"}:
        Dataset = LifecycleTrainPEMS
    elif args.test_for_changed:
        Dataset = DynamicPEMS
    else:
        Dataset = StaticPEMS
    if flag != 'train':
        shuffle_flag = False
        drop_last = False
        batch_size = 1 if args.test_for_async_lifecycle and flag == "test" else 256
        # if not (args.test_for_changed
        #  or args.inference_only) else 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = args.lifecycle_training_strategy not in {"lcbt_v1", "lcbt_v2"}
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
            ples_temporal_blocks=args.ples_temporal_blocks,
            ples_cohort_count=args.ples_cohort_count,
            ples_cohort_fraction=args.ples_cohort_fraction,
            ples_min_event_gap=args.ples_min_event_gap,
            ples_seed=args.ples_seed,
            ples_bootstrap_samples=args.ples_bootstrap_samples,
            ples_bootstrap_seed=args.ples_bootstrap_seed,
            ples_protocol_version=args.ples_protocol_version,
            lce_protocol_version=args.lce_protocol_version,
            ples_manifest_root=args.ples_manifest_root,
            ples_overwrite_manifest=args.ples_overwrite_manifest,
        )
        if args.lifecycle_training_strategy in {"lcbt_v1", "lcbt_v2"}:
            dataset_kwargs.update(
                lcbt_manifest_root=args.lcbt_manifest_root,
                lcbt_scaler_end_ratio=args.lcbt_scaler_end_ratio,
                lcbt_test_start_ratio=args.lcbt_test_start_ratio,
                lcbt_protocol_version=(
                    2 if args.lifecycle_training_strategy == "lcbt_v2" else 1
                ),
                lcbt_cohort_fractions=args.lcbt_cohort_fractions,
                lcbt_cohort_weights=args.lcbt_cohort_weights,
            )
    elif Dataset is LifecycleCoveragePEMS:
        dataset_kwargs.update(
            lifecycle_role_features=args.lifecycle_role_features,
            lifecycle_history_age_feature=args.lifecycle_history_age_feature,
            lifecycle_observation_mask_feature=args.lifecycle_observation_mask_feature,
            lifecycle_aas_loss_weight=args.lifecycle_aas_loss_weight,
            lifecycle_nas_loss_weight=args.lifecycle_nas_loss_weight,
            lifecycle_vs_loss_weight=args.lifecycle_vs_loss_weight,
            lifecycle_fs_loss_weight=args.lifecycle_fs_loss_weight,
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
            lcbt_protocol_version=(
                2 if args.lifecycle_training_strategy == "lcbt_v2" else 1
            ),
            lcbt_unseen_validation_ratio=args.lcbt_unseen_validation_ratio,
            lcbt_generated_unseen_fraction=args.lcbt_generated_unseen_fraction,
            lifecycle_age_mode=args.lifecycle_age_mode,
            lcbt_overwrite_manifest=args.lcbt_overwrite_manifest,
        )
    elif Dataset is LifecycleTrainPEMS:
        dataset_kwargs.update(
            lifecycle_seed=args.seed,
            async_num_events=args.async_num_events,
            lifecycle_event_sampling=args.lifecycle_event_sampling,
            lifecycle_training_strategy=args.lifecycle_training_strategy,
            lifecycle_role_features=args.lifecycle_role_features,
            lifecycle_history_age_feature=args.lifecycle_history_age_feature,
            lifecycle_aas_loss_weight=args.lifecycle_aas_loss_weight,
            lifecycle_nas_loss_weight=args.lifecycle_nas_loss_weight,
            lifecycle_vs_loss_weight=args.lifecycle_vs_loss_weight,
            lifecycle_fs_loss_weight=args.lifecycle_fs_loss_weight,
        )

    data_set = Dataset(**dataset_kwargs)

    return DataLoader(data_set,
                      batch_size=batch_size,
                      shuffle=shuffle_flag,
                      num_workers=args.num_workers,
                      drop_last=drop_last)
