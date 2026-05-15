from data_provider.graph_data_loader import AsyncLifecyclePEMS, DynamicPEMS, LifecycleTrainPEMS, StaticPEMS
from torch.utils.data import DataLoader


def data_provider(args, flag):
    if args.test_for_async_lifecycle and flag == "test":
        Dataset = AsyncLifecyclePEMS
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
        drop_last = True
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
            async_num_events=args.async_num_events,
            async_commissioning_k=args.async_commissioning_k,
            async_event_seed=args.async_event_seed,
            lifecycle_role_features=args.lifecycle_role_features,
            lifecycle_history_age_feature=args.lifecycle_history_age_feature,
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
