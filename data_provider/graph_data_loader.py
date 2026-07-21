import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import warnings
from data_provider.lcbt_protocol import (
    LCBT_PHASES,
    LCBT_PROTOCOL_NAME,
    LCBT_PROTOCOL_VERSION,
    array_hash,
    build_lcbt_schedule,
    calendar_regime,
    compose_effective_lifecycle_features,
    lcbt_manifest_hash,
    resolve_lcbt_boundaries,
    stable_hash,
)
from data_provider.ples_protocol import build_protocol_manifest
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_CHANNEL_COUNT = 4

ASYNC_EVENT_NODE_COUNTS = {
    207: (20, 20),
    325: (32, 32),
}
LIFECYCLE_LOCAL_VS_RATIO = 0.25
LIFECYCLE_MIXED_CLUSTER_PROB = 0.5
LIFECYCLE_TRAINING_PROFILES = {
    "v1": {
        "nas_scale": 1.0,
        "fs_scale": 1.0,
        "vs_ratio": LIFECYCLE_LOCAL_VS_RATIO,
        "event_time_mode": "uniform",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.0,
        },
    },
    "round1": {
        "nas_scale": 0.5,
        "fs_scale": 0.5,
        "vs_ratio": 0.10,
        "event_time_mode": "uniform",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.0,
        },
    },
    "round2": {
        "nas_scale": 0.5,
        "fs_scale": 0.5,
        "vs_ratio": 0.0,
        "event_time_mode": "uniform",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.0,
        },
    },
    "round3": {
        "nas_scale": 0.5,
        "fs_scale": 0.75,
        "vs_ratio": 0.05,
        "event_time_mode": "early",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.35,
            ROLE_FS: 1.60,
        },
    },
    "round3_event_only": {
        "nas_scale": 0.5,
        "fs_scale": 0.75,
        "vs_ratio": 0.05,
        "event_time_mode": "early",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.0,
        },
    },
    "round3_weight_only": {
        "nas_scale": 0.5,
        "fs_scale": 0.5,
        "vs_ratio": 0.0,
        "event_time_mode": "uniform",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.60,
        },
    },
    "round4_dense_event_only": {
        "nas_scale": 1.0,
        "fs_scale": 1.0,
        "vs_ratio": 0.10,
        "event_time_mode": "early",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.0,
            ROLE_VS: 1.0,
            ROLE_FS: 1.0,
        },
    },
    "round4_dense_weighted": {
        "nas_scale": 1.0,
        "fs_scale": 1.0,
        "vs_ratio": 0.10,
        "event_time_mode": "early",
        "role_loss_weights": {
            ROLE_AAS: 1.0,
            ROLE_NAS: 1.20,
            ROLE_VS: 1.20,
            ROLE_FS: 1.30,
        },
    },
}


def _resolve_async_event_node_counts(num_nodes):
    if num_nodes in ASYNC_EVENT_NODE_COUNTS:
        return ASYNC_EVENT_NODE_COUNTS[num_nodes]
    event_count = max(1, int(round(num_nodes * 0.10)))
    return event_count, event_count


def _unwrap_raw_data(raw_data):
    if isinstance(raw_data, np.ndarray) and raw_data.shape == ():
        return raw_data.item()
    return raw_data


def _resolve_lifecycle_training_profile(strategy):
    if strategy not in LIFECYCLE_TRAINING_PROFILES:
        raise ValueError(f"Unsupported lifecycle_training_strategy: {strategy}")
    return LIFECYCLE_TRAINING_PROFILES[strategy]


def _sample_random_nodes(eligible_nodes, count, rng):
    if count <= 0 or not eligible_nodes:
        return []
    eligible = np.array(sorted(eligible_nodes), dtype=int)
    count = min(count, len(eligible))
    return sorted(rng.choice(eligible, size=count, replace=False).tolist())


def _sample_clustered_nodes(eligible_nodes, count, adj, rng):
    if count <= 0 or not eligible_nodes:
        return []
    eligible = sorted(eligible_nodes)
    count = min(count, len(eligible))
    center = int(rng.choice(np.array(eligible, dtype=int)))
    positive_neighbors = [center]
    neighbor_scores = np.asarray(adj[center], dtype=np.float64)
    ranked_neighbors = [node for node in np.argsort(-neighbor_scores, kind="stable").tolist() if node in eligible and node != center and neighbor_scores[node] > 0]
    positive_neighbors.extend(ranked_neighbors[: max(0, count - 1)])
    if len(positive_neighbors) >= count:
        return sorted(positive_neighbors[:count])

    remaining = [node for node in eligible if node not in positive_neighbors]
    if remaining:
        supplement_count = min(count - len(positive_neighbors), len(remaining))
        supplement = rng.choice(np.array(remaining, dtype=int), size=supplement_count, replace=False).tolist()
        positive_neighbors.extend(supplement)
    return sorted(positive_neighbors[:count])


def _sample_event_nodes(eligible_nodes, count, adj, rng, sampling_mode):
    if sampling_mode == "random":
        return _sample_random_nodes(eligible_nodes, count, rng)
    if sampling_mode == "clustered":
        return _sample_clustered_nodes(eligible_nodes, count, adj, rng)
    if sampling_mode == "mixed":
        chosen_mode = "clustered" if rng.random() < LIFECYCLE_MIXED_CLUSTER_PROB else "random"
        return _sample_event_nodes(eligible_nodes, count, adj, rng, chosen_mode)
    raise ValueError(f"Unsupported lifecycle_event_sampling: {sampling_mode}")


def _build_history_age(visibility_mask, input_len):
    visible_counts = np.cumsum(visibility_mask.astype(np.float32), axis=0)
    return visible_counts / float(input_len)


def _build_role_one_hot(role_ids):
    role_features = np.zeros(role_ids.shape + (ROLE_CHANNEL_COUNT,), dtype=np.float32)
    for role_idx in range(ROLE_CHANNEL_COUNT):
        role_features[..., role_idx] = (role_ids == role_idx).astype(np.float32)
    return role_features


def _compose_lifecycle_input_features(seq_x, visibility_mask, role_ids, use_role_features, use_history_age):
    masked_seq_x = np.where(visibility_mask[..., None], seq_x, np.nan).astype(np.float32, copy=False)
    features = [masked_seq_x]
    if use_role_features:
        features.append(_build_role_one_hot(role_ids))
    if use_history_age:
        history_age = _build_history_age(visibility_mask, visibility_mask.shape[0])[..., None]
        features.append(history_age.astype(np.float32))
    if len(features) == 1:
        return masked_seq_x
    return np.concatenate(features, axis=-1).astype(np.float32, copy=False)


def _build_target_role_weights(seq_y, nas_nodes, vs_nodes, fs_nodes, role_loss_weights):
    target_weights = np.ones_like(seq_y, dtype=np.float32)
    if nas_nodes:
        target_weights[:, nas_nodes, :] = float(role_loss_weights[ROLE_NAS])
    if vs_nodes:
        target_weights[:, vs_nodes, :] = float(role_loss_weights[ROLE_VS])
    if fs_nodes:
        target_weights[:, fs_nodes, :] = float(role_loss_weights[ROLE_FS])
    return target_weights


class StaticPEMS(Dataset):

    def __init__(self,
                 data_path: str,
                 size: list[int],
                 flag: str,
                 scale: bool,
                 freq: str,
                 target_features: str,
                 input_features: str,
                 dloader_name: str,
                 splits: list,
                 step: int = 1,
                 inverse=False,
                 unknown_nodes_path='',
                 unknown_nodes_list=[],
                 unknown_nodes_num=0):
        assert flag in ['train', 'test',
                        'val'], 'flag should be train, test or val'
        fea_dict = {
            'q': [0],
            'o': [1],
            'v': [2],
            'qv': [0, 2],
            'qov': [0, 1, 2],
        }
        self.tar_feas = fea_dict[target_features]
        self.in_feas = fea_dict[input_features]
        self.flag = flag
        self.input_len = size[0]
        self.look_back = size[1]
        assert self.look_back <= self.input_len, 'look_back should be smaller than input_len'
        self.pred_len = size[2]
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.data_path = data_path
        self.unknown_nodes = unknown_nodes_list
        self.unknown_nodes_num = unknown_nodes_num
        self.unknown_nodes_path = unknown_nodes_path
        self.dloader_name = dloader_name
        self.step = step
        self.train_ratio, self.valid_ratio = splits
        self.__read_data__()

    def __read_data__(self):
        raw_data = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))

        # 时间戳 shape (time_horizon,)
        # features: timestamp
        self.time_mark = raw_data['time_stamp']
        if self.time_mark is not None:
            self.time_mark = time_features(
                self.time_mark, freq=self.freq)  #不进行sacle，因为后面用这个整数做embedding

        # 空间位置戳 shape (num_nodes,features_dim)
        # features: lon,lat
        self.pos_mark = raw_data['pos_stamp']
        # scale the value to [-0.5,0.5]
        if self.pos_mark is not None:
            self.pos_mark = (self.pos_mark -
                             np.min(self.pos_mark, keepdims=True)) / (
                                 np.max(self.pos_mark, keepdims=True) -
                                 np.min(self.pos_mark, keepdims=True)) - 0.5
        # 邻接矩阵 shape (num_nodes,num_nodes)
        self.adj = raw_data['adj']  #based on distance
        # time 按时间从小到大
        raw_data = raw_data['data']
        self.full_num_nodes = raw_data.shape[1]
        raw_data[raw_data <= 0] = np.nan
        if len(self.unknown_nodes_path):
            self.unknown_nodes = np.load(self.unknown_nodes_path)
            self.known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                            self.unknown_nodes)
        elif len(self.unknown_nodes):
            self.known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                            self.unknown_nodes)
        else:
            np.random.seed(0)
            self.unknown_nodes = np.random.choice(range(raw_data.shape[1]),
                                                  self.unknown_nodes_num,
                                                  replace=False)
            self.known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                            self.unknown_nodes)
        self.unknown_nodes = self.unknown_nodes.tolist()
        self.known_nodes = self.known_nodes.tolist()
        train_data = raw_data[:int(0.7 * len(raw_data))][:, self.known_nodes]
        if self.flag == 'train':
            self.data = train_data
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 *
                                                     len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.known_nodes]
            self.adj = self.adj[self.known_nodes][:, self.known_nodes]
        elif self.flag == 'val':
            self.data = raw_data[:int(0.7 * len(raw_data))][:,
                                                            self.known_nodes]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 *
                                                     len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.known_nodes]
            self.adj = self.adj[self.known_nodes][:, self.known_nodes]
        else:
            self.data = raw_data[int(0.7 * len(raw_data)):]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[int(0.7 *
                                                    len(self.time_mark)):]
            #历史原因，我们从self.input_len开始测试集合，这里其实可以去掉
            self.data = self.data[self.input_len:]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[self.input_len:]
        if self.scale:
            self.mean_ = np.nanmean(train_data, axis=(0, 1), keepdims=True)
            self.std_ = np.nanstd(train_data, axis=(0, 1), keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        self.data = (self.data - self.mean_) / (self.std_)
        self.data_y = self.data.copy()
        if self.flag == 'test':
            self.data[:, self.unknown_nodes] = np.nan
        self.data = self.data[..., self.in_feas]
        self.data_y = self.data_y[..., self.tar_feas]
        self.num_nodes = self.data.shape[1]

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.known_nodes, self.unknown_nodes, [], []

    def __getitem__(self, index):
        index = index * self.step
        input_begin = index
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        seq_x = self.data[input_begin:input_end]
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len -
                self.pred_len) // self.step + 1  #type:ignore

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[...,
                                                              self.tar_feas]


class DynamicPEMS(Dataset):

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path='',
        unknown_nodes_list=[],
        unknown_nodes_num=0,
    ):
        assert flag in ['train', 'test',
                        'val'], 'flag should be train, test or val'
        fea_dict = {
            'q': [0],
            'o': [1],
            'v': [2],
            'qv': [0, 2],
            'qov': [0, 1, 2],
        }
        self.tar_feas = fea_dict[target_features]
        self.in_feas = fea_dict[input_features]
        self.flag = flag
        self.input_len = size[0]
        self.look_back = size[1]
        assert self.look_back <= self.input_len, 'look_back should be smaller than input_len'
        self.pred_len = size[2]
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.data_path = data_path
        self.train_unknown_nodes = unknown_nodes_list
        self.unknown_nodes_num = unknown_nodes_num
        self.unknown_nodes_path = unknown_nodes_path
        self.dloader_name = dloader_name
        self.step = step
        self.train_ratio, self.valid_ratio = splits
        self.__read_data__()

    def __read_data__(self):
        raw_data = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))

        # 时间戳 shape (time_horizon,)
        # features: timestamp
        self.time_mark = raw_data['time_stamp']
        if self.time_mark is not None:
            self.time_mark = time_features(
                self.time_mark, freq=self.freq)  #不进行sacle，因为后面用这个整数做embedding

        # 空间位置戳 shape (num_nodes,features_dim)
        # features: lon,lat
        self.pos_mark = raw_data['pos_stamp']
        # scale the value to [-0.5,0.5]
        if self.pos_mark is not None:
            self.pos_mark = (self.pos_mark -
                             np.min(self.pos_mark, keepdims=True)) / (
                                 np.max(self.pos_mark, keepdims=True) -
                                 np.min(self.pos_mark, keepdims=True)) - 0.5
        # 邻接矩阵 shape (num_nodes,num_nodes)
        self.adj = raw_data['adj']  #based on distance
        # assume input data shape (time_horizon, num_nodes, features_dim)
        # time 按时间从小到大
        raw_data = raw_data['data']
        self.full_num_nodes = raw_data.shape[1]
        num_nodes = raw_data.shape[1]
        num_add_nodes, num_failed_nodes = _resolve_async_event_node_counts(num_nodes)
        raw_data[raw_data <= 0] = np.nan
        if len(self.unknown_nodes_path):
            self.train_unknown_nodes = np.load(self.unknown_nodes_path)
            self.train_known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                                  self.train_unknown_nodes)
        elif len(self.train_unknown_nodes):
            self.train_known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                                  self.train_unknown_nodes)
        else:
            np.random.seed(0)
            self.train_unknown_nodes = np.random.choice(range(
                raw_data.shape[1]),
                                                        self.unknown_nodes_num,
                                                        replace=False)
            self.train_known_nodes = np.setdiff1d(range(raw_data.shape[1]),
                                                  self.train_unknown_nodes)
        self.train_unknown_nodes = self.train_unknown_nodes.tolist()
        self.train_known_nodes = self.train_known_nodes.tolist()

        self.new_add_nodes = np.random.choice(
            self.train_unknown_nodes, num_add_nodes,
            replace=False).tolist()  #从原有的known里面挑一部分处理让他先是unknown然后在test中added
        self.failed_nodes = np.random.choice(
            self.train_known_nodes, num_failed_nodes, replace=False).tolist(
            )  #从原有的unknown里面挑一部分处理让他先是known然后在test中failed

        # use the following code to reproduce the results
        dataset_dir = Path(self.data_path).resolve().parent
        new_add_nodes_path = dataset_dir / "new_add_nodes.npy"
        failed_nodes_path = dataset_dir / "failed_nodes.npy"
        if new_add_nodes_path.exists() and failed_nodes_path.exists():
            self.new_add_nodes = np.load(new_add_nodes_path).tolist()
            self.failed_nodes = np.load(failed_nodes_path).tolist()

        self.test_known_nodes = list((set(self.train_known_nodes) -
                                      set(self.failed_nodes))
                                     | set(self.new_add_nodes))
        self.test_unknown_nodes = list(
            ((set(self.train_unknown_nodes) - set(self.new_add_nodes))
             | set(self.failed_nodes)))
        self.never_known_nodes = list(
            (set(self.train_unknown_nodes) - set(self.new_add_nodes)))
        self.always_known_nodes = list(
            set(self.train_known_nodes) - set(self.failed_nodes))
        self.train_known_nodes = list(
            set(self.train_known_nodes) | set(self.new_add_nodes))
        self.train_unknown_nodes = list(
            set(self.train_unknown_nodes) - set(self.new_add_nodes))
        train_data = raw_data[:int(0.7 *
                                   len(raw_data))][:, self.train_known_nodes]
        if self.flag == 'train':
            self.data = train_data
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 *
                                                     len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.train_known_nodes]
            self.adj = self.adj[self.train_known_nodes][:,
                                                        self.train_known_nodes]
        elif self.flag == 'val':
            self.data = raw_data[:int(0.7 * len(self.time_mark)
                                      )][:, self.train_known_nodes]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 *
                                                     len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.train_known_nodes]
            self.adj = self.adj[self.train_known_nodes][:,
                                                        self.train_known_nodes]
        else:
            self.data = raw_data[int(0.7 * len(self.time_mark)):]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[int(0.7 *
                                                    len(self.time_mark)):]
            #历史原因，我们从self.input_len开始测试集合，这里其实可以去掉
            self.data = self.data[self.input_len:]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[self.input_len:]
        if self.scale:
            self.mean_ = np.nanmean(train_data, axis=(0, 1), keepdims=True)
            self.std_ = np.nanstd(train_data, axis=(0, 1), keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        self.data = (self.data - self.mean_) / (self.std_)
        self.data_y = self.data.copy()
        if self.flag == 'test':
            self.data[:, self.test_unknown_nodes] = np.nan
        self.data = self.data[..., self.in_feas]
        self.data_y = self.data_y[..., self.tar_feas]
        self.num_nodes = self.data.shape[1]

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.always_known_nodes, self.never_known_nodes, self.new_add_nodes, self.failed_nodes

    def __getitem__(self, index):
        index = index * self.step
        input_begin = index
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        seq_x = self.data[input_begin:input_end]
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len -
                self.pred_len) // self.step + 1  #type:ignore

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[...,
                                                              self.tar_feas]


class LifecycleTrainPEMS(StaticPEMS):

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path='',
        unknown_nodes_list=[],
        unknown_nodes_num=0,
        lifecycle_seed: int = 42,
        async_num_events: int = 4,
        lifecycle_event_sampling: str = "mixed",
        lifecycle_training_strategy: str = "v1",
        lifecycle_role_features: bool = False,
        lifecycle_history_age_feature: bool = False,
        lifecycle_aas_loss_weight: float = 1.0,
        lifecycle_nas_loss_weight: float = 1.0,
        lifecycle_vs_loss_weight: float = 1.0,
        lifecycle_fs_loss_weight: float = 1.0,
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError("LifecycleTrainPEMS only supports train/val/test splits")
        self.lifecycle_seed = lifecycle_seed
        self.async_num_events = async_num_events
        self.lifecycle_event_sampling = lifecycle_event_sampling
        self.lifecycle_training_strategy = lifecycle_training_strategy
        self.lifecycle_role_features = lifecycle_role_features
        self.lifecycle_history_age_feature = lifecycle_history_age_feature
        self.lifecycle_role_loss_weights = {
            ROLE_AAS: float(lifecycle_aas_loss_weight),
            ROLE_NAS: float(lifecycle_nas_loss_weight),
            ROLE_VS: float(lifecycle_vs_loss_weight),
            ROLE_FS: float(lifecycle_fs_loss_weight),
        }
        self.current_epoch = 0
        if self.async_num_events <= 0:
            raise ValueError("async_num_events should be positive for lifecycle-aware training")
        super().__init__(
            data_path=data_path,
            size=size,
            flag=flag,
            scale=scale,
            freq=freq,
            target_features=target_features,
            input_features=input_features,
            dloader_name=dloader_name,
            splits=splits,
            step=step,
            inverse=inverse,
            unknown_nodes_path=unknown_nodes_path,
            unknown_nodes_list=unknown_nodes_list,
            unknown_nodes_num=unknown_nodes_num,
        )
        total_nas, total_fs = _resolve_async_event_node_counts(self.full_num_nodes)
        profile = _resolve_lifecycle_training_profile(self.lifecycle_training_strategy)
        self.lifecycle_profile = profile
        base_nas_count = max(1, total_nas // self.async_num_events)
        base_fs_count = max(1, total_fs // self.async_num_events)
        self.local_nas_count = max(1, int(np.ceil(base_nas_count * profile["nas_scale"])))
        self.local_fs_count = max(1, int(np.ceil(base_fs_count * profile["fs_scale"])))
        if profile["vs_ratio"] <= 0:
            self.local_vs_count = 0
        else:
            self.local_vs_count = max(1, int(round(profile["vs_ratio"] * self.num_nodes)))

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _build_rng(self, index):
        if self.flag in {"val", "test"}:
            seed = self.lifecycle_seed + index
        else:
            seed = self.lifecycle_seed + self.current_epoch * max(1, len(self)) + index
        return np.random.default_rng(seed)

    def _sample_event_time(self, rng):
        event_time_mode = self.lifecycle_profile["event_time_mode"]
        if event_time_mode == "uniform":
            return int(rng.integers(1, self.input_len))
        if event_time_mode == "early":
            upper_bound = max(2, self.input_len // 3 + 1)
            return int(rng.integers(1, upper_bound))
        raise ValueError(f"Unsupported lifecycle event_time_mode: {event_time_mode}")

    def _sample_local_role_groups(self, rng):
        sampling_mode = self.lifecycle_event_sampling
        if sampling_mode == "mixed":
            sampling_mode = "clustered" if rng.random() < LIFECYCLE_MIXED_CLUSTER_PROB else "random"
        eligible_nodes = set(range(self.num_nodes))
        nas_nodes = _sample_event_nodes(eligible_nodes, self.local_nas_count, self.adj, rng, sampling_mode)
        eligible_nodes -= set(nas_nodes)
        fs_nodes = _sample_event_nodes(eligible_nodes, self.local_fs_count, self.adj, rng, sampling_mode)
        eligible_nodes -= set(fs_nodes)
        vs_nodes = _sample_event_nodes(eligible_nodes, self.local_vs_count, self.adj, rng, sampling_mode)
        aas_nodes = sorted(eligible_nodes - set(vs_nodes))
        return aas_nodes, nas_nodes, vs_nodes, fs_nodes

    def _build_local_role_state(self, event_time, nas_nodes, vs_nodes, fs_nodes):
        role_ids = np.full((self.input_len, self.num_nodes), ROLE_AAS, dtype=np.int64)
        visibility_mask = np.ones((self.input_len, self.num_nodes), dtype=bool)

        if vs_nodes:
            role_ids[:, vs_nodes] = ROLE_VS
            visibility_mask[:, vs_nodes] = False
        if nas_nodes:
            role_ids[:event_time, nas_nodes] = ROLE_VS
            role_ids[event_time:, nas_nodes] = ROLE_NAS
            visibility_mask[:event_time, nas_nodes] = False
            visibility_mask[event_time:, nas_nodes] = True
        if fs_nodes:
            role_ids[:event_time, fs_nodes] = ROLE_AAS
            role_ids[event_time:, fs_nodes] = ROLE_FS
            visibility_mask[:event_time, fs_nodes] = True
            visibility_mask[event_time:, fs_nodes] = False
        return role_ids, visibility_mask

    def __getitem__(self, index):
        sample_index = index
        index = index * self.step
        input_begin = index
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        seq_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end]
        rng = self._build_rng(sample_index)
        event_time = self._sample_event_time(rng)
        _, nas_nodes, vs_nodes, fs_nodes = self._sample_local_role_groups(rng)
        role_ids, visibility_mask = self._build_local_role_state(event_time, nas_nodes, vs_nodes, fs_nodes)
        seq_x = _compose_lifecycle_input_features(
            seq_x,
            visibility_mask,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )
        target_role_weights = _build_target_role_weights(
            seq_y,
            nas_nodes,
            vs_nodes,
            fs_nodes,
            self.lifecycle_role_loss_weights,
        )
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark, target_role_weights


def _lcbt_window_regimes(timestamps, segment_start, segment_end, input_len, pred_len):
    segment_length = int(segment_end) - int(segment_start)
    num_samples = (segment_length - int(input_len) - int(pred_len)) + 1
    if num_samples <= 0:
        raise ValueError("LCBT segment is too short for the requested input/prediction lengths")
    if timestamps is None:
        return ["OffPeak"] * num_samples
    return [
        calendar_regime(timestamps[int(segment_start) + index + int(input_len) - 1])
        for index in range(num_samples)
    ]


def _lcbt_schedule_summary(schedule, num_nodes, input_len):
    scenario_histogram = {name: 0 for name in ("join", "fault", "steady")}
    phase_histogram = {phase: 0 for phase in LCBT_PHASES}
    transition_histogram = {
        "join": {str(index): 0 for index in range(int(input_len) + 1)},
        "fault": {str(index): 0 for index in range(int(input_len) + 1)},
    }
    topology_histogram = {"clustered": 0, "scattered": 0, "none": 0}
    cohort_histogram = {}
    calendar_crosstab = {}
    calendar_phase_crosstab = {}
    exposure = np.zeros(int(num_nodes), dtype=np.int64)
    for item in schedule:
        scenario = item["scenario"]
        scenario_histogram[scenario] += 1
        topology_histogram[item["topology_mode"]] += 1
        regime = item["calendar_regime"]
        calendar_crosstab.setdefault(regime, {name: 0 for name in ("join", "fault", "steady")})
        calendar_phase_crosstab.setdefault(
            regime,
            {phase: 0 for phase in LCBT_PHASES},
        )
        calendar_crosstab[regime][scenario] += 1
        if scenario != "steady":
            transition_histogram[scenario][str(item["transition_steps"])] += 1
            phase_name = LCBT_PHASES[item["phase_id"]]
            phase_histogram[phase_name] += 1
            calendar_phase_crosstab[regime][phase_name] += 1
            fraction_key = f"{float(item['cohort_fraction']):.6f}"
            cohort_histogram[fraction_key] = cohort_histogram.get(fraction_key, 0) + 1
            exposure[np.asarray(item["changed_nodes"], dtype=np.int64)] += 1
    return {
        "num_samples": len(schedule),
        "scenario_histogram": scenario_histogram,
        "phase_histogram": phase_histogram,
        "transition_histogram": transition_histogram,
        "topology_histogram": topology_histogram,
        "cohort_fraction_histogram": cohort_histogram,
        "calendar_crosstab": calendar_crosstab,
        "calendar_phase_crosstab": calendar_phase_crosstab,
        "node_exposure": {
            "minimum": int(exposure.min()) if len(exposure) else 0,
            "mean": float(exposure.mean()) if len(exposure) else 0.0,
            "maximum": int(exposure.max()) if len(exposure) else 0,
            "counts": exposure.astype(int).tolist(),
        },
    }


class LifecycleCoveragePEMS(Dataset):
    """LCBT v1 train/validation dataset with strict chronological splits."""

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path="",
        unknown_nodes_list=None,
        unknown_nodes_num=0,
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        lifecycle_observation_mask_feature=True,
        lifecycle_aas_loss_weight=1.0,
        lifecycle_nas_loss_weight=1.0,
        lifecycle_vs_loss_weight=1.0,
        lifecycle_fs_loss_weight=1.0,
        lcbt_manifest_root="./protocol_manifests",
        lcbt_protocol_seed=2026,
        lcbt_validation_seed=12026,
        lcbt_train_end_ratio=0.50,
        lcbt_val_end_ratio=0.60,
        lcbt_test_start_ratio=0.60,
        lcbt_scaler_end_ratio=0.50,
        lcbt_join_ratio=0.40,
        lcbt_fault_ratio=0.40,
        lcbt_steady_ratio=0.20,
        lcbt_cohort_fractions=(0.025, 0.05, 0.10),
        lcbt_cohort_weights=(0.25, 0.50, 0.25),
        lcbt_clustered_ratio=0.50,
        lcbt_overwrite_manifest=False,
    ):
        if flag not in {"train", "val"}:
            raise ValueError("LifecycleCoveragePEMS only supports train/val splits")
        self.flag = flag
        self.data_path = data_path
        self.input_len, self.look_back, self.pred_len = [int(value) for value in size]
        if self.look_back > self.input_len:
            raise ValueError("look_back should be smaller than input_len")
        self.scale = bool(scale)
        self.inverse = inverse
        self.freq = freq
        self.dloader_name = dloader_name
        self.step = 1
        self.unknown_nodes_path = unknown_nodes_path
        self.unknown_nodes = list(unknown_nodes_list or [])
        self.unknown_nodes_num = int(unknown_nodes_num)
        self.lifecycle_role_features = bool(lifecycle_role_features)
        self.lifecycle_history_age_feature = bool(lifecycle_history_age_feature)
        self.lifecycle_observation_mask_feature = bool(lifecycle_observation_mask_feature)
        self.input_features = input_features
        self.target_features = target_features
        self.role_loss_weights = {
            ROLE_AAS: float(lifecycle_aas_loss_weight),
            ROLE_NAS: float(lifecycle_nas_loss_weight),
            ROLE_VS: float(lifecycle_vs_loss_weight),
            ROLE_FS: float(lifecycle_fs_loss_weight),
        }
        self.manifest_root = Path(lcbt_manifest_root)
        self.protocol_seed = int(lcbt_protocol_seed)
        self.validation_seed = int(lcbt_validation_seed)
        self.overwrite_manifest = bool(lcbt_overwrite_manifest)
        self.split_ratios = {
            "train_end_ratio": float(lcbt_train_end_ratio),
            "val_end_ratio": float(lcbt_val_end_ratio),
            "test_start_ratio": float(lcbt_test_start_ratio),
            "scaler_end_ratio": float(lcbt_scaler_end_ratio),
        }
        self.scenario_ratios = (
            float(lcbt_join_ratio),
            float(lcbt_fault_ratio),
            float(lcbt_steady_ratio),
        )
        self.cohort_fractions = tuple(float(value) for value in lcbt_cohort_fractions)
        self.cohort_weights = tuple(float(value) for value in lcbt_cohort_weights)
        self.clustered_ratio = float(lcbt_clustered_ratio)
        self.current_epoch = 0
        self._read_data(input_features, target_features)
        self.protocol_manifest = self._ensure_protocol_manifest()
        self.training_protocol_hash = self.protocol_manifest["protocol_hash"]
        self._refresh_schedule()

    def _read_data(self, input_features, target_features):
        feature_map = {
            "q": [0],
            "o": [1],
            "v": [2],
            "qv": [0, 2],
            "qov": [0, 1, 2],
        }
        self.in_feas = feature_map[input_features]
        self.tar_feas = feature_map[target_features]
        payload = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))
        raw_source = np.asarray(payload["data"])
        self.raw_data_shape = [int(value) for value in raw_source.shape]
        self.raw_data_hash = array_hash(raw_source)
        raw_values = raw_source.astype(np.float32, copy=True)
        raw_values[raw_values <= 0] = np.nan
        self.full_timestamps = payload.get("time_stamp")
        self.timestamp_hash = (
            array_hash(np.asarray(self.full_timestamps))
            if self.full_timestamps is not None
            else None
        )
        self.full_num_nodes = int(raw_values.shape[1])
        self.full_adjacency = np.asarray(payload["adj"])
        self.adjacency_hash = array_hash(self.full_adjacency)

        if self.unknown_nodes_path:
            self.unknown_nodes = np.load(self.unknown_nodes_path).astype(int).tolist()
        elif self.unknown_nodes:
            self.unknown_nodes = [int(node) for node in self.unknown_nodes]
        else:
            rng = np.random.default_rng(0)
            self.unknown_nodes = sorted(
                rng.choice(self.full_num_nodes, size=self.unknown_nodes_num, replace=False).astype(int).tolist()
            )
        self.known_nodes = np.setdiff1d(
            np.arange(self.full_num_nodes, dtype=np.int64),
            np.asarray(self.unknown_nodes, dtype=np.int64),
        ).astype(int).tolist()
        self.boundaries = resolve_lcbt_boundaries(
            len(raw_values),
            train_end_ratio=self.split_ratios["train_end_ratio"],
            val_end_ratio=self.split_ratios["val_end_ratio"],
            test_start_ratio=self.split_ratios["test_start_ratio"],
            scaler_end_ratio=self.split_ratios["scaler_end_ratio"],
        )

        scaler_data = raw_values[
            self.boundaries["scaler_start"]:self.boundaries["scaler_end"],
            self.known_nodes,
        ]
        self.mean_ = np.nanmean(scaler_data, axis=(0, 1), keepdims=True)
        self.std_ = np.nanstd(scaler_data, axis=(0, 1), keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        if not self.scale:
            self.mean_ = np.zeros_like(self.mean_)
            self.std_ = np.ones_like(self.std_)

        if self.flag == "train":
            self.segment_start = self.boundaries["train_start"]
            self.segment_end = self.boundaries["train_end"]
        else:
            self.segment_start = self.boundaries["val_start"]
            self.segment_end = self.boundaries["val_end"]
        segment = raw_values[self.segment_start:self.segment_end, self.known_nodes]
        segment = (segment - self.mean_) / self.std_
        self.data = segment[..., self.in_feas]
        self.data_y = segment[..., self.tar_feas]
        self.num_nodes = int(self.data.shape[1])

        raw_time_mark = payload.get("time_stamp")
        self.time_mark = None
        if raw_time_mark is not None:
            transformed = time_features(raw_time_mark, freq=self.freq)
            self.time_mark = transformed[self.segment_start:self.segment_end]
        self.pos_mark = payload.get("pos_stamp")
        if self.pos_mark is not None:
            self.pos_mark = np.asarray(self.pos_mark)[self.known_nodes]
            denominator = np.max(self.pos_mark, keepdims=True) - np.min(self.pos_mark, keepdims=True)
            denominator[denominator == 0] = 1.0
            self.pos_mark = (self.pos_mark - np.min(self.pos_mark, keepdims=True)) / denominator - 0.5
        self.adj = self.full_adjacency[np.ix_(self.known_nodes, self.known_nodes)]
        self.calendar_regimes = _lcbt_window_regimes(
            self.full_timestamps,
            self.segment_start,
            self.segment_end,
            self.input_len,
            self.pred_len,
        )

    def _build_schedule(self, flag, seed, epoch):
        if flag == "train":
            start, end = self.boundaries["train_start"], self.boundaries["train_end"]
        else:
            start, end = self.boundaries["val_start"], self.boundaries["val_end"]
        regimes = _lcbt_window_regimes(
            self.full_timestamps,
            start,
            end,
            self.input_len,
            self.pred_len,
        )
        return build_lcbt_schedule(
            len(regimes),
            self.input_len,
            self.adj,
            regimes,
            seed=seed,
            epoch=epoch,
            scenario_ratios=self.scenario_ratios,
            cohort_fractions=self.cohort_fractions,
            cohort_weights=self.cohort_weights,
            clustered_ratio=self.clustered_ratio,
        )

    def _manifest_payload(self):
        train_schedule = self._build_schedule("train", self.protocol_seed, 0)
        val_schedule = self._build_schedule("val", self.validation_seed, 0)
        payload = {
            "schema_version": 1,
            "protocol_name": LCBT_PROTOCOL_NAME,
            "protocol_version": LCBT_PROTOCOL_VERSION,
            "dataset": self.dloader_name,
            "data_path": str(Path(self.data_path).resolve()),
            "data_shape": self.raw_data_shape,
            "data_hash": self.raw_data_hash,
            "timestamp_hash": self.timestamp_hash,
            "adjacency_hash": self.adjacency_hash,
            "known_nodes_hash": stable_hash(self.known_nodes),
            "unknown_nodes_hash": stable_hash(sorted(self.unknown_nodes)),
            "boundaries": self.boundaries,
            "scaler": {
                "fit_start": self.boundaries["scaler_start"],
                "fit_end": self.boundaries["scaler_end"],
                "mean_hash": array_hash(self.mean_),
                "std_hash": array_hash(self.std_),
            },
            "input_semantics": {
                "input_features": self.input_features,
                "target_features": self.target_features,
                "scale": self.scale,
                "effective_mask": "natural_finite_and_lifecycle_visible",
                "role_features": self.lifecycle_role_features,
                "history_age_feature": self.lifecycle_history_age_feature,
                "observation_mask_feature": self.lifecycle_observation_mask_feature,
                "phase_labels_are_model_inputs": False,
            },
            "role_loss_weights": {
                "steady_observed": self.role_loss_weights[ROLE_AAS],
                "post_join": self.role_loss_weights[ROLE_NAS],
                "pre_join": self.role_loss_weights[ROLE_VS],
                "post_fault": self.role_loss_weights[ROLE_FS],
            },
            "input_len": self.input_len,
            "pred_len": self.pred_len,
            "window_step": 1,
            "seeds": {
                "training": self.protocol_seed,
                "validation": self.validation_seed,
            },
            "scenario_ratios": {
                "join": self.scenario_ratios[0],
                "fault": self.scenario_ratios[1],
                "steady": self.scenario_ratios[2],
            },
            "cohort_fractions": list(self.cohort_fractions),
            "cohort_weights": list(self.cohort_weights),
            "clustered_ratio": self.clustered_ratio,
            "reference_training_schedule": _lcbt_schedule_summary(
                train_schedule,
                self.num_nodes,
                self.input_len,
            ),
            "validation_schedule": _lcbt_schedule_summary(
                val_schedule,
                self.num_nodes,
                self.input_len,
            ),
        }
        payload["protocol_hash"] = lcbt_manifest_hash(payload)
        return payload

    def _ensure_protocol_manifest(self):
        manifest = self._manifest_payload()
        manifest_path = self.manifest_root / self.dloader_name / "lcbt_v1" / "protocol_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("protocol_hash") != manifest["protocol_hash"] and not self.overwrite_manifest:
                raise ValueError(
                    f"LCBT manifest mismatch at {manifest_path}. "
                    "Use lcbt_overwrite_manifest=true only for an intentional protocol replacement."
                )
            if existing.get("protocol_hash") == manifest["protocol_hash"]:
                return existing
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return manifest

    def _refresh_schedule(self):
        if self.flag == "train":
            self.schedule = build_lcbt_schedule(
                len(self),
                self.input_len,
                self.adj,
                self.calendar_regimes,
                seed=self.protocol_seed,
                epoch=self.current_epoch,
                scenario_ratios=self.scenario_ratios,
                cohort_fractions=self.cohort_fractions,
                cohort_weights=self.cohort_weights,
                clustered_ratio=self.clustered_ratio,
            )
        else:
            self.schedule = build_lcbt_schedule(
                len(self),
                self.input_len,
                self.adj,
                self.calendar_regimes,
                seed=self.validation_seed,
                epoch=0,
                scenario_ratios=self.scenario_ratios,
                cohort_fractions=self.cohort_fractions,
                cohort_weights=self.cohort_weights,
                clustered_ratio=self.clustered_ratio,
            )

    def set_epoch(self, epoch):
        if self.flag == "train":
            self.current_epoch = int(epoch)
            self._refresh_schedule()

    def _lifecycle_state(self, item):
        role_ids = np.full((self.input_len, self.num_nodes), ROLE_AAS, dtype=np.int64)
        lifecycle_mask = np.ones((self.input_len, self.num_nodes), dtype=bool)
        changed_nodes = item["changed_nodes"]
        transition_steps = int(item["transition_steps"])
        cutoff = self.input_len - transition_steps
        if item["scenario"] == "join" and changed_nodes:
            role_ids[:cutoff, changed_nodes] = ROLE_VS
            role_ids[cutoff:, changed_nodes] = ROLE_NAS
            lifecycle_mask[:cutoff, changed_nodes] = False
        elif item["scenario"] == "fault" and changed_nodes:
            role_ids[cutoff:, changed_nodes] = ROLE_FS
            lifecycle_mask[cutoff:, changed_nodes] = False
        return role_ids, lifecycle_mask

    def __getitem__(self, index):
        input_begin = int(index)
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        raw_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end]
        item = self.schedule[int(index)]
        role_ids, lifecycle_mask = self._lifecycle_state(item)
        seq_x, _ = compose_effective_lifecycle_features(
            raw_x,
            lifecycle_mask,
            role_ids,
            append_role=self.lifecycle_role_features,
            append_age=self.lifecycle_history_age_feature,
            append_observation_mask=self.lifecycle_observation_mask_feature,
        )
        target_weights = np.ones_like(seq_y, dtype=np.float32)
        if item["scenario"] == "join" and item["changed_nodes"]:
            target_weights[:, item["changed_nodes"], :] = self.role_loss_weights[ROLE_NAS]
        elif item["scenario"] == "fault" and item["changed_nodes"]:
            target_weights[:, item["changed_nodes"], :] = self.role_loss_weights[ROLE_FS]
        changed_mask = np.zeros(self.num_nodes, dtype=np.float32)
        changed_mask[item["changed_nodes"]] = 1.0
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return (
            seq_x,
            seq_y,
            x_t_mark,
            y_t_mark,
            target_weights,
            np.int64(item["phase_id"]),
            changed_mask,
        )

    def __len__(self):
        return (len(self.data) - self.input_len - self.pred_len) + 1

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return list(range(self.num_nodes)), [], [], []

    def get_protocol_manifest(self):
        return self.protocol_manifest


class AsyncLifecyclePEMS(Dataset):
    """Dataset-adaptive paired lifecycle event study test provider."""

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path="",
        unknown_nodes_list=[],
        unknown_nodes_num=0,
        lifecycle_role_features: bool = False,
        lifecycle_history_age_feature: bool = False,
        lifecycle_observation_mask_feature: bool = False,
        lcbt_manifest_root: str = "./protocol_manifests",
        lcbt_scaler_end_ratio=None,
        lcbt_test_start_ratio=None,
        ples_pre_steps: int = 12,
        ples_post_steps: int = 36,
        ples_temporal_blocks: int = 4,
        ples_cohort_count: int = 5,
        ples_cohort_fraction: float = 0.05,
        ples_min_event_gap: int = 72,
        ples_seed: int = 2026,
        ples_bootstrap_samples: int = 2000,
        ples_bootstrap_seed: int = 2026,
        **legacy_async_options,
    ):
        if flag != "test":
            raise ValueError("AsyncLifecyclePEMS only supports the test split")
        fea_dict = {
            "q": [0],
            "o": [1],
            "v": [2],
            "qv": [0, 2],
            "qov": [0, 1, 2],
        }
        self.tar_feas = fea_dict[target_features]
        self.in_feas = fea_dict[input_features]
        self.target_features = target_features
        self.input_features = input_features
        self.raw_enc_in = len(self.in_feas)
        self.flag = flag
        self.input_len = int(size[0])
        self.look_back = int(size[1])
        if self.look_back > self.input_len:
            raise ValueError("look_back should be smaller than input_len")
        self.pred_len = int(size[2])
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.data_path = data_path
        self.train_unknown_nodes = unknown_nodes_list
        self.unknown_nodes_num = unknown_nodes_num
        self.unknown_nodes_path = unknown_nodes_path
        self.dloader_name = dloader_name
        self.step = int(step)
        self.train_ratio, self.valid_ratio = splits
        self.lifecycle_role_features = lifecycle_role_features
        self.lifecycle_history_age_feature = lifecycle_history_age_feature
        self.lifecycle_observation_mask_feature = lifecycle_observation_mask_feature
        self.lcbt_manifest_root = lcbt_manifest_root
        self.lcbt_scaler_end_ratio = lcbt_scaler_end_ratio
        self.lcbt_test_start_ratio = lcbt_test_start_ratio
        self.training_protocol_hash = None
        self.ples_options = {
            "pre_steps": int(ples_pre_steps),
            "post_steps": int(ples_post_steps),
            "temporal_blocks": int(ples_temporal_blocks),
            "requested_cohorts": int(ples_cohort_count),
            "cohort_fraction": float(ples_cohort_fraction),
            "min_event_gap": int(ples_min_event_gap),
            "seed": int(ples_seed),
            "bootstrap_samples": int(ples_bootstrap_samples),
            "bootstrap_seed": int(ples_bootstrap_seed),
        }
        self.legacy_async_options = legacy_async_options
        self.__read_data__()

    def __read_data__(self):
        raw_data = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))
        self.time_stamp_raw = raw_data["time_stamp"]
        if self.time_stamp_raw is None:
            raise ValueError("PLES requires dataset timestamps for calendar-stratified origins")
        full_time_mark = time_features(self.time_stamp_raw, freq=self.freq)

        self.pos_mark = raw_data["pos_stamp"]
        if self.pos_mark is not None:
            denominator = np.max(self.pos_mark, keepdims=True) - np.min(self.pos_mark, keepdims=True)
            denominator = np.where(denominator == 0, 1.0, denominator)
            self.pos_mark = (self.pos_mark - np.min(self.pos_mark, keepdims=True)) / denominator - 0.5

        self.adj = raw_data["adj"]
        raw_source = np.asarray(raw_data["data"])
        raw_series_hash = array_hash(raw_source)
        raw_series = raw_source.astype(np.float32)
        self.full_num_nodes = raw_series.shape[1]
        self.num_nodes = raw_series.shape[1]
        raw_series[raw_series <= 0] = np.nan

        if len(self.unknown_nodes_path):
            self.train_unknown_nodes = np.load(self.unknown_nodes_path)
        elif len(self.train_unknown_nodes):
            self.train_unknown_nodes = np.asarray(self.train_unknown_nodes)
        else:
            rng = np.random.default_rng(0)
            self.train_unknown_nodes = rng.choice(
                range(raw_series.shape[1]),
                self.unknown_nodes_num,
                replace=False,
            )
        self.train_known_nodes = np.setdiff1d(range(raw_series.shape[1]), self.train_unknown_nodes)
        self.train_unknown_nodes = sorted(np.asarray(self.train_unknown_nodes, dtype=int).tolist())
        self.train_known_nodes = sorted(np.asarray(self.train_known_nodes, dtype=int).tolist())
        self.initial_aas_nodes = self.train_known_nodes.copy()
        self.initial_vs_nodes = self.train_unknown_nodes.copy()

        if self.lcbt_scaler_end_ratio is None or self.lcbt_test_start_ratio is None:
            scaler_cutoff = int(float(self.train_ratio) * len(raw_series))
            test_cutoff = scaler_cutoff
        else:
            scaler_cutoff = int(float(self.lcbt_scaler_end_ratio) * len(raw_series))
            test_cutoff = int(float(self.lcbt_test_start_ratio) * len(raw_series))
            manifest_path = (
                Path(self.lcbt_manifest_root)
                / self.dloader_name
                / "lcbt_v1"
                / "protocol_manifest.json"
            )
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"LCBT training manifest not found: {manifest_path}. "
                    "Build the train/validation loaders before running test_async."
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                training_manifest = json.load(handle)
            expected = training_manifest.get("boundaries", {})
            if expected.get("scaler_end") != scaler_cutoff or expected.get("test_start") != test_cutoff:
                raise ValueError("LCBT train manifest split boundaries do not match async-test settings")
            if training_manifest.get("data_hash") != raw_series_hash:
                raise ValueError("LCBT train manifest data fingerprint does not match async-test data")
            if training_manifest.get("adjacency_hash") != array_hash(self.adj):
                raise ValueError("LCBT train manifest adjacency fingerprint does not match async-test graph")
            self.training_protocol_hash = training_manifest.get("protocol_hash")

        train_data = raw_series[:scaler_cutoff][:, self.train_known_nodes]
        test_data = raw_series[test_cutoff:]
        self.time_mark = full_time_mark[test_cutoff:]
        self.time_stamp_raw = np.asarray(self.time_stamp_raw)[test_cutoff:]

        if self.scale:
            self.mean_ = np.nanmean(train_data, axis=(0, 1), keepdims=True)
            self.std_ = np.nanstd(train_data, axis=(0, 1), keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        else:
            self.mean_ = np.zeros((1, 1, train_data.shape[-1]), dtype=np.float32)
            self.std_ = np.ones((1, 1, train_data.shape[-1]), dtype=np.float32)

        scaled_test = (test_data - self.mean_) / self.std_
        self.data = scaled_test[..., self.in_feas]
        self.data_y = scaled_test[..., self.tar_feas]
        self.protocol_manifest = build_protocol_manifest(
            dataset_name=self.dloader_name,
            timestamps=self.time_stamp_raw,
            adjacency=self.adj,
            known_nodes=self.train_known_nodes,
            unknown_nodes=self.train_unknown_nodes,
            test_values=test_data[..., self.tar_feas],
            input_features=self.input_features,
            target_features=self.target_features,
            input_len=self.input_len,
            pred_len=self.pred_len,
            **self.ples_options,
        )

    def _base_visibility_and_roles(self):
        visibility = np.zeros((self.input_len, self.num_nodes), dtype=bool)
        visibility[:, self.initial_aas_nodes] = True
        role_ids = np.full((self.input_len, self.num_nodes), ROLE_AAS, dtype=np.int64)
        role_ids[:, self.initial_vs_nodes] = ROLE_VS
        return visibility, role_ids

    def _time_marks(self, input_begin, input_end, output_begin, output_end):
        if self.time_mark is None:
            return (
                np.zeros((input_end - input_begin, 0), dtype=np.float32),
                np.zeros((output_end - output_begin, 0), dtype=np.float32),
            )
        return self.time_mark[input_begin:input_end], self.time_mark[output_begin:output_end]

    def _compose_input(self, raw_x, visibility, role_ids):
        if getattr(self, "lifecycle_observation_mask_feature", False):
            seq_x, _ = compose_effective_lifecycle_features(
                raw_x,
                visibility,
                role_ids,
                append_role=self.lifecycle_role_features,
                append_age=self.lifecycle_history_age_feature,
                append_observation_mask=True,
            )
            return seq_x
        return _compose_lifecycle_input_features(
            raw_x,
            visibility,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )

    def build_ples_sample(self, origin, cohort, track, arm, tau):
        if track not in {"join", "fault"}:
            raise ValueError(f"Unsupported PLES track: {track}")
        if arm not in {"control", "treatment"}:
            raise ValueError(f"Unsupported PLES arm: {arm}")

        event_time = int(origin["event_time_index"])
        input_end = event_time + int(tau)
        input_begin = input_end - self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        if input_begin < 0 or output_end > len(self.data):
            raise IndexError("PLES sample exceeds the dataset timeline")

        raw_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end].copy()
        visibility, role_ids = self._base_visibility_and_roles()
        if arm == "treatment":
            post_rows = np.flatnonzero(np.arange(input_begin, input_end) >= event_time)
            if track == "join" and len(post_rows):
                nodes = np.asarray(cohort["join_nodes"], dtype=np.int64)
                visibility[np.ix_(post_rows, nodes)] = True
                role_ids[np.ix_(post_rows, nodes)] = ROLE_NAS
            elif track == "fault" and len(post_rows):
                nodes = np.asarray(cohort["fault_nodes"], dtype=np.int64)
                visibility[np.ix_(post_rows, nodes)] = False
                role_ids[np.ix_(post_rows, nodes)] = ROLE_FS

        seq_x = self._compose_input(raw_x, visibility, role_ids)
        x_t_mark, y_t_mark = self._time_marks(input_begin, input_end, output_begin, output_end)
        return seq_x, seq_y, x_t_mark, y_t_mark

    def get_protocol_manifest(self):
        return self.protocol_manifest

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.initial_aas_nodes, self.initial_vs_nodes, [], []

    def __getitem__(self, index):
        input_begin = int(index) * self.step
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        raw_x = self.data[input_begin:input_end].copy()
        visibility, role_ids = self._base_visibility_and_roles()
        seq_x = self._compose_input(raw_x, visibility, role_ids)
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = self._time_marks(input_begin, input_end, output_begin, output_end)
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len - self.pred_len) // self.step + 1

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]
