from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import warnings
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
        raw_series = raw_data["data"].astype(np.float32)
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

        train_cutoff = int(float(self.train_ratio) * len(raw_series))
        train_data = raw_series[:train_cutoff][:, self.train_known_nodes]
        test_data = raw_series[train_cutoff:]
        self.time_mark = full_time_mark[train_cutoff:]
        self.time_stamp_raw = np.asarray(self.time_stamp_raw)[train_cutoff:]

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

        seq_x = _compose_lifecycle_input_features(
            raw_x,
            visibility,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )
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
        seq_x = _compose_lifecycle_input_features(
            raw_x,
            visibility,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = self._time_marks(input_begin, input_end, output_begin, output_end)
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len - self.pred_len) // self.step + 1

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]
