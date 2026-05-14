import numpy as np
from torch.utils.data import Dataset
import warnings
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
        "event_scale": 1.0,
        "vs_ratio": LIFECYCLE_LOCAL_VS_RATIO,
    },
    "round1": {
        "event_scale": 0.5,
        "vs_ratio": 0.10,
    },
    "round2": {
        "event_scale": 0.5,
        "vs_ratio": 0.0,
    },
}


def _resolve_async_event_node_counts(num_nodes):
    if num_nodes in ASYNC_EVENT_NODE_COUNTS:
        return ASYNC_EVENT_NODE_COUNTS[num_nodes]
    raise ValueError("num_nodes should be 207 or 325")


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
        raw_data = np.load(self.data_path, allow_pickle=True)

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
        raw_data = np.load(self.data_path, allow_pickle=True)

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
        if num_nodes == 207:  #metr-la
            num_failed_nodes = num_add_nodes = 20  #~5%
        elif num_nodes == 325:  #pemsbay
            num_failed_nodes = num_add_nodes = 32  #~5%
        else:
            raise ValueError('num_nodes should be 207 or 325')
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
        if 'METR' in self.dloader_name:
            self.new_add_nodes = np.load(
                'data/METR-LA/new_add_nodes.npy').tolist()
            self.failed_nodes = np.load(
                'data/METR-LA/failed_nodes.npy').tolist()
        else:
            self.new_add_nodes = np.load(
                'data/PEMS-BAY/new_add_nodes.npy').tolist()
            self.failed_nodes = np.load(
                'data/PEMS-BAY/failed_nodes.npy').tolist()

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
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError("LifecycleTrainPEMS only supports train/val/test splits")
        self.lifecycle_seed = lifecycle_seed
        self.async_num_events = async_num_events
        self.lifecycle_event_sampling = lifecycle_event_sampling
        self.lifecycle_training_strategy = lifecycle_training_strategy
        self.lifecycle_role_features = lifecycle_role_features
        self.lifecycle_history_age_feature = lifecycle_history_age_feature
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
        base_nas_count = max(1, total_nas // self.async_num_events)
        base_fs_count = max(1, total_fs // self.async_num_events)
        self.local_nas_count = max(1, int(np.ceil(base_nas_count * profile["event_scale"])))
        self.local_fs_count = max(1, int(np.ceil(base_fs_count * profile["event_scale"])))
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
        event_time = int(rng.integers(1, self.input_len))
        _, nas_nodes, vs_nodes, fs_nodes = self._sample_local_role_groups(rng)
        role_ids, visibility_mask = self._build_local_role_state(event_time, nas_nodes, vs_nodes, fs_nodes)
        seq_x = _compose_lifecycle_input_features(
            seq_x,
            visibility_mask,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark


def _unwrap_raw_data(raw_data):
    if isinstance(raw_data, np.ndarray) and raw_data.shape == ():
        return raw_data.item()
    return raw_data


class AsyncLifecyclePEMS(Dataset):

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
        async_num_events: int = 4,
        async_commissioning_k: int = 12,
        async_event_seed: int = 0,
        lifecycle_role_features: bool = False,
        lifecycle_history_age_feature: bool = False,
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
        self.flag = flag
        self.input_len = size[0]
        self.look_back = size[1]
        if self.look_back > self.input_len:
            raise ValueError("look_back should be smaller than input_len")
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
        self.async_num_events = async_num_events
        self.async_commissioning_k = async_commissioning_k
        self.async_event_seed = async_event_seed
        self.lifecycle_role_features = lifecycle_role_features
        self.lifecycle_history_age_feature = lifecycle_history_age_feature
        self.__read_data__()

    def _build_event_plan(self, total_time_steps):
        if self.async_num_events <= 0:
            raise ValueError("async_num_events should be positive")
        total_nas, total_fs = _resolve_async_event_node_counts(self.num_nodes)
        if self.async_num_events > min(total_nas, total_fs):
            raise ValueError("async_num_events should not exceed the total NAS/FS count")
        if len(self.initial_vs_nodes) < total_nas:
            raise ValueError("not enough VS nodes to allocate async NAS events")
        if len(self.initial_aas_nodes) < total_fs:
            raise ValueError("not enough AAS nodes to allocate async FS events")

        rng = np.random.default_rng(self.async_event_seed)
        nas_pool = rng.permutation(np.array(sorted(self.initial_vs_nodes)))[:total_nas].tolist()
        fs_pool = rng.permutation(np.array(sorted(self.initial_aas_nodes)))[:total_fs].tolist()
        nas_chunks = [sorted(chunk.tolist()) for chunk in np.array_split(np.array(nas_pool), self.async_num_events)]
        fs_chunks = [sorted(chunk.tolist()) for chunk in np.array_split(np.array(fs_pool), self.async_num_events)]

        self.num_windows = self.__len__()
        start_windows = np.floor((np.arange(1, self.async_num_events + 1) * self.num_windows) / (self.async_num_events + 1)).astype(int)
        if len(np.unique(start_windows)) != len(start_windows):
            raise ValueError("async event starts collide; reduce async_num_events")

        window_stage = ["warmup"] * self.num_windows
        window_event_index = [-1] * self.num_windows
        event_plan = []
        active_aas = set(self.initial_aas_nodes)
        remaining_vs = set(self.initial_vs_nodes)

        for event_ix, start_window in enumerate(start_windows.tolist()):
            next_start = start_windows[event_ix + 1] if event_ix + 1 < len(start_windows) else self.num_windows
            commissioning_end = start_window + self.async_commissioning_k - 1
            evaluation_start = start_window + self.async_commissioning_k
            evaluation_end = next_start - 1
            if commissioning_end >= self.num_windows:
                raise ValueError("commissioning windows exceed the available test windows")
            if evaluation_start > evaluation_end:
                raise ValueError("async lifecycle event has no evaluation windows under the current commissioning_k")

            nas_nodes = nas_chunks[event_ix]
            fs_nodes = fs_chunks[event_ix]
            stable_aas = sorted(active_aas - set(fs_nodes))
            current_vs = sorted(remaining_vs - set(nas_nodes))
            event_time = start_window * self.step + self.input_len
            if event_time > total_time_steps:
                raise ValueError("async event starts after the test timeline ends")

            for window_idx in range(start_window, commissioning_end + 1):
                window_stage[window_idx] = "commissioning"
                window_event_index[window_idx] = event_ix
            for window_idx in range(evaluation_start, evaluation_end + 1):
                window_stage[window_idx] = "evaluation"
                window_event_index[window_idx] = event_ix

            commissioning_times = [start_window * self.step, commissioning_end * self.step + self.input_len - 1]
            evaluation_times = [evaluation_start * self.step, evaluation_end * self.step + self.input_len - 1]
            event_plan.append(
                {
                    "event_id": event_ix + 1,
                    "event_index": event_ix,
                    "start_window": start_window,
                    "event_time_index": event_time,
                    "commissioning_window_range": [start_window, commissioning_end],
                    "evaluation_window_range": [evaluation_start, evaluation_end],
                    "commissioning_time_range": commissioning_times,
                    "evaluation_time_range": evaluation_times,
                    "nas_nodes": nas_nodes,
                    "fs_nodes": fs_nodes,
                    "aas_nodes": stable_aas,
                    "vs_nodes": current_vs,
                    "calibration_nodes": sorted(stable_aas + nas_nodes),
                }
            )

            remaining_vs -= set(nas_nodes)
            active_aas = set(stable_aas) | set(nas_nodes)

        self.window_stage = window_stage
        self.window_event_index = window_event_index
        self.event_plan = event_plan

    def _build_global_lifecycle_state(self):
        self.node_onboard_time = np.full(self.num_nodes, -1, dtype=np.int64)
        self.node_fail_time = np.full(self.num_nodes, -1, dtype=np.int64)
        for event in self.event_plan:
            event_time = event["event_time_index"]
            if event["nas_nodes"]:
                self.node_onboard_time[event["nas_nodes"]] = event_time
            if event["fs_nodes"]:
                self.node_fail_time[event["fs_nodes"]] = event_time

    def _build_role_ids_for_window(self, input_begin, input_end):
        time_indices = np.arange(input_begin, input_end, dtype=np.int64)[:, None]
        role_ids = np.full((input_end - input_begin, self.num_nodes), ROLE_AAS, dtype=np.int64)
        role_ids[:, self.initial_vs_nodes] = ROLE_VS

        onboarding_nodes = np.where(self.node_onboard_time >= 0)[0]
        if len(onboarding_nodes):
            onboard_times = self.node_onboard_time[onboarding_nodes][None, :]
            role_ids[:, onboarding_nodes] = np.where(time_indices >= onboard_times, ROLE_NAS, ROLE_VS)

        failed_nodes = np.where(self.node_fail_time >= 0)[0]
        if len(failed_nodes):
            fail_times = self.node_fail_time[failed_nodes][None, :]
            role_ids[:, failed_nodes] = np.where(time_indices >= fail_times, ROLE_FS, role_ids[:, failed_nodes])
        return role_ids

    def __read_data__(self):
        raw_data = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))

        self.time_stamp_raw = raw_data["time_stamp"]
        self.time_mark = self.time_stamp_raw
        if self.time_mark is not None:
            self.time_mark = time_features(self.time_mark, freq=self.freq)

        self.pos_mark = raw_data["pos_stamp"]
        if self.pos_mark is not None:
            self.pos_mark = (self.pos_mark - np.min(self.pos_mark, keepdims=True)) / (
                np.max(self.pos_mark, keepdims=True) - np.min(self.pos_mark, keepdims=True)
            ) - 0.5

        self.adj = raw_data["adj"]
        raw_series = raw_data["data"].astype(np.float32)
        self.full_num_nodes = raw_series.shape[1]
        self.num_nodes = raw_series.shape[1]
        raw_series[raw_series <= 0] = np.nan

        if len(self.unknown_nodes_path):
            self.train_unknown_nodes = np.load(self.unknown_nodes_path)
            self.train_known_nodes = np.setdiff1d(range(raw_series.shape[1]), self.train_unknown_nodes)
        elif len(self.train_unknown_nodes):
            self.train_known_nodes = np.setdiff1d(range(raw_series.shape[1]), self.train_unknown_nodes)
        else:
            np.random.seed(0)
            self.train_unknown_nodes = np.random.choice(range(raw_series.shape[1]), self.unknown_nodes_num, replace=False)
            self.train_known_nodes = np.setdiff1d(range(raw_series.shape[1]), self.train_unknown_nodes)

        self.train_unknown_nodes = sorted(self.train_unknown_nodes.tolist())
        self.train_known_nodes = sorted(self.train_known_nodes.tolist())
        self.initial_aas_nodes = self.train_known_nodes.copy()
        self.initial_vs_nodes = self.train_unknown_nodes.copy()

        train_cutoff = int(0.7 * len(raw_series))
        train_data = raw_series[:train_cutoff][:, self.train_known_nodes]
        test_data = raw_series[train_cutoff:]
        test_time_mark = self.time_mark[train_cutoff:] if self.time_mark is not None else None
        test_time_stamp_raw = self.time_stamp_raw[train_cutoff:] if self.time_stamp_raw is not None else None
        test_data = test_data[self.input_len :]
        if test_time_mark is not None:
            test_time_mark = test_time_mark[self.input_len :]
        if test_time_stamp_raw is not None:
            test_time_stamp_raw = test_time_stamp_raw[self.input_len :]

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
        self.time_mark = test_time_mark
        self.time_stamp_raw = test_time_stamp_raw

        self._build_event_plan(total_time_steps=len(self.data))
        self._build_global_lifecycle_state()
        self.availability_mask = np.zeros((len(self.data), self.num_nodes), dtype=bool)
        self.availability_mask[:, self.initial_aas_nodes] = True

        for event in self.event_plan:
            event_time = event["event_time_index"]
            if event["nas_nodes"]:
                self.availability_mask[event_time:, event["nas_nodes"]] = True
            if event["fs_nodes"]:
                self.availability_mask[event_time:, event["fs_nodes"]] = False

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.initial_aas_nodes, self.initial_vs_nodes, [], []

    def get_async_lifecycle_bundle(self):
        return {
            "event_plan": self.event_plan,
            "window_stage": self.window_stage,
            "window_event_index": self.window_event_index,
            "initial_aas_nodes": self.initial_aas_nodes,
            "initial_vs_nodes": self.initial_vs_nodes,
        }

    def __getitem__(self, index):
        index = index * self.step
        input_begin = index
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        seq_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end]
        visibility_mask = self.availability_mask[input_begin:input_end]
        role_ids = self._build_role_ids_for_window(input_begin, input_end)
        seq_x = _compose_lifecycle_input_features(
            seq_x,
            visibility_mask,
            role_ids,
            self.lifecycle_role_features,
            self.lifecycle_history_age_feature,
        )
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len - self.pred_len) // self.step + 1  # type: ignore

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]
