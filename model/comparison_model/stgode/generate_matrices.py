import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from lib.v4_payload import load_raw_payload
import torch
from fastdtw import fastdtw
from tqdm import tqdm


def get_normalized_adj(A):
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 1e-5] = 1e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def _load_raw_payload(data_path):
    raw = load_raw_payload(data_path)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "files"):
        return {key: raw[key] for key in raw.files}
    raise ValueError(f"Unsupported raw data payload type: {type(raw)}")


def generate_dtw_spa_matrix(data_path, output_dir, speed_feature_index=2, train_ratio=0.7, sigma1=0.1, thres1=0.3):
    raw = _load_raw_payload(data_path)
    data = np.asarray(raw["data"], dtype=np.float32)
    adj = np.asarray(raw["adj"], dtype=np.float32)
    num_node = data.shape[1]

    train_length = int(len(data) * train_ratio)
    speed_data = data[:train_length, :, speed_feature_index]
    steps_per_day = 24 * 12
    usable = (speed_data.shape[0] // steps_per_day) * steps_per_day
    speed_data = speed_data[:usable]
    if usable == 0:
        raise ValueError("Not enough training samples to build STGODE semantic matrix")

    daily_mean = np.mean(
        speed_data.reshape(speed_data.shape[0] // steps_per_day, steps_per_day, num_node),
        axis=0,
    ).T
    dtw_distance = np.zeros((num_node, num_node), dtype=np.float32)
    for i in tqdm(range(num_node), desc="STGODE semantic matrix"):
        for j in range(i, num_node):
            dtw_distance[i, j] = fastdtw(daily_mean[i], daily_mean[j], radius=6)[0]
    dtw_distance = dtw_distance + dtw_distance.T - np.diag(np.diag(dtw_distance))

    mean = np.mean(dtw_distance)
    std = np.std(dtw_distance)
    std = 1.0 if std == 0 else std
    dtw_score = np.exp(-(((dtw_distance - mean) / std) ** 2) / (sigma1**2))
    dtw_matrix = np.zeros_like(dtw_score, dtype=np.float32)
    dtw_matrix[dtw_score > thres1] = 1.0

    sp_matrix = np.where(adj > 0, adj, 0.0).astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtw_matrix = get_normalized_adj(dtw_matrix).numpy()
    sp_matrix = get_normalized_adj(sp_matrix).numpy()
    with open(output_dir / "dtw_matrix_normalized.pkl", "wb") as handle:
        pickle.dump(dtw_matrix, handle)
    with open(output_dir / "sp_matrix_normalized.pkl", "wb") as handle:
        pickle.dump(sp_matrix, handle)
    print(f"Saved STGODE matrices to {output_dir.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate STGODE semantic/spatial matrices for project raw_data.pkl")
    parser.add_argument("--data_path", required=True, help="path to raw_data.pkl used by this project")
    parser.add_argument("--output_dir", required=True, help="directory to save dtw/sp matrices")
    parser.add_argument("--speed_feature_index", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--sigma1", type=float, default=0.1)
    parser.add_argument("--thres1", type=float, default=0.3)
    cli_args = parser.parse_args()
    generate_dtw_spa_matrix(
        data_path=cli_args.data_path,
        output_dir=cli_args.output_dir,
        speed_feature_index=cli_args.speed_feature_index,
        train_ratio=cli_args.train_ratio,
        sigma1=cli_args.sigma1,
        thres1=cli_args.thres1,
    )
