import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PeMSD4/PeMSD8 npz+csv files to this project's raw_data payload.")
    parser.add_argument("--npz_path", required=True, help="Path to PEMS04.npz/PEMS08.npz.")
    parser.add_argument("--csv_path", required=True, help="Path to the distance/edge csv file.")
    parser.add_argument("--out_path", required=True, help="Output raw_data.pkl path.")
    parser.add_argument("--start_time", required=True, help="Start timestamp, e.g. 2016-07-01 00:00:00.")
    parser.add_argument("--freq", default="5min", help="Pandas frequency string. PeMSD4/8 usually use 5min.")
    return parser.parse_args()


def _load_data(npz_path):
    payload = np.load(npz_path, allow_pickle=True)
    if "data" in payload.files:
        data = payload["data"]
    elif len(payload.files) == 1:
        data = payload[payload.files[0]]
    else:
        raise ValueError(f"Cannot infer data array from {npz_path}; keys={payload.files}")
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"Expected data shape [T,N,C], got {data.shape}")
    return data.astype(np.float32)


def _normalise_edge_columns(df):
    lower = {str(col).lower(): col for col in df.columns}
    source_candidates = ["from", "from_node", "src", "source", "i", "start"]
    target_candidates = ["to", "to_node", "dst", "target", "j", "end"]
    distance_candidates = ["distance", "dist", "cost", "weight"]
    src = next((lower[name] for name in source_candidates if name in lower), None)
    dst = next((lower[name] for name in target_candidates if name in lower), None)
    dist = next((lower[name] for name in distance_candidates if name in lower), None)
    return src, dst, dist


def _load_adjacency(csv_path, num_nodes):
    df = pd.read_csv(csv_path)
    if df.shape[0] == num_nodes and df.shape[1] == num_nodes:
        adj = df.to_numpy(dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj

    src_col, dst_col, dist_col = _normalise_edge_columns(df)
    if src_col is None or dst_col is None:
        if df.shape[1] >= 2:
            src_col, dst_col = df.columns[0], df.columns[1]
            dist_col = df.columns[2] if df.shape[1] >= 3 else None
        else:
            raise ValueError(f"Cannot infer edge columns from {csv_path}: columns={df.columns.tolist()}")

    src = df[src_col].to_numpy(dtype=int)
    dst = df[dst_col].to_numpy(dtype=int)
    if src.min(initial=0) >= 1 and dst.min(initial=0) >= 1 and src.max(initial=0) <= num_nodes and dst.max(initial=0) <= num_nodes:
        src = src - 1
        dst = dst - 1
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src = src[valid]
    dst = dst[valid]

    if dist_col is not None:
        distance = df[dist_col].to_numpy(dtype=np.float32)[valid]
        positive = distance[distance > 0]
        sigma = float(np.std(positive)) if positive.size else 1.0
        sigma = sigma if sigma > 0 else 1.0
        weight = np.exp(-np.square(distance / sigma)).astype(np.float32)
    else:
        weight = np.ones_like(src, dtype=np.float32)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[src, dst] = weight
    adj[dst, src] = np.maximum(adj[dst, src], weight)
    np.fill_diagonal(adj, 0.0)
    return adj


def _spectral_positions(adj):
    sym = np.maximum(adj, adj.T).astype(np.float32)
    degree = sym.sum(axis=1)
    lap = np.diag(degree) - sym
    try:
        _, eigvecs = np.linalg.eigh(lap)
        if eigvecs.shape[1] >= 3:
            pos = eigvecs[:, 1:3]
        else:
            pos = eigvecs[:, :2]
    except np.linalg.LinAlgError:
        theta = np.linspace(0, 2 * np.pi, adj.shape[0], endpoint=False)
        pos = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if pos.shape[1] == 1:
        pos = np.concatenate([pos, np.zeros_like(pos)], axis=1)
    return pos.astype(np.float32)


def main():
    args = parse_args()
    data = _load_data(args.npz_path)
    adj = _load_adjacency(args.csv_path, data.shape[1])
    pos_stamp = _spectral_positions(adj)
    time_stamp = pd.date_range(args.start_time, periods=data.shape[0], freq=args.freq).to_numpy()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        np.save(
            handle,
            {
                "data": data,
                "adj": adj,
                "pos_stamp": pos_stamp,
                "time_stamp": time_stamp,
            },
            allow_pickle=True,
        )
    print(f"saved {out_path}")
    print(f"data={data.shape} adj={adj.shape} pos={pos_stamp.shape} time={time_stamp.shape}")


if __name__ == "__main__":
    main()
