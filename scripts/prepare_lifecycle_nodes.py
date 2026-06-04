import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lifecycle node split files for async lifecycle experiments.")
    parser.add_argument("--data_path", required=True, help="Project raw_data file containing a data array.")
    parser.add_argument("--out_dir", required=True, help="Directory to save unknown/new_add/failed node files.")
    parser.add_argument("--unknown_ratio", type=float, default=0.25)
    parser.add_argument("--event_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=32)
    return parser.parse_args()


def load_num_nodes(data_path):
    raw = np.load(data_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    return int(raw["data"].shape[1])


def main():
    args = parse_args()
    num_nodes = load_num_nodes(args.data_path)
    rng = np.random.default_rng(args.seed)

    unknown_count = max(1, int(round(num_nodes * args.unknown_ratio)))
    event_count = max(1, int(round(num_nodes * args.event_ratio)))
    if unknown_count < event_count:
        raise ValueError(
            f"unknown_count={unknown_count} is smaller than event_count={event_count}; "
            "increase unknown_ratio or decrease event_ratio."
        )
    if num_nodes - unknown_count < event_count:
        raise ValueError(
            f"known_count={num_nodes - unknown_count} is smaller than event_count={event_count}; "
            "decrease unknown_ratio or event_ratio."
        )

    all_nodes = np.arange(num_nodes)
    unknown_nodes = np.sort(rng.choice(all_nodes, size=unknown_count, replace=False))
    known_nodes = np.setdiff1d(all_nodes, unknown_nodes)
    new_add_nodes = np.sort(rng.choice(unknown_nodes, size=event_count, replace=False))
    failed_nodes = np.sort(rng.choice(known_nodes, size=event_count, replace=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "unknown_nodes.npy", unknown_nodes.astype(np.int64))
    np.save(out_dir / "new_add_nodes.npy", new_add_nodes.astype(np.int64))
    np.save(out_dir / "failed_nodes.npy", failed_nodes.astype(np.int64))
    print(f"num_nodes={num_nodes}")
    print(f"unknown_nodes={len(unknown_nodes)} -> {out_dir / 'unknown_nodes.npy'}")
    print(f"new_add_nodes={len(new_add_nodes)} -> {out_dir / 'new_add_nodes.npy'}")
    print(f"failed_nodes={len(failed_nodes)} -> {out_dir / 'failed_nodes.npy'}")


if __name__ == "__main__":
    main()
