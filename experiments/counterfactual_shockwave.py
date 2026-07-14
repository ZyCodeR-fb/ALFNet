import csv
import heapq
import json
import os
import re

import numpy as np
import torch

from data_provider.graph_data_loader import ROLE_AAS, ROLE_VS, _compose_lifecycle_input_features
from utils.model_output import unpack_model_output


PAPER_LIFECYCLE_PHASES = {
    "commissioning": ("Cold-start", "Warm-up", "Adapted"),
    "failure": ("Steady", "Degrading", "Outage"),
}


def lifecycle_phase(trajectory, tau, input_len):
    if trajectory not in PAPER_LIFECYCLE_PHASES:
        raise ValueError(f"Unsupported lifecycle trajectory: {trajectory}")
    before, transition, settled = PAPER_LIFECYCLE_PHASES[trajectory]
    if tau < 0:
        return before
    if tau < input_len:
        return transition
    return settled


def build_source_hop_partition(adjacency, sources, max_hop=3):
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if max_hop < 1:
        raise ValueError("max_hop must be at least 1")

    num_nodes = adjacency.shape[0]
    sources = sorted({int(node) for node in sources})
    if not sources:
        raise ValueError("sources must not be empty")
    if sources[0] < 0 or sources[-1] >= num_nodes:
        raise ValueError("source node index is out of range")

    edge_mask = np.isfinite(adjacency) & (np.abs(adjacency) > 0)
    edge_mask = edge_mask | edge_mask.T
    np.fill_diagonal(edge_mask, False)
    neighbors = [np.flatnonzero(edge_mask[node]).tolist() for node in range(num_nodes)]

    distances = np.full(num_nodes, np.iinfo(np.int32).max, dtype=np.int64)
    owners = np.full(num_nodes, -1, dtype=np.int64)
    queue = []
    for source in sources:
        distances[source] = 0
        owners[source] = source
        heapq.heappush(queue, (0, source, source))

    while queue:
        distance, owner, node = heapq.heappop(queue)
        if distance != distances[node] or owner != owners[node]:
            continue
        for neighbor in neighbors[node]:
            candidate = (distance + 1, owner)
            current = (distances[neighbor], owners[neighbor])
            if candidate < current:
                distances[neighbor] = distance + 1
                owners[neighbor] = owner
                heapq.heappush(queue, (distance + 1, owner, neighbor))

    partition = {source: {hop: [] for hop in range(max_hop + 1)} for source in sources}
    for node in range(num_nodes):
        owner = int(owners[node])
        if owner < 0:
            continue
        hop = min(int(distances[node]), max_hop)
        partition[owner][hop].append(node)
    return partition


def _override_current_event(visibility, role_ids, event, input_begin, input_end, trajectory):
    if trajectory not in {"control", "commissioning", "failure"}:
        raise ValueError(f"Unsupported intervention trajectory: {trajectory}")
    event_time = int(event["event_time_index"])
    post_event_rows = np.flatnonzero(np.arange(input_begin, input_end) >= event_time)
    if not len(post_event_rows):
        return

    if trajectory != "commissioning" and event.get("nas_nodes"):
        commissioned_nodes = np.asarray(event["nas_nodes"], dtype=np.int64)
        visibility[np.ix_(post_event_rows, commissioned_nodes)] = False
        role_ids[np.ix_(post_event_rows, commissioned_nodes)] = ROLE_VS
    if trajectory != "failure" and event.get("fs_nodes"):
        failed_nodes = np.asarray(event["fs_nodes"], dtype=np.int64)
        visibility[np.ix_(post_event_rows, failed_nodes)] = True
        role_ids[np.ix_(post_event_rows, failed_nodes)] = ROLE_AAS


def build_intervention_sample(dataset, index, event, trajectory):
    input_begin = int(index) * int(dataset.step)
    input_end = input_begin + int(dataset.input_len)
    raw_x = dataset.data[input_begin:input_end].copy()
    visibility = dataset.availability_mask[input_begin:input_end].copy()
    role_ids = dataset._build_role_ids_for_window(input_begin, input_end).copy()
    _override_current_event(visibility, role_ids, event, input_begin, input_end, trajectory)
    seq_x = _compose_lifecycle_input_features(
        raw_x,
        visibility,
        role_ids,
        bool(dataset.lifecycle_role_features),
        bool(dataset.lifecycle_history_age_feature),
    )
    _, seq_y, x_t_mark, y_t_mark = dataset[index]
    return seq_x, seq_y, x_t_mark, y_t_mark


def paired_error_sums(treated, control, target, nodes):
    if not nodes:
        return None
    treated = treated[..., nodes, :]
    control = control[..., nodes, :]
    target = target[..., nodes, :]
    valid = torch.isfinite(target) & torch.isfinite(treated) & torch.isfinite(control)
    if not torch.any(valid):
        return None

    target_valid = target[valid]
    treated_diff = treated[valid] - target_valid
    control_diff = control[valid] - target_valid
    denominator = torch.clamp(torch.abs(target_valid), min=1e-6)
    return {
        "count": int(valid.sum().item()),
        "treated_ape_sum": float(torch.sum(torch.abs(treated_diff) / denominator).item()),
        "control_ape_sum": float(torch.sum(torch.abs(control_diff) / denominator).item()),
        "treated_ae_sum": float(torch.sum(torch.abs(treated_diff)).item()),
        "control_ae_sum": float(torch.sum(torch.abs(control_diff)).item()),
        "treated_se_sum": float(torch.sum(treated_diff**2).item()),
        "control_se_sum": float(torch.sum(control_diff**2).item()),
    }


def _inverse_transform(dataset, tensor):
    std = torch.as_tensor(dataset.std_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    mean = torch.as_tensor(dataset.mean_[..., dataset.tar_feas], dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean


def _to_model_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


def _predict_sample(model, sample, dataset, args, adj, pos_mark):
    seq_x, seq_y, x_t_mark, y_t_mark = sample
    x_enc = _to_model_tensor(seq_x, args.device)
    x_enc = torch.nan_to_num(x_enc, nan=0.0)
    enc_t_mark = _to_model_tensor(x_t_mark, args.device) if args.t_mark else None
    dec_t_mark = _to_model_tensor(y_t_mark, args.device)
    with torch.no_grad():
        output = model(adj, x_enc, enc_t_mark, pos_mark, None, dec_t_mark, 0)
        prediction, _ = unpack_model_output(output)
        prediction = _inverse_transform(dataset, prediction)
        target = _to_model_tensor(seq_y[-args.pred_len :], args.device)
        target = _inverse_transform(dataset, target)
    return prediction.detach().cpu(), target.detach().cpu()


def _hop_label(hop, max_hop):
    return f"{max_hop}+" if hop == max_hop else str(hop)


def _safe_file_token(value):
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_").lower()
    return token or "model"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _event_timestamp(dataset, event_time):
    timestamps = getattr(dataset, "time_stamp_raw", None)
    if timestamps is None or event_time < 0 or event_time >= len(timestamps):
        return None
    return str(timestamps[event_time])


def _write_rows(path, rows):
    fieldnames = [
        "model",
        "event_id",
        "trajectory",
        "source_sensor",
        "window_index",
        "tau",
        "phase",
        "hop",
        "hop_label",
        "target_node_count",
        "count",
        "treated_ape_sum",
        "control_ape_sum",
        "treated_ae_sum",
        "control_ae_sum",
        "treated_se_sum",
        "control_se_sum",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counterfactual_shockwave_inference(model, data_loader, args):
    if int(args.slide_step) != 1:
        raise ValueError("Counterfactual Lifecycle Shockwave requires slide_step=1 for dense replay")
    if args.async_adaptor != "base":
        raise ValueError("Counterfactual Lifecycle Shockwave only supports async_adaptor=base")
    if args.shockwave_pre_steps < 1 or args.shockwave_post_steps < args.input_len:
        raise ValueError("shockwave_pre_steps must be positive and shockwave_post_steps must cover input_len")

    dataset = data_loader.dataset
    bundle = dataset.get_async_lifecycle_bundle()
    event_plan = bundle["event_plan"]
    model_label = args.shockwave_model_name or args.model
    output_dir = os.path.abspath(args.shockwave_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    file_token = _safe_file_token(model_label)
    rows_path = os.path.join(output_dir, f"{file_token}_shockwave_rows.csv")
    metadata_path = os.path.join(output_dir, f"{file_token}_shockwave_meta.json")

    adj_np, pos_np = dataset.get_adj_and_pos_mark()
    adj = torch.as_tensor(adj_np, dtype=torch.float32, device=args.device)
    if args.pos_mark and pos_np is not None:
        pos_mark = torch.as_tensor(pos_np, dtype=torch.float32, device=args.device)
    else:
        pos_mark = None

    rows = []
    paper_events = []
    partitions = []
    model.eval()
    print(f"-> Start Counterfactual Lifecycle Shockwave export for {model_label}...")
    for event in event_plan:
        event_id = int(event["event_id"])
        start_window = int(event["start_window"])
        selected_start = max(0, start_window - int(args.shockwave_pre_steps))
        selected_end = min(len(dataset) - 1, start_window + int(args.shockwave_post_steps))
        trajectory_sources = {
            "commissioning": [int(node) for node in event.get("nas_nodes", [])],
            "failure": [int(node) for node in event.get("fs_nodes", [])],
        }
        trajectory_partitions = {
            trajectory: build_source_hop_partition(adj_np, sources, max_hop=int(args.shockwave_max_hop))
            for trajectory, sources in trajectory_sources.items()
            if sources
        }
        paper_events.append(
            {
                "event_id": event_id,
                "start_window": start_window,
                "event_time_index": int(event["event_time_index"]),
                "event_timestamp": _event_timestamp(dataset, int(event["event_time_index"])),
                "commissioned_nodes": trajectory_sources["commissioning"],
                "failed_nodes": trajectory_sources["failure"],
            }
        )
        partition_record = {"event_id": event_id}
        partition_record.update(
            {
                trajectory: {
                    str(source): {str(hop): nodes for hop, nodes in hop_groups.items()}
                    for source, hop_groups in partition.items()
                }
                for trajectory, partition in trajectory_partitions.items()
            }
        )
        partitions.append(partition_record)

        for window_index in range(selected_start, selected_end + 1):
            tau = window_index - start_window
            control_sample = build_intervention_sample(dataset, window_index, event, "control")
            control_prediction, target = _predict_sample(model, control_sample, dataset, args, adj, pos_mark)
            for trajectory, partition in trajectory_partitions.items():
                if tau <= 0:
                    treated_prediction = control_prediction
                else:
                    treated_sample = build_intervention_sample(dataset, window_index, event, trajectory)
                    treated_prediction, _ = _predict_sample(model, treated_sample, dataset, args, adj, pos_mark)
                phase = lifecycle_phase(trajectory, tau, int(args.input_len))
                for source, hop_groups in partition.items():
                    for hop, target_nodes in hop_groups.items():
                        sums = paired_error_sums(treated_prediction, control_prediction, target, target_nodes)
                        if sums is None:
                            continue
                        rows.append(
                            {
                                "model": model_label,
                                "event_id": event_id,
                                "trajectory": trajectory,
                                "source_sensor": int(source),
                                "window_index": window_index,
                                "tau": tau,
                                "phase": phase,
                                "hop": int(hop),
                                "hop_label": _hop_label(int(hop), int(args.shockwave_max_hop)),
                                "target_node_count": len(target_nodes),
                                **sums,
                            }
                        )
        print(
            f"   event={event_id} | windows={selected_start}:{selected_end} | "
            f"commissioned={len(trajectory_sources['commissioning'])} | failed={len(trajectory_sources['failure'])}"
        )

    _write_rows(rows_path, rows)
    metadata = {
        "schema_version": 1,
        "experiment": "Counterfactual Lifecycle Shockwave",
        "model": model_label,
        "rows_path": rows_path,
        "design": {
            "control": "suppress both current commissioning and failure while preserving earlier events",
            "commissioning_treatment": "apply current commissioning and suppress concurrent failure",
            "failure_treatment": "apply current failure and suppress concurrent commissioning",
            "model_effect": "ACE = MAPE_treated - MAPE_control",
            "comparison_effect": "attenuation_gain = ACE_MoGERNN - ACE_ALFNet",
            "positive_meaning": "ALFNet adapts faster to commissioning or absorbs failure shock better",
        },
        "state_trajectories": PAPER_LIFECYCLE_PHASES,
        "config": {
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "dense_replay_slide_step": int(args.slide_step),
            "pre_steps": int(args.shockwave_pre_steps),
            "post_steps": int(args.shockwave_post_steps),
            "max_hop_bucket": int(args.shockwave_max_hop),
            "async_num_events": int(args.async_num_events),
            "async_commissioning_k": int(args.async_commissioning_k),
            "async_event_seed": int(args.async_event_seed),
            "time_step_minutes": 5,
            "adjacency": "undirected support from nonzero finite entries of A or A^T",
            "source_assignment": "nearest changed sensor; ties use the smaller sensor index",
        },
        "checkpoint": getattr(args, "loaded_checkpoint_metadata", None),
        "events": paper_events,
        "partitions": partitions,
        "row_count": len(rows),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, ensure_ascii=False, indent=2)
    print(f"-> Shockwave rows saved to {rows_path}")
    print(f"-> Shockwave metadata saved to {metadata_path}")
    return {"rows_path": rows_path, "metadata_path": metadata_path, "row_count": len(rows)}
