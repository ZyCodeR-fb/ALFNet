"""One reader for the v4 named batch, shared by every v4 model.

Two models now consume the same batch, and the whole point of the exercise is
that they consume the *same* tensor. Two copies of this validation would be two
places for the common input to drift apart, and the drift would be invisible --
both models would still run, on slightly different data. So the checks live here
and each model composes only what is genuinely its own on top.

Everything refuses rather than repairs. A wrong node axis, a batch built under a
different exposure, a missing field: all raise, and none of them fall back to the
v3 ten-tuple. A fallback would look like success at every call site while quietly
reinstating the K-node path.
"""

import torch

COMMON_FIELDS = ("base_visible_input", "encoder_time_marks", "decoder_time_marks",
                 "context_identity")
# What masked teacher forcing needs on top of the common fields.
TARGET_FIELDS = ("target", "decoder_available_mask")


def collapse_batch_identity(identity, who):
    """One identity per batch, or an error. Never a majority vote.

    The default collate turns each string field into a list of B copies and each
    bool into a (B,) tensor. Samples drawn from two different contexts would
    arrive here as a list with two distinct values, which is a mixed batch and
    not something to average over.
    """
    if not isinstance(identity, dict):
        raise ValueError(
            f"{who}: the v4 batch carries no context_identity mapping; the "
            "protocol is never inferred from tensor shapes")
    collapsed = {}
    for key, value in identity.items():
        if torch.is_tensor(value):
            values = [element.item() for element in value.reshape(-1)]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            collapsed[key] = value
            continue
        distinct = sorted(set(values))
        if len(distinct) != 1:
            raise ValueError(
                f"{who}: the v4 batch mixes {key}: {distinct}; samples from two "
                "exposures may not share a batch")
        collapsed[key] = distinct[0]
    return collapsed


def resolve_adjacency(runtime_context, cache, device, dtype):
    """The context's adjacency, converted once and then reused.

    Cached per (device, dtype) so the per-batch check is a shape comparison and
    an already-verified static hash, not a file read or a re-hash. The expensive
    verification happened when the context was built.
    """
    key = (str(device), str(dtype))
    if key not in cache:
        adjacency = runtime_context.adjacency.to(device=device, dtype=dtype)
        nodes = runtime_context.num_nodes
        if tuple(adjacency.shape) != (nodes, nodes):
            raise ValueError(
                f"context adjacency is {tuple(adjacency.shape)}, expected "
                f"({nodes}, {nodes})")
        cache[key] = adjacency
    return cache[key]


def read_common(batch, runtime_context, static_asset_hash, raw_enc_in, who,
                adjacency_cache, extra_node_fields=(), require_target=False):
    """Validate one v4 named batch and return the tensors every model shares.

    `extra_node_fields` are additional `[B, T, N, C]` fields whose node axis is
    checked here but whose meaning belongs to the caller -- ALFNet's
    `lifecycle_context` is the only one, and MoGERNN must never ask for it.
    """
    if runtime_context is None:
        raise ValueError(
            f"{who}.forward_v4 requires a RuntimeDataContext; v4 fails closed "
            "rather than falling back to data_path")
    if static_asset_hash is None:
        raise ValueError(
            f"{who}.forward_v4 requires a model built through "
            "from_runtime_context; the archive path is not a v4 model")
    if static_asset_hash != runtime_context.static_asset_hash():
        raise ValueError(
            f"{who} was built on different static graph assets than the context "
            "this batch came from")
    if not isinstance(batch, dict):
        raise ValueError(
            f"{who}.forward_v4 takes the v4 named batch; the v3 ten-tuple is not "
            "accepted and shapes are never sniffed to tell them apart")
    required = list(COMMON_FIELDS) + list(extra_node_fields)
    if require_target:
        required += list(TARGET_FIELDS)
    missing = [field for field in required if field not in batch]
    if missing:
        raise ValueError(f"{who}: the v4 batch is missing {missing}")

    # Exposure is per batch, not per constructor: one static model runs all three
    # phases, and the phase the batch was drawn under has to match the context
    # the caller is claiming.
    runtime_context.assert_exposure_matches(
        collapse_batch_identity(batch["context_identity"], who),
        f"the v4 batch handed to {who}")

    nodes = runtime_context.num_nodes
    base = batch["base_visible_input"]
    for name in ("base_visible_input",) + tuple(extra_node_fields):
        tensor = batch[name]
        if tensor.dim() != 4:
            raise ValueError(
                f"{who}: {name} must be [batch, time, node, channel], got "
                f"{tuple(tensor.shape)}")
        if tensor.shape[2] != nodes:
            raise ValueError(
                f"{who}: {name} has {tensor.shape[2]} nodes, the context has "
                f"{nodes}; v4 runs on the full static node axis and a short axis "
                "is never padded")
    if base.shape[-1] != raw_enc_in:
        raise ValueError(
            f"{who}: base_visible_input has {base.shape[-1]} channels, the model "
            f"reads {raw_enc_in}")

    marks = batch["encoder_time_marks"].to(base.dtype)
    horizon = batch["decoder_time_marks"].to(base.dtype)
    for name, tensor, steps in (("encoder_time_marks", marks, base.shape[1]),
                                ("decoder_time_marks", horizon, None)):
        if tensor.dim() != 3 or tensor.shape[-1] < 3:
            raise ValueError(
                f"{who}: {name} must be [batch, step, >=3] with minute/hour/"
                f"day-of-week, got {tuple(tensor.shape)}")
        if steps is not None and tensor.shape[1] != steps:
            raise ValueError(
                f"{who}: {name} covers {tensor.shape[1]} steps, the input window "
                f"is {steps}")

    if require_target:
        target = batch["target"]
        if target.dim() != 4 or target.shape[2] != nodes:
            raise ValueError(
                f"{who}: target must be [batch, horizon, node, channel] on the "
                f"full node axis, got {tuple(target.shape)}")
        if batch["decoder_available_mask"].shape != target.shape[:3]:
            raise ValueError(
                f"{who}: decoder_available_mask "
                f"{tuple(batch['decoder_available_mask'].shape)} does not match "
                f"target {tuple(target.shape)}")

    return {
        "adjacency": resolve_adjacency(runtime_context, adjacency_cache,
                                       base.device, base.dtype),
        "base_visible_input": base,
        "encoder_time_marks": marks,
        "decoder_time_marks": horizon,
        "num_nodes": nodes,
    }


def bind_static_assets(model, configs, runtime_context, who):
    """Record what a v4 model may be held to, and refuse a config that disagrees.

    Deliberately the static assets only -- no phase, no split, no scope. Binding
    a constructor to an exposure would make one architecture look like three
    different models across train, validation and test.
    """
    if runtime_context is None:
        raise ValueError(
            f"{who}.from_runtime_context requires a RuntimeDataContext; v4 fails "
            "closed rather than falling back to data_path")
    from lib.context_binding import assert_config_agrees_with_context

    assert_config_agrees_with_context(configs, runtime_context, who)
    identity = runtime_context.static_asset_identity()
    model.runtime_context = runtime_context
    model.static_asset_hash = runtime_context.static_asset_hash()
    model.context_full_num_nodes = identity["full_num_nodes"]
    model.context_node_ids_sha256 = identity["node_ids_sha256"]
    model.context_adjacency_sha256 = identity["adjacency_sha256"]
    model.context_dataset = identity["dataset"]
    model.context_target = identity["target"]
    model.context_target_channel = identity["target_channel"]
    model._v4_adjacency_cache = {}
    return model
