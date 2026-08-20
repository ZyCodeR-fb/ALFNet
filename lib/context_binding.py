"""Shared construction-time checks for v4 model entry points."""


def assert_config_agrees_with_context(configs, runtime_context, who):
    """A config that contradicts the context must lose loudly, not silently.

    Node count, dataset, target and channel are facts about the data. If a YAML
    disagrees with the payload the run will read, one of them is wrong and the
    answer is not "whichever the code happens to consult first".

    The target is checked by *name*, not only by width. Every dataset here is
    single-channel, so a config declaring speed while the payload is flow has
    the right shape everywhere and produces a run that trains on one quantity
    and reports another. That is exactly the failure a shape check cannot see:
    the v3 PEMS08 configs declare `v` while the v4 payload's target is `q`.
    """
    identity = runtime_context.static_asset_identity()
    checks = (
        ("node_num", identity["full_num_nodes"]),
        ("num_nodes", identity["full_num_nodes"]),
        ("target_channel", identity["target_channel"]),
    )
    for attribute, expected in checks:
        declared = getattr(configs, attribute, None)
        if declared is not None and int(declared) != int(expected):
            raise ValueError(
                f"{who}: config declares {attribute}={declared} but the runtime "
                f"context has {expected}; refusing rather than silently winning")
    declared_dataset = getattr(configs, "dloader_name", None)
    if declared_dataset:
        normalised = str(declared_dataset).replace("-dynamic", "").replace("-", "")
        if normalised.upper() != identity["dataset"].replace("-", "").upper():
            raise ValueError(
                f"{who}: config dataset {declared_dataset!r} does not match the "
                f"context dataset {identity['dataset']!r}")
    assert_target_agrees_with_context(configs, runtime_context, who)


def assert_target_agrees_with_context(configs, runtime_context, who):
    """`input_features` and `target_features` must name the context's target."""
    identity = runtime_context.static_asset_identity()
    expected = str(identity["target"])
    for attribute in ("input_features", "target_features"):
        declared = getattr(configs, attribute, None)
        if declared is None:
            raise ValueError(
                f"{who}: config declares no {attribute}; the v4 target is named, "
                "never inferred from a channel width")
        declared = str(declared)
        if len(declared) != 1:
            raise ValueError(
                f"{who}: config declares {attribute}={declared!r}; v4 runs one "
                "channel and a multi-feature string would silently select the "
                "first")
        if declared != expected:
            raise ValueError(
                f"{who}: config declares {attribute}={declared!r} but the "
                f"{identity['dataset']} payload's target is {expected!r} at "
                f"channel {identity['target_channel']}; both are one channel "
                "wide, so nothing downstream would notice")
