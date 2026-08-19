"""Shared construction-time checks for v4 model entry points."""

def assert_config_agrees_with_context(configs, runtime_context, who):
    """A config that contradicts the context must lose loudly, not silently.

    Node count, dataset, target and channel are facts about the data. If a YAML
    disagrees with the payload the run will read, one of them is wrong and the
    answer is not "whichever the code happens to consult first".
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
