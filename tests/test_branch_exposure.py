"""What ALFNet's two branches are allowed to see.

ALFGraph runs on the K known nodes. ALFTemporal scatters the same K-node tensor
into a full-N tensor, fills the N-K held-out slots with zeros, attends across all
N, then gathers K back. The zeros are a value the model invented for nodes it
does not have, and spatial attention mixes them into the known-node
representation.

The acceptance test for that fix is `xfail(strict=True)` rather than an assertion
of the current behaviour, so it starts passing by itself the moment the branch
stops depending on the fabricated fill -- and fails loudly if someone marks the
work done without changing the behaviour.
"""

import pytest
import torch

from scripts.exposure_probe import probe


@pytest.fixture(scope="module")
def report():
    return probe("configs/ALFNet/alfnet.yaml")


def test_the_probe_is_not_vacuous(report):
    """A fabricated fill exists at all: there must be held-out nodes to fill."""
    assert report["held_out_nodes"] > 0
    assert report["known_nodes"] + report["held_out_nodes"] == report["full_nodes"]


def test_forecast_sensitivity_is_uninformative_at_random_init(report):
    """Pins why the forecast-level number may not be read as evidence.

    `nn.init.zeros_(self.output_proj.weight)` makes the FiLM head an identity at
    initialisation, so the conditioner contributes exactly 0.0 whatever it is
    fed. A zero here is a property of the initialisation, not of the exposure.
    """
    assert all(value == 0.0 for value in report["forecast_sensitivity"].values())


@pytest.mark.xfail(strict=True, reason=(
    "Stage 2 Part B is not implemented: ALFTemporal still fabricates zeros for "
    "held-out nodes and its output depends on them. This passes when the branch "
    "takes a masked full-N input with an explicit availability mask instead."))
def test_alftemporal_output_does_not_depend_on_fabricated_held_out_values(report):
    for fill, sensitivity in report["branch_sensitivity"].items():
        assert sensitivity == 0.0, (
            f"filling held-out slots with {fill} moved ALFTemporal's features by "
            f"{sensitivity:.6f}; the branch is reading invented values")
