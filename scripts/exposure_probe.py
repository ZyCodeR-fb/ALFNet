"""Measure what each ALFNet branch is allowed to see.

ALFGraph runs on the K known nodes. ALFTemporal takes the same K-node tensor and
scatters it into a full-N tensor, filling the N-K held-out slots with zeros, runs
spatial attention over all N, then gathers K back out. Those zeros are a
fabricated signal for nodes the model is not supposed to have, and this script
measures whether they reach the branch's output.

    python scripts/exposure_probe.py --config configs/ALFNet/alfnet.yaml

Read the forecast-level number with care: the FiLM head is zero-initialised
(`nn.init.zeros_(self.output_proj.weight)`), so on a randomly initialised model
the conditioner contributes exactly 0.0 whatever it is fed, and a zero there says
nothing about a trained model. The branch-level number is the informative one.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_inputs(args, num_known, seed=7):
    """A Steady, fully-observed batch: [raw, role x4, age, obs_mask]."""
    generator = torch.Generator().manual_seed(seed)
    x_enc = torch.zeros(2, args.input_len, num_known, args.enc_in)
    x_enc[..., 0] = torch.randn(2, args.input_len, num_known, generator=generator)
    x_enc[..., 2] = 1.0                                   # Steady role
    x_enc[..., args.enc_in - 2] = 1.0                     # age
    x_enc[..., args.enc_in - 1] = 1.0                     # observed
    return (torch.rand(num_known, num_known, generator=generator), x_enc,
            torch.rand(2, args.input_len, 3, generator=generator) * 5,
            torch.rand(2, args.pred_len, 3, generator=generator) * 5)


def probe(config_path):
    from args import load_args_from_config
    from model.regist_model import load_model_class

    args = load_args_from_config(config_path, [])
    args.device = "cpu"
    net = load_model_class(args.model)(args).eval()
    patch = net.patch_expert
    num_known, num_nodes = len(net.known_nodes), net.full_num_nodes
    adjacency, x_enc, marks, horizon = build_inputs(args, num_known)
    scatter = patch._scatter_to_full_nodes

    captured = {}
    def capture(tag):
        return lambda module, inputs, output: captured.__setitem__(
            tag, output["features"].detach().clone())

    def run(tag, fill=None):
        if fill is not None:
            def spy(tensor, ids):
                full = scatter(tensor, ids).clone()
                held = torch.ones(num_nodes, dtype=torch.bool)
                held[list(ids)] = False
                full[:, :, held, :] = fill
                return full
            patch._scatter_to_full_nodes = spy
        handle = patch.register_forward_hook(capture(tag))
        with torch.no_grad():
            prediction = net(adjacency, x_enc, marks, None, None, horizon, 0)["pred"]
        handle.remove()
        patch._scatter_to_full_nodes = scatter
        return prediction

    base = run("base")
    report = {"known_nodes": num_known, "full_nodes": num_nodes,
              "held_out_nodes": num_nodes - num_known,
              "branch_sensitivity": {}, "forecast_sensitivity": {}}
    for fill in (1.0, -3.0):
        prediction = run(str(fill), fill=fill)
        report["branch_sensitivity"][str(fill)] = float(
            (captured[str(fill)] - captured["base"]).abs().max())
        report["forecast_sensitivity"][str(fill)] = float((prediction - base).abs().max())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ALFNet/alfnet.yaml")
    report = probe(parser.parse_args().config)
    import json
    print(json.dumps(report, indent=2))
    print("\nbranch_sensitivity > 0 means the fabricated held-out fill reaches "
          "ALFTemporal's output.\nforecast_sensitivity is uninformative at random "
          "init: the FiLM head is zero-initialised.")


if __name__ == "__main__":
    main()
