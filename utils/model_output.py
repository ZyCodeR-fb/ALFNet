from typing import Any, Dict, Tuple

import torch


def unpack_model_output(model_output: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if isinstance(model_output, dict):
        if "pred" not in model_output:
            raise ValueError("structured model output must contain a 'pred' tensor")
        return model_output["pred"], {key: value for key, value in model_output.items() if key != "pred"}

    if isinstance(model_output, tuple):
        if not model_output:
            raise ValueError("empty tuple model output is not supported")
        if torch.is_tensor(model_output[0]):
            return model_output[0], {}
        if torch.is_tensor(model_output[-1]):
            return model_output[-1], {}
        raise ValueError("tuple model output does not contain a tensor prediction")

    if torch.is_tensor(model_output):
        return model_output, {}

    raise ValueError(f"Unsupported model output type: {type(model_output)}")


def sum_aux_losses(aux_losses: Any, device=None):
    if not aux_losses:
        return None
    total = None
    for value in aux_losses.values():
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=device)
        total = value if total is None else total + value
    return total
