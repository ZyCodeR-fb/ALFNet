import importlib


model_dict = {
    "ALFGraph": "model.ALFGraph.ALFGraph",
    "ALFNet": "model.ALFNet.ALFNet",
    "ALFTemporal": "model.ALFTemporal.ALFTemporal",
    "MoGERNN": "model.comparison_model.MoGERNN",
    "STFiT": "model.comparison_model.STFiT",
    "STGCN": "model.comparison_model.STGCN",
    "STGODE": "model.comparison_model.stgode",
    "HA": "model.comparison_model.HA",
    "FCLSTM": "model.comparison_model.FCLSTM",
    "GWNET": "model.comparison_model.GWNet",
    "STID": "model.comparison_model.STID",
    "PatchSTG": "model.comparison_model.PatchSTG",
    "SLPF": "model.comparison_model.SLPF",
    "STBP": "model.comparison_model.STBP",
    "MCPST_FSL": "model.comparison_model.MCPST_FSL",
}


def load_model_class(name):
    try:
        module_path = model_dict[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc
    return importlib.import_module(module_path).Model
