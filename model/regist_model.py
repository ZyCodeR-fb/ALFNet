import importlib


model_dict = {
    "ALFGraph": "model.ALFGraph.ALFGraph",
    "ALFNet": "model.ALFNet.ALFNet",
    "ALFTemporal": "model.ALFTemporal.ALFTemporal",
    "MoGERNN": "model.comparison_model.MoGERNN",
    "STGCN": "model.comparison_model.STGCN",
    "STID": "model.comparison_model.STID",
    "PatchSTG": "model.comparison_model.PatchSTG",
    "SLPF": "model.comparison_model.SLPF",
    "STBP": "model.comparison_model.STBP",
}


def load_model_class(name):
    try:
        module_path = model_dict[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc
    return importlib.import_module(module_path).Model
