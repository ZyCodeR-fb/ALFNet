import importlib as _importlib

ALFGraph = _importlib.import_module("model.ALFGraph.ALFGraph")
ALFNet = _importlib.import_module("model.ALFNet.ALFNet")
ALFTemporal = _importlib.import_module("model.ALFTemporal.ALFTemporal")

from .comparison_model import (
    FCLSTM,
    GWNet as GWNET,
    HA,
    MCPST_FSL,
    MoGERNN,
    PatchSTG,
    SLPF,
    STBP,
    STFiT,
    STGCN,
    STGODE,
    STID,
)
