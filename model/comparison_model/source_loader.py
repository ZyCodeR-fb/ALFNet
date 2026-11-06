"""Import byte-identical released source trees that use absolute ``src.*`` imports.

STONE and STOP ship as ``src/...`` packages. Their copies under
``model/comparison_model/<method>/source/src`` are loaded with that directory
temporarily first on ``sys.path``; afterwards every ``src`` entry is removed
again, so the two releases never see each other's ``src`` and nothing in the
project resolves to them. Import-time global side effects of the released files
(the autograd anomaly-detection switch and ``CUDA_LAUNCH_BLOCKING``) are debugging
aids, not computation, and STONE's ``multiprocessing.set_start_method('spawn',
True)`` only changes how worker processes would be started; all three are
restored to their prior values and the change is reported to the caller.
"""
import importlib
import multiprocessing
import os
import sys

import torch


def load_released(source_root, module_names, top='src'):
    saved = {k: v for k, v in sys.modules.items() if k == top or k.startswith(top + '.')}
    for key in saved:
        del sys.modules[key]
    anomaly = torch.is_anomaly_enabled()
    blocking = os.environ.get('CUDA_LAUNCH_BLOCKING')
    start_method = multiprocessing.get_start_method(allow_none=True)
    sys.path.insert(0, str(source_root))
    try:
        modules = {name: importlib.import_module(name) for name in module_names}
        for name, module in modules.items():
            if not str(module.__file__).startswith(str(source_root)):
                raise RuntimeError(f'{name} did not load from {source_root}')
    finally:
        sys.path.remove(str(source_root))
        for key in [k for k in sys.modules if k == top or k.startswith(top + '.')]:
            del sys.modules[key]
        sys.modules.update(saved)
        side_effects = {'detect_anomaly_set': torch.is_anomaly_enabled() != anomaly,
                        'cuda_launch_blocking_set': os.environ.get('CUDA_LAUNCH_BLOCKING') != blocking,
                        'mp_start_method_set': multiprocessing.get_start_method(allow_none=True) != start_method}
        torch.set_anomaly_enabled(anomaly)
        multiprocessing.set_start_method(start_method, force=True)
        if blocking is None:
            os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
        else:
            os.environ['CUDA_LAUNCH_BLOCKING'] = blocking
    return modules, side_effects
