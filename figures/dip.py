"""Deep Image Prior baseline for figure_12.

Imports the network and helpers from a clone of https://github.com/DmitryUlyanov/deep-image-prior
(Apache License 2.0, Copyright Dmitry Ulyanov) next to this file:

    git clone https://github.com/DmitryUlyanov/deep-image-prior.git figures/deep-image-prior
    git -C figures/deep-image-prior checkout 042e0d4c1e93f4b1eb0932781de55b8cff5e0f40
"""

import importlib
import importlib.util
import sys
from pathlib import Path

DIP_DIR = Path(__file__).resolve().parent / 'deep-image-prior'


def _load_package(alias, name):
    # the packages are called models and utils, which clash with the DAS-CN2S modules
    init = DIP_DIR / name / '__init__.py'
    if not init.exists():
        raise ImportError(f'Deep Image Prior not found in {DIP_DIR}, see the docstring of {__file__}')
    spec = importlib.util.spec_from_file_location(alias, init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


skip = _load_package('dip_models', 'models').skip
_load_package('dip_utils', 'utils')
_common_utils = importlib.import_module('dip_utils.common_utils')
get_noise = _common_utils.get_noise
get_params = _common_utils.get_params
np_to_torch = _common_utils.np_to_torch
torch_to_np = _common_utils.torch_to_np
optimize = _common_utils.optimize
