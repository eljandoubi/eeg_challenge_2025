import importlib

import pytest

# Importing every src module must stay free of network/GPU/filesystem side
# effects, so this is safe to run in any CPU-only CI environment.
MODULES = [
    "configs",
    "data_utils",
    "torch_dataset",
    "model",
    "early_stop",
    "prepare_yaml",
    "check_subjects",
    "pretrainer",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
