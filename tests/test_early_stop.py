import torch.nn as nn
from accelerate import Accelerator

from early_stop import EarlyStopping


def make_accelerator(tmp_path) -> Accelerator:
    # Force CPU so this test suite never needs a GPU.
    return Accelerator(cpu=True, project_dir=str(tmp_path))


def test_first_call_always_improves_and_saves_checkpoint(tmp_path):
    accelerator = make_accelerator(tmp_path)
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2, save_path="best.pt")

    stopper.step(1.0, model, accelerator)

    assert stopper.best_score == 1.0
    assert stopper.counter == 0
    assert not stopper.should_stop
    assert (tmp_path / "best.pt").exists()


def test_stops_after_patience_exhausted(tmp_path):
    accelerator = make_accelerator(tmp_path)
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2, save_path="best.pt")

    stopper.step(1.0, model, accelerator)  # improvement -> counter 0
    stopper.step(1.0, model, accelerator)  # no improvement -> counter 1
    assert not stopper.should_stop

    stopper.step(1.0, model, accelerator)  # no improvement -> counter 2 == patience
    assert stopper.should_stop


def test_improvement_resets_counter(tmp_path):
    accelerator = make_accelerator(tmp_path)
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2, save_path="best.pt")

    stopper.step(1.0, model, accelerator)
    stopper.step(1.0, model, accelerator)  # counter -> 1
    stopper.step(0.5, model, accelerator)  # improvement -> counter reset

    assert stopper.counter == 0
    assert stopper.best_score == 0.5
    assert not stopper.should_stop


def test_min_delta_requires_meaningful_improvement(tmp_path):
    accelerator = make_accelerator(tmp_path)
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=1, min_delta=0.1, save_path="best.pt")

    stopper.step(1.0, model, accelerator)
    stopper.step(0.95, model, accelerator)  # improvement smaller than min_delta

    assert stopper.should_stop
