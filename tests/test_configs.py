from pathlib import Path

import pytest
import torch

from configs import DataConfig, VQVAEConfig, VQVAETrainingLosses, VQVALossesWeights


def test_basic_to_dict():
    cfg = VQVAEConfig()
    d = cfg.to_dict()
    assert d["n_times"] == 200
    assert d["drop_rate"] == 0.1


def test_data_config_single_task_becomes_str(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # DataConfig creates directories relative to cwd
    cfg = DataConfig(data_path=str(tmp_path / "data"), tasks="RestingState")
    assert cfg.tasks == "RestingState"
    assert isinstance(cfg.data_path, Path)
    assert cfg.data_path.is_dir()


def test_data_config_multi_task_stays_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(
        data_path=str(tmp_path / "data"), tasks="RestingState,symbolSearch"
    )
    assert cfg.tasks == ["RestingState", "symbolSearch"]


def test_data_config_default_train_inter_excludes_five(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(data_path=str(tmp_path / "data"))
    assert cfg.train_inter == list(range(1, 5)) + list(range(6, 12))


def test_data_config_explicit_train_inter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(data_path=str(tmp_path / "data"), train_inter="1,2,3")
    assert cfg.train_inter == [1, 2, 3]


def test_data_config_creates_directories(tmp_path):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    DataConfig(
        data_path=str(tmp_path / "raw"),
        train_data=str(train_dir),
        eval_data=str(eval_dir),
    )
    assert train_dir.is_dir()
    assert eval_dir.is_dir()


def test_vqvae_training_losses_to_dict_variants():
    losses = VQVAETrainingLosses(
        amplitude_loss=torch.tensor(1.0, requires_grad=True),
        phase_loss=torch.tensor(2.0),
        orthogonality_loss=torch.tensor(3.0),
        encoder_perplexity=torch.tensor(4.0),
    )

    items = losses.to_dict(post="item")
    assert items == {
        "amplitude_loss": 1.0,
        "phase_loss": 2.0,
        "orthogonality_loss": 3.0,
        "encoder_perplexity": 4.0,
    }

    detached = losses.to_dict(post="detach")
    assert all(not t.requires_grad for t in detached.values())


def test_vqvae_training_losses_add():
    a = VQVAETrainingLosses(
        amplitude_loss=torch.tensor(1.0),
        phase_loss=torch.tensor(1.0),
        orthogonality_loss=torch.tensor(1.0),
        encoder_perplexity=torch.tensor(1.0),
    )
    b = VQVAETrainingLosses(
        amplitude_loss=torch.tensor(2.0),
        phase_loss=torch.tensor(2.0),
        orthogonality_loss=torch.tensor(2.0),
        encoder_perplexity=torch.tensor(2.0),
    )
    c = a + b
    assert c.amplitude_loss.item() == 3.0
    assert c.encoder_perplexity.item() == 3.0


def test_vqvae_losses_weights_mul():
    w = VQVALossesWeights()
    losses = VQVAETrainingLosses(
        amplitude_loss=torch.tensor(4.0),
        phase_loss=torch.tensor(4.0),
        orthogonality_loss=torch.tensor(4.0),
        encoder_perplexity=torch.tensor(4.0),
    )
    total = w * losses
    assert total.item() == pytest.approx(4.0)


def test_vqvae_losses_weights_update_stays_normalized():
    w = VQVALossesWeights()
    losses = VQVAETrainingLosses(
        amplitude_loss=torch.tensor(1.0),
        phase_loss=torch.tensor(2.0),
        orthogonality_loss=torch.tensor(3.0),
        encoder_perplexity=torch.tensor(4.0),
    )
    w.update(losses)
    total_weight = (
        w.w_amplitude_loss
        + w.w_phase_loss
        + w.w_orthogonality_loss
        + w.w_encoder_perplexity
    )
    assert total_weight == pytest.approx(1.0, abs=1e-5)
