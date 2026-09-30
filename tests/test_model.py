import pytest
import torch

from configs import GumbelConfig, VQVALossesWeights
from model import VQVAE, VectorQuantizerEmbedding

# n_times is fixed at 200 because the encoder's patch size (from braindecode's
# Labram defaults) is not parameterized and implicitly expects 200 samples per
# window (matching the 2s @ 100Hz windows produced by data_utils). Only the
# channel/vocab/embedding sizes are shrunk here to keep the test fast on CPU.
TINY_CFG = {
    "n_times": 200,
    "n_chans": 4,
    "vocab_size": 32,
    "embedding_dim": 200,
    "drop_rate": 0.0,
}


def make_input(batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, TINY_CFG["n_chans"], TINY_CFG["n_times"])


def test_vector_quantizer_requires_embedding_dim_at_least_vocab_size():
    with pytest.raises(AssertionError):
        VectorQuantizerEmbedding(n_chans=4, vocab_size=64, embedding_dim=32)


def test_vqvae_requires_embedding_dim_at_least_n_times():
    with pytest.raises(AssertionError):
        VQVAE(n_times=200, n_chans=4, vocab_size=32, embedding_dim=64)


def test_vqvae_forward_produces_finite_scalar_losses():
    torch.manual_seed(0)
    vqvae = VQVAE(**TINY_CFG)
    losses = vqvae(make_input())

    for value in losses.to_dict().values():
        assert value.shape == ()
        assert torch.isfinite(value)


def test_vqvae_backward_populates_codebook_gradient():
    torch.manual_seed(0)
    vqvae = VQVAE(**TINY_CFG)
    losses = vqvae(make_input())
    loss = VQVALossesWeights() * losses
    loss.backward()

    assert vqvae.vq.codebook.grad is not None
    assert vqvae.vq.codebook.grad.norm().item() > 0


def test_vqvae_default_gumbel_config_is_not_shared_between_instances():
    """Regression test: gumbel_cfg used to default to a shared mutable instance."""
    vqvae_a = VQVAE(**TINY_CFG)
    vqvae_b = VQVAE(**TINY_CFG)
    assert vqvae_a.gumbel_cfg is not vqvae_b.gumbel_cfg

    vqvae_a.update_gumbel_temperature()

    assert vqvae_a.gumbel_cfg.step == 1
    assert vqvae_b.gumbel_cfg.step == 0


def test_vqvae_explicit_gumbel_config_is_reused():
    shared = GumbelConfig(step=5)
    vqvae = VQVAE(**TINY_CFG, gumbel_cfg=shared)
    assert vqvae.gumbel_cfg is shared
