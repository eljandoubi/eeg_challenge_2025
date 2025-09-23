from functools import partial
import torch
from torch import nn
from braindecode.models import Labram
from einops import rearrange

encoder = Labram(
    n_times=200,
    n_chans=128,
    qkv_bias=True,
    n_outputs=16,
)

# decoder = Labram(
#     n_times=200,
#     n_chans=128,
#     in_channels=128,
#     patch_size=25,
#     n_outputs=0,
#     init_values=0.1,
#     qkv_bias=False,
#     qk_norm=partial(nn.LayerNorm, eps=1e-6),
#     norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     neural_tokenizer=False,
#     use_abs_pos_emb=False,
# )

decoder = Labram(
    n_times=16,
    n_chans=128,
    qkv_bias=True,
    n_outputs=16,
)

x = torch.randn(5, 128, 200)

y = encoder(x, return_patch_tokens=True)

print(y.shape)
# z = rearrange(y, "b (c p) s -> b c p s", c=128)
# print(z.shape)

x_ = decoder(y)
print(x_.shape)
