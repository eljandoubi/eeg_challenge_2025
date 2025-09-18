from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def conversion(path: str, interpolate: bool = True) -> OrderedDict:
    try:
        # It's good practice to load to the CPU first
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(
            "Please replace 'path/to/your/pretrained_weights.pth' with the actual file path."
        )
        state_dict = {}  # Assign empty dict to prevent further errors in this example

    # The weights might be inside a 'model' key in the checkpoint file
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # 3. Create a new state_dict with corrected key names
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Skip layers from the pre-training heads
        if "mask_token" in key:
            continue

        new_key: str = key

        # Remove the 'student.' prefix
        if new_key.startswith("student."):
            new_key = new_key.replace("student.", "")

            if new_key == "pos_embed":
                new_key = "position_embedding"
            elif new_key == "time_embed":
                new_key = "temporal_embedding"
                if interpolate:
                    value = F.interpolate(
                        value.transpose(-1, -2),
                        scale_factor=1 / 8,
                    ).transpose(-1, -2)
                else:
                    continue
            elif "patch_embed.conv" in new_key:
                new_key = new_key.replace(
                    "patch_embed.conv", "patch_embed.temporal_conv.conv"
                )
            elif "patch_embed.norm" in new_key:
                new_key = new_key.replace(
                    "patch_embed.norm", "patch_embed.temporal_conv.norm"
                )
            elif "mlp.fc1" in new_key:
                new_key = new_key.replace("mlp.fc1", "mlp.0")
            elif "mlp.fc2" in new_key:
                new_key = new_key.replace("mlp.fc2", "mlp.2")
            elif "lm_head" in new_key:
                new_key = new_key.replace("lm_head", "final_layer")
            elif new_key.startswith("norm."):
                new_key = new_key.replace("norm.", "fc_norm.")

            new_state_dict[new_key] = value

    return new_state_dict


if __name__ == "__main__":
    from braindecode.models import Labram

    model = Labram(
        n_times=200,
        n_chans=128,
        n_outputs=8192,
        init_values=0.1,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    states = conversion("src/weights/labram-base.pth")
    mis = model.load_state_dict(states, strict=False)
    assert len(mis.unexpected_keys) == 0
    print(mis.missing_keys)
