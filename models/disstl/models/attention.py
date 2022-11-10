"""
Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/recorder.py
"""
from typing import Tuple

import torch
from timm.models.vision_transformer import Attention
from torch import nn


class Recorder(nn.Module):
    """Module with additional hooks for recording attention maps of each transformer layer

    args:
        model:  an Encoder model with a Vision Transformer (ViT) backbone
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False

    def _hook(self, _, input: torch.Tensor, output: torch.Tensor):
        self.recordings.append(output.clone().detach())

    def _register_hook(self) -> None:
        modules = [m for m in self.model.modules() if isinstance(m, Attention)]
        for module in modules:
            handle = module.attn_drop.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self) -> nn.Module:
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self) -> None:
        self.recordings.clear()

    def record(self, attn: torch.Tensor) -> None:
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, c, h, w) or (batch_size, c, h, w)

        returns:
            z_t: (batch_size, seq_len, z_dim) or (batch_size, z_dim)
            attns: (batch_size, seq_len, num_layers, num_heads, num_patches+1, num_patches+1) or
                    (batch_size, num_layers, num_heads, num_patches+1, num_patches+1)
        """
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()

        if not self.hook_registered:
            self._register_hook()

        # image sequence
        if x.ndim == 5:
            bs, seq_len, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
            z_t = self.model(x)
            z_t = z_t.reshape(bs, seq_len, -1)
            attns = torch.stack(self.recordings, dim=1)
            attns = attns.reshape(bs, seq_len, *attns.shape[1:])
        # single image
        else:
            z_t = self.model(x)
            attns = torch.stack(self.recordings, dim=1)

        return z_t, attns
