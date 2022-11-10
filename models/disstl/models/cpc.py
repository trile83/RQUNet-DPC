import collections
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from disstl.models.encoder import Autoregressor, Encoder


class Predictor(nn.Module):
    """
    Predictor Network for predicting k future codes given context vector

    args:
        input_dim:   input code dim
        output_dim:  output code dim
        k:           num fc layers for k future predictions
    """

    def __init__(self, input_dim: int, output_dim: int, k: int):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(k)])

    def forward(self, c_t: torch.Tensor) -> torch.Tensor:
        """
        args:
            c_t: (batch_size, c_dim)

        returns:
            z_hat: (batch_size, k, emb_dim)
        """
        z_hat = []
        for W_k in self.W:
            z_hat.append(W_k(c_t))  # W_k * c_t

        # (bs, k, z_dim)
        z_hat = torch.stack(z_hat, dim=1)

        return z_hat


class CPC(nn.Module):
    """
    Contrastive Predictive Coding (CPC) module

    args:
        genc_backbone:  valid model in torchvision.models
        gar_hidden_dim: hidden_dim (also output_dim due to n_layers=1) of gar
        channels:       num input channels
        t:              context vector used at step t for predicting future codes
        k:              codes from z[t:t+k] are predicted using c_t
        pretrained:     use pretrained weights or not
        image_size:     input image size (vit only)
        patch_size:     size of image patches (vit only)
        depth:          num transformer layers (vit only)
    """

    def __init__(
        self,
        genc_backbone: str,
        gar_hidden_dim: int,
        channels: int,
        t: int,
        k: int,
        pretrained: bool = False,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        super().__init__()
        self.genc_backbone = genc_backbone
        self.gar_hidden_dim = self.c_dim = gar_hidden_dim
        self.t = t
        self.k = k

        self.genc = Encoder(
            backbone=genc_backbone,
            channels=channels,
            pretrained=pretrained,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
        )
        self.gar = Autoregressor(input_dim=self.genc.emb_dim, hidden_dim=gar_hidden_dim)
        self.W = Predictor(input_dim=gar_hidden_dim, output_dim=self.genc.emb_dim, k=self.k)
        self.z_dim = self.genc.emb_dim
        self.backbone_parameters = self.parameters()
        self.inference = False
        self.result_tuple = collections.namedtuple("cpc_result", ["z", "z_hat", "z_t", "c_t", "c", "h"])

    def freeze(self, freeze: bool) -> None:
        """Freeze CPC weights excluding classifier"""
        requires_grad = not freeze

        for param in self.genc.parameters():
            param.requires_grad = requires_grad

        for param in self.gar.parameters():
            param.requires_grad = requires_grad


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape

        # Encode all images. Combine all images in sequences to process at once
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z_t = self.genc(x)
        z_t = rearrange(z_t, "(b t) z -> b t z", t=seq_len)

        # Process sequences using context model
        # c: (bs, seq_len, c_dim)
        # h: (bs, c_dim)
        c, h = self.gar(z_t)

        # Predict future codes produced by genc using context at time t
        # c_t: (bs, c_dim) at index t
        if self.inference:
            c_t = torch.zeros(bs, self.c_dim).to(c.device).to(c.dtype)
        else:
            c_t = c[:, self.t - 1, :]

        # (bs, k z_dim)
        if self.inference:
            z_hat = torch.zeros(bs, self.k, self.z_dim).to(c_t.device).to(c_t.dtype)
        else:
            z_hat = self.W(c_t)

        # Only return ground truth for codes being predicted
        if self.inference:
            z = torch.zeros(bs, self.k, self.z_dim).to(z_t.device).to(z_t.dtype)
        else:
            z = z_t[:, self.t : (self.t + self.k), :]

        return self.result_tuple(z=z, z_hat=z_hat, z_t=z_t, c_t=c_t, c=c, h=h)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            z: (batch_size, k, emb_dim)
            c_t: (batch_size, c_dim)
        """
        bs, seq_len, c, h, w = x.shape

        # Encode all images. Combine all images in sequences to process at once
        # (bs, seq_len, c, h, w) -> (bs*seq_len, c, h, w)
        x = x.reshape(-1, c, h, w)
        z_t = self.genc(x)

        # Reshape back to sequences
        # (bs*seq_len, z_dim) -> (bs, seq_len, z_dim)
        z_t = z_t.reshape(bs, seq_len, -1)

        # Process sequences using context model
        # c: (bs, seq_len, c_dim)
        # h: (bs, c_dim)
        c, h = self.gar(z_t)

        # Predict future codes produced by genc using context at time t
        # c_t: (bs, c_dim) at index t
        c_t = c[:, self.t - 1, :]

        return z_t, c_t
