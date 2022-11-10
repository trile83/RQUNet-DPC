from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import _create_vision_transformer

from disstl.models.convrnn import ConvGRU


class MultiSpectralProjectionBlock(nn.Sequential):
    """Projects Multispectral Bands to N feature maps

    args:
        in_channels:  num input bands
        out_channels: num output feature maps
        kernel_size:  conv filter dimension
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = (kernel_size - 1) / 2  # same padding
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=int(padding)
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )


class Encoder(nn.Module):
    """
    Spatial Encoder (G_enc)

    args:
        backbone:   valid model in torchvision.models.resnet
        channels:   num input channels
        pretrained: initialize with ImageNet weights
        image_size: input image size (vit only)
        patch_size: size of image patches (vit only)
        depth:      num transformer layers (vit only)
    """

    def __init__(
        self,
        backbone: str,
        channels: int,
        pretrained: bool,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        super().__init__()

        if "resnet" in backbone:
            # Multispectral projection block
            if channels != 3:
                self.head = MultiSpectralProjectionBlock(in_channels=channels, out_channels=3, kernel_size=3)
            else:
                self.head = nn.Identity()

            resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
            self.emb_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.model = resnet

        elif "vit" in backbone:
            self.head = nn.Identity()
            params = dict(in_chans=channels, img_size=image_size, num_classes=0)
            if patch_size or depth:
                if patch_size:
                    params["patch_size"] = patch_size
                if depth:
                    params["depth"] = depth

                self.model = _create_vision_transformer(backbone, **params)
            else:
                params["pretrained"] = pretrained
                self.model = timm.create_model(backbone, **params)

            self.emb_dim = self.model.embed_dim
        else:
            raise ValueError(
                """Unknown backbone type. Must be one of torchvision.models.resnet
                or timm.models.vision_transformer or timm.models.vision_transformer_hybrid"""
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, channels, h, w)

        returns:
            z: (batch_size, z_dim)
        """
        return self.model(self.head(x)).squeeze()


class Autoregressor(nn.Module):
    """
    Autoregressive Encoder (G_ar)

    args:
        input_dim:  input sequence dim
        hidden_dim: hidden_dim (also output_dim due to n_layers=1) of gar
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, bidirectional=False, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, input_size)
            h0: (num_layers, batch_size, hidden_size)

        returns:
            c: (batch_size, seq_len, hidden_size)
            h: (num_layers, batch_size, hidden_size)
        """
        h0 = self.init_hidden(x.size(0)).to(x.device)
        c, h = self.rnn(x, h0)
        return c, h.squeeze(dim=0)

    def init_hidden(self, batch_size: int):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim)


class Autoregressor2d(nn.Module):
    """
    Convolutional Autoregressive Encoder (G_ar)

    args:
        input_dim:  input sequence dim
        hidden_dim: hidden_dim (output_dim==hidden_dim when n_layers=1) of gar
        kernel_size: kernel size of conv layers
        num_layers: number of repeated ConvGRU layers
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.rnn = ConvGRU(input_dim, hidden_dim, kernel_size, num_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, input_dim, h, w)

        returns:
            c: (batch_size, seq_len, hidden_dim, h, w)
            h: (num_layers, batch_size, hidden_dim, h, w)
        """
        h0 = None
        c, h = self.rnn(x, h0)
        return c, h.squeeze(dim=0)
