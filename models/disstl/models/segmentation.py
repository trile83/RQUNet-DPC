import collections
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from disstl.models.unet_1 import UNet_VAE_RQ_scheme1_encoder, UNet_VAE_Stacked
from disstl.models.cpc import CPC
from disstl.models.encoder import Autoregressor2d, Encoder


class FPN(nn.Module):
    """ResNet Feature Pyramid Network (FPN)

    args:
        encoder:  disstl.models.Encoder module
        backbone: resnet backbone (e.g. resnet18, resnet50, etc.)
        trainable_layers: set this between [0, 5] to make the last n resnet layers trainable
    """

    def __init__(self, encoder: nn.Module, backbone: str, out_channels: int, trainable_layers: int = 3):
        super().__init__()
        self.fpn = resnet_fpn_backbone(backbone, pretrained=True, trainable_layers=trainable_layers)
        self.fpn.load_state_dict(encoder.model.state_dict(), strict=False)
        self.head = encoder.head
        in_channels = self.fpn.out_channels * len(self.fpn.body.return_layers)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        features = self.fpn(self.head(x))
        del features["pool"]

        # Upsample feature maps
        for k in features:
            features[k] = F.interpolate(features[k], size=(h, w), mode="bilinear", align_corners=True)

        # concatenate on channel dim
        z = torch.cat([v for v in features.values()], dim=1)

        # reduce channel dim
        z = self.conv1x1(z)

        return z


class SegmentationHead(nn.Sequential):
    """
    Convolutional Segmentation head

    args:
        input_dim:  input layer size
        hidden_dim: hidden layer size
        output_dim: output layer size
        kernel_size: conv kernel size
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size, padding=padding),
        )



class SegmentationHead_new(nn.Sequential): 
    """
    Convolutional Segmentation head

    args:
        input_dim:  input layer size
        hidden_dim: hidden layer size
        output_dim: output layer size
        kernel_size: conv kernel size
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            # nn.Upsample(scale_factor=2),
            nn.Conv3d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, output_dim, kernel_size, padding=padding),
        )


class CPCMultiTemporalSegmentation(nn.Module):
    """
    Contrastive Predictive Coding + Segmentation head module

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        t:                      context vector used at step t for predicting future codes
        k:                      codes from z[t:t+k] are predicted using c_t
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
        cpc_state_dict_path:    path to file containing pretrained CPC state dict
        cpc_freeze_weights:     freeze CPC model weights if True
    """

    def __init__(
        self,
        genc_backbone: str,
        gar_hidden_dim: int,
        channels: int,
        t: int,
        k: int,
        mlp_hidden_dim: int,
        num_classes: int,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        cpc_state_dict_path: str = None,
        cpc_freeze_weights: bool = False,
    ):
        super().__init__()

        assert "resnet" in genc_backbone, "only resnet backbones are supported for now"
        self.cpc = CPC(
            genc_backbone=genc_backbone,
            gar_hidden_dim=gar_hidden_dim,
            channels=channels,
            t=t,
            k=k,
            pretrained=pretrained,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
        )
        if cpc_state_dict_path is not None:
            incoming_state_dict = torch.load(cpc_state_dict_path, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k.replace("model.", "", 1): v for k, v in incoming_state_dict.items()}
            self.cpc.load_state_dict(state_dict, strict=False)
            self.cpc.freeze(cpc_freeze_weights)

        self.encoder = FPN(
            encoder=self.cpc.genc,
            backbone=genc_backbone,
            out_channels=gar_hidden_dim,
            trainable_layers=0 if cpc_freeze_weights else 5,
        )
        self.gar = Autoregressor2d(input_dim=gar_hidden_dim, hidden_dim=gar_hidden_dim)
        self.head = SegmentationHead(input_dim=gar_hidden_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes)
        self.result_tuple = collections.namedtuple("cpc_multitemporal_seg_result", ["cpc_result", "y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes, h, w)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape

        # Returning cpc output for consistency
        with torch.no_grad():
            output = self.cpc(x)

        # Extract 2D feature maps
        x = rearrange(x, "b t c h w -> (b t) c h w")
        z_t = self.encoder(x)
        z_t = rearrange(z_t, "(b t) c h w -> b t c h w", t=seq_len)

        # Conv RNN
        c, h = self.gar(z_t)

        # Pad front of t dim with zero for initial c_t-1
        c_t_1 = c[:, :-1, ...]
        c_t_1 = F.pad(c_t_1, pad=(0, 0, 0, 0, 0, 0, 1, 0), mode="constant", value=0)

        # Concat z_t and prior c_t on channel dim
        inputs = rearrange([c_t_1, z_t], "f b t c h w -> b t (f c) h w")

        inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
        y_pred = self.head(inputs)
        y_pred = rearrange(y_pred, "(b t) c h w -> b t c h w", t=seq_len)

        return self.result_tuple(cpc_result=output, y_pred=y_pred)


class BiDirectionalTemporalSegmentation(nn.Module):
    """
    Bi-directional temporal segmentation using forward and backward autoregressors

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
        freeze_weights:         freeze encoder model weights if True
    """

    def __init__(
        self,
        genc_backbone: str,
        encoder_dim: int,
        gar_dim: int,
        channels: int,
        mlp_hidden_dim: int,
        num_classes: int,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        freeze_weights: bool = False,
        encoder_arch: str = "cpc_fpn",
    ):
        super().__init__()

        if encoder_arch == "cpc_fpn":
            assert "resnet" in genc_backbone, "only resnet backbones are supported for now"

            self.genc = Encoder(
                backbone=genc_backbone,
                channels=channels,
                pretrained=pretrained,
                image_size=image_size,
                patch_size=patch_size,
                depth=depth,
            )

            self.encoder = FPN(
                encoder=self.genc,
                backbone=genc_backbone,
                out_channels=encoder_dim,
                trainable_layers=0 if freeze_weights else 5,
            )
        elif encoder_arch == "unet":
            self.encoder = smp.Unet(
                encoder_name=genc_backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=channels,
                classes=encoder_dim,
            )
        else:
            raise NotImplementedError(f"Encoder architecture {encoder_arch} not implemented.")

        self.gar_forward = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.gar_reverse = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.head = SegmentationHead(
            input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        )
        self.result_tuple = collections.namedtuple("bidirtemporal_seg_result", ["y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes, h, w)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape
        print ("input shape: ", bs, seq_len, c, h, w )

        # Extract 2D feature maps
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        z_t = self.encoder(x)
        z_t = rearrange(z_t, "(b t) c h w -> b t c h w", t=seq_len)


        # Conv RNN
        c_f, _ = self.gar_forward(z_t)
        c_r, _ = self.gar_reverse(z_t.flip(dims=[1]))


        # Concatenate c_f, z_t, and c_r on the embedding dimension
        inputs = torch.cat([c_f, z_t, c_r], axis=2)

        inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
        y_pred = self.head(inputs)

        y_pred = rearrange(y_pred, "(b t) c h w -> b t c h w", t=seq_len)
        return self.result_tuple(y_pred=y_pred)



class BiDirectionalTemporalSegmentation_new(nn.Module):
    """
    Bi-directional temporal segmentation using forward and backward autoregressors

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
        freeze_weights:         freeze encoder model weights if True
    """

    def __init__(
        self,
        genc_backbone: str,
        encoder_dim: int,
        gar_dim: int,
        channels: int,
        mlp_hidden_dim: int,
        num_classes: int,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        freeze_weights: bool = False,
        encoder_arch: str = "cpc_fpn",
    ):
        super().__init__()

        if encoder_arch == "cpc_fpn":
            assert "resnet" in genc_backbone, "only resnet backbones are supported for now"

            self.genc = Encoder(
                backbone=genc_backbone,
                channels=channels,
                pretrained=pretrained,
                image_size=image_size,
                patch_size=patch_size,
                depth=depth,
            )

            self.encoder = FPN(
                encoder=self.genc,
                backbone=genc_backbone,
                out_channels=encoder_dim,
                trainable_layers=0 if freeze_weights else 5,
            )
        elif encoder_arch == "unet":
            self.encoder = smp.Unet(
                encoder_name=genc_backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=channels,
                classes=encoder_dim,
            )
        else:
            raise NotImplementedError(f"Encoder architecture {encoder_arch} not implemented.")

        self.gar_forward = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.gar_reverse = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        # self.head = SegmentationHead(
        #     input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        # )
        self.head = SegmentationHead_new(
            input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        )
        self.result_tuple = collections.namedtuple("bidirtemporal_seg_result", ["y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes, h, w)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape
        # print ("input shape: ", bs, seq_len, c, h, w )

        # Extract 2D feature maps
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        z_t = self.encoder(x)
        z_t = rearrange(z_t, "(b t) c h w -> b t c h w", t=seq_len)

        
        # Conv RNN
        c_f, _ = self.gar_forward(z_t)
        c_r, _ = self.gar_reverse(z_t.flip(dims=[1]))

        # Concatenate c_f, z_t, and c_r on the embedding dimension
        inputs = torch.cat([c_f, z_t, c_r], axis=2)



        inputs = rearrange(inputs, "b t c h w -> b c t h w")
        y_pred = self.head(inputs)
        y_pred = rearrange(y_pred, "b c t h w -> b t c h w")
        # inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
        # y_pred = self.head(inputs)
        # y_pred = rearrange(y_pred, "(b t) c h w -> b t c h w", t=seq_len)
        # print ("seg head y_pred shape: ", y_pred.shape)
        return self.result_tuple(y_pred=y_pred)



class BiDirectionalTemporalSegmentation_RQUnet(nn.Module):
    """
    Bi-directional temporal segmentation using forward and backward autoregressors

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
        freeze_weights:         freeze encoder model weights if True
    """

    def __init__(
        self,
        genc_backbone: str,
        encoder_dim: int,
        gar_dim: int,
        channels: int,
        mlp_hidden_dim: int,
        num_classes: int,
        device,
        model_weight: str,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        freeze_weights: bool = False,
        encoder_arch: str = "cpc_fpn",
    ):
        super().__init__()

        # if encoder_arch == "cpc_fpn":
        #     assert "resnet" in genc_backbone, "only resnet backbones are supported for now"

        #     self.genc = Encoder(
        #         backbone=genc_backbone,
        #         channels=channels,
        #         pretrained=pretrained,
        #         image_size=image_size,
        #         patch_size=patch_size,
        #         depth=depth,
        #     )

        #     self.encoder = FPN(
        #         encoder=self.genc,
        #         backbone=genc_backbone,
        #         out_channels=encoder_dim,
        #         trainable_layers=0 if freeze_weights else 5,
        #     )
        # elif encoder_arch == "unet":
        #     self.encoder = smp.Unet(
        #         encoder_name=genc_backbone,
        #         encoder_weights="imagenet" if pretrained else None,
        #         in_channels=channels,
        #         classes=encoder_dim,
        #     )
        # else:
        #     raise NotImplementedError(f"Encoder architecture {encoder_arch} not implemented.")
        self.encoder = UNet_VAE_RQ_scheme1_encoder(encoder_dim, False, 0.1, channels, depth=3)
        if model_weight is not None:
            incoming_state_dict = torch.load(model_weight, map_location=device)

            state_dict = {str("encoder."+k): v for k, v in incoming_state_dict.items()}
            for param in state_dict:
                print(param, "\t")
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder.freeze(freeze_weights)

        self.gar_forward = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.gar_reverse = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        # self.head = SegmentationHead(
        #     input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        # )
        self.head = SegmentationHead_new(
            input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        )
        self.result_tuple = collections.namedtuple("bidirtemporal_seg_result", ["y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes, h, w)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape
        # print ("input shape: ", bs, seq_len, c, h, w )

        # Extract 2D feature maps
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        z_t = self.encoder(x)
        z_t = rearrange(z_t, "(b t) c h w -> b t c h w", t=seq_len)
        
        #print('z_t shape: ', z_t.shape)

        # Conv RNN
        c_f, _ = self.gar_forward(z_t)
        c_r, _ = self.gar_reverse(z_t.flip(dims=[1]))


        # Concatenate c_f, z_t, and c_r on the embedding dimension
        inputs = torch.cat([c_f, z_t, c_r], axis=2)


        inputs = rearrange(inputs, "b t c h w -> b c t h w")
        y_pred = self.head(inputs)
        y_pred = rearrange(y_pred, "b c t h w -> b t c h w")
        # inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
        # y_pred = self.head(inputs)
        # y_pred = rearrange(y_pred, "(b t) c h w -> b t c h w", t=seq_len)
        # print ("seg head y_pred shape: ", y_pred.shape)
        return self.result_tuple(y_pred=y_pred)
        
        
        
class BiDirectionalTemporalSegmentation_StackedRQUnet(nn.Module):
    """
    Bi-directional temporal segmentation using forward and backward autoregressors

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
        freeze_weights:         freeze encoder model weights if True
    """

    def __init__(
        self,
        genc_backbone: str,
        encoder_dim: int,
        gar_dim: int,
        channels: int,
        mlp_hidden_dim: int,
        num_classes: int,
        device,
        model_weight: str,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        freeze_weights: bool = False,
        encoder_arch: str = "cpc_fpn",
    ):
        super().__init__()

        self.encoder = UNet_VAE_Stacked(channels, False, 0.05, device, model_weight, channels)
        self.gar_forward = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.gar_reverse = Autoregressor2d(input_dim=encoder_dim, hidden_dim=gar_dim)
        self.head = SegmentationHead_new(
            input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes
        )
        self.result_tuple = collections.namedtuple("bidirtemporal_seg_result", ["y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes, h, w)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape
        # print ("input shape: ", bs, seq_len, c, h, w )

        # Extract 2D feature maps
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        z_t = self.encoder(x)
        z_t = rearrange(z_t, "(b t) c h w -> b t c h w", t=seq_len)
        
        # print('z_t shape: ', z_t.shape)

        # Conv RNN
        c_f, _ = self.gar_forward(z_t)
        c_r, _ = self.gar_reverse(z_t.flip(dims=[1]))

        # Concatenate c_f, z_t, and c_r on the embedding dimension
        inputs = torch.cat([c_f, z_t, c_r], axis=2)

        inputs = rearrange(inputs, "b t c h w -> b c t h w")
        y_pred = self.head(inputs)
        y_pred = rearrange(y_pred, "b c t h w -> b t c h w")

        return self.result_tuple(y_pred=y_pred)


