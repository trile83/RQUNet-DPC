import collections
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from disstl.models.cpc import CPC
from disstl.models.encoder import Encoder
from disstl.models.grl import GradientReversalModule


class FourLayerMLP(nn.Module):
    """
    Multilayer Perceptron Classifier

    args:
        input_dim:  input layer size
        hidden_dim: hidden layer size
        output_dim: output layer size
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, input_dim)

        returns:
            y: (batch_size, hidden_dim)
        """
        return self.model(x)


class MLP(nn.Module):
    """
    Multilayer Perceptron Classifier

    args:
        input_dim:  input layer size
        hidden_dim: hidden layer size
        output_dim: output layer size
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, input_dim)

        returns:
            y: (batch_size, hidden_dim)
        """
        return self.model(x)


class CPCClassifier(nn.Module):
    """
    Contrastive Predictive Coding + MLP Classifier head module

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
        mlp: nn.Module = MLP,
        mlp_hidden_dim: int = None,
        mlp_dropout: float = None,
        num_classes: int = None,
        pretrained: bool = False,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        cpc_state_dict_path: str = None,
        cpc_freeze_weights: bool = False,
    ):
        super().__init__()
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

        # Classifier
        self.mlp = mlp(
            input_dim=gar_hidden_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=num_classes, dropout=mlp_dropout
        )
        self.backbone_parameters = [self.cpc.parameters()]
        self.head_parameters = self.mlp.parameters()
        self.result_tuple = collections.namedtuple("cpc_classifier_result", ["cpc_result", "y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape

        output = self.cpc(x)
        c_t, h = output.c[:, -1, :], output.h

        # Predict class from context vector
        y_pred = self.mlp(torch.cat([c_t, h], dim=-1))

        return self.result_tuple(cpc_result=output, y_pred=y_pred)


class CPCGRLClassifier(CPCClassifier):
    """
    Contrastive Predictive Coding + Gradient-Reversed MLP Classifier head module

    This class only prepends an nn.Module to the existing classifier of its parent class, all other nn.Module methods
    are inherited from its parent

    args:
        args of CPCClassifier, plus:
        grad_weight:             float, scales the reversed portion of the gradient of the classifiation loss
    """

    def __init__(
        self,
        genc_backbone: str,
        gar_hidden_dim: int,
        channels: int,
        t: int,
        k: int,
        mlp: nn.Module = MLP,
        mlp_hidden_dim: int = 64,
        mlp_dropout: float = 0.2,
        num_classes: int = None,
        pretrained: bool = False,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        cpc_state_dict_path: str = None,
        cpc_freeze_weights: bool = False,
        grad_weight: float = 1.0,
    ):
        super().__init__(
            genc_backbone=genc_backbone,
            gar_hidden_dim=gar_hidden_dim,
            channels=channels,
            t=t,
            k=k,
            num_classes=num_classes,
            mlp=mlp,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
            cpc_state_dict_path=cpc_state_dict_path,
            cpc_freeze_weights=cpc_freeze_weights,
        )
        # prepent Gradient Reversal module to the classifier of CPCClassifier
        self.mlp = nn.Sequential(GradientReversalModule(grad_weight=grad_weight), self.mlp)
        self.backbone_parameters = [self.cpc.parameters()]
        self.head_parameters = self.mlp.parameters()


class BaselineClassifier(nn.Module):
    """
    Contrastive Predictive Coding + MLP Classifier head module

    args:
        genc_backbone:          valid model in torchvision.models
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_classes:            output dim of last layer for classification
        pretrained:             use pretrained weights or not
        image_size:             input image size (vit only)
        patch_size:             size of image patches (vit only)
        depth:                  num transformer layers (vit only)
    """

    def __init__(
        self,
        genc_backbone: str,
        channels: int,
        mlp_hidden_dim: int,
        num_classes: int,
        pretrained: bool = False,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        super().__init__()
        self.genc = Encoder(
            backbone=genc_backbone,
            channels=channels,
            pretrained=pretrained,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
        )

        # Classifier
        self.mlp = MLP(input_dim=self.genc.emb_dim, hidden_dim=mlp_hidden_dim, output_dim=num_classes)
        self.backbone_parameters = [self.genc.parameters()]
        self.head_parameters = self.mlp.parameters()
        self.result_tuple = collections.namedtuple("baseline_classifier_result", ["y_pred", "z_t", "c_t"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, num_classes)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, z_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape

        # Encode all images. Combine all images in sequences to process at once
        # (bs, seq_len, c, h, w) -> (bs*seq_len, c, h, w)
        x = x.reshape(-1, c, h, w)
        z_t = self.genc(x)

        # Reshape back to sequences
        # (bs*seq_len, z_dim) -> (bs, seq_len, z_dim)
        z_t = z_t.reshape(bs, seq_len, -1)

        # Simple temporal fusion via average pooling the seq_len dim (dim=1)
        # (bs, seq_len, z_dim) -> (bs, z_dim)
        c_t = torch.mean(z_t, dim=1)

        # Predict class from context vector
        y_pred = self.mlp(c_t)

        return self.result_tuple(y_pred=y_pred, z_t=z_t, c_t=c_t)


class CPCMultiTemporalClassifier(nn.Module):
    """
    Contrastive Predictive Coding + MLP Classifier head at each time step module
    Based on MemDPC from Memory-augmented Dense Predictive Coding for Video Representation Learning
    Appendix B https://arxiv.org/abs/2008.01065

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
        mlp: nn.Module = MLP,
        mlp_hidden_dim: int = 64,
        mlp_dropout: float = 0.2,
        num_classes: int = None,
        pretrained: bool = False,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        depth: Optional[int] = None,
        cpc_state_dict_path: str = None,
        cpc_freeze_weights: bool = False,
    ):
        super().__init__()
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

        # Classifier
        self.mlp = mlp(
            input_dim=self.cpc.genc.emb_dim + gar_hidden_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=num_classes,
            dropout=mlp_dropout,
        )
        self.backbone_parameters = [self.cpc.parameters()]
        self.head_parameters = self.mlp.parameters()
        self.result_tuple = collections.namedtuple("cpc_multitemporalclassifier_result", ["cpc_result", "y_pred"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, c, h, w)

        returns:
            dict(
                y_pred: (batch_size, seq_len, num_classes)
                z: (batch_size, k, z_dim)
                z_hat: (batch_size, k, z_dim)
                z_t: (batch_size, seq_len, z_dim)
                c_t: (batch_size, c_dim)
                c: (batch_size, seq_len, c_dim)
                h: (batch_size, c_dim)
            )
        """
        bs, seq_len, c, h, w = x.shape

        output = self.cpc(x)
        c_t, z_t = output.c, output.z_t

        # inputs to time t are z_t and c_t-1
        c_t_1 = c_t[:, :-1, :]

        # Pad front of t dim with zero for initial c_t-1
        c_t_1 = F.pad(c_t_1, pad=(0, 0, 1, 0), mode="constant", value=0)

        # Predict class from image code and prior context vector
        inputs = torch.cat([c_t_1, z_t], dim=-1)

        inputs = inputs.reshape(bs * seq_len, -1)
        y_pred = self.mlp(inputs)
        y_pred = y_pred.reshape(bs, seq_len, -1)

        return self.result_tuple(cpc_result=output, y_pred=y_pred)
