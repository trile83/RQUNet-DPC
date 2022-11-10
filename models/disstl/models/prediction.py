import collections
from typing import Dict, Optional

import torch
import torch.nn as nn

from disstl.models.classifier import MLP
from disstl.models.cpc import CPC


class CPCPredictor(nn.Module):
    """
    Contrastive Predictive Coding + MLP Activity Prediction

    args:
        genc_backbone:          valid model in torchvision.models
        gar_hidden_dim:         hidden_dim (also output_dim due to n_layers=1) of gar
        t:                      context vector used at step t for predicting future codes
        k:                      codes from z[t:t+k] are predicted using c_t
        channels:               input image channels
        mlp_hidden_dim:         hidden dim of mlp classifier
        num_years:              output dim of year head
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
        num_years: int,
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

        # Phase embedding
        self.phase_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=gar_hidden_dim)

        # Predictor
        self.num_years = num_years
        self.mlp_year = MLP(input_dim=gar_hidden_dim * 3, hidden_dim=mlp_hidden_dim, output_dim=num_years)
        self.mlp_month = MLP(input_dim=gar_hidden_dim * 3, hidden_dim=mlp_hidden_dim, output_dim=12)
        self.mlp_week = MLP(input_dim=gar_hidden_dim * 3, hidden_dim=mlp_hidden_dim, output_dim=4)
        self.result_tuple = collections.namedtuple(
            "cpc_predictor_result", ["cpc_result", "y_pred_year", "y_pred_month", "y_pred_week"]
        )

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        args:
            x: (batch_size, seq_len, c, h, w)
            phase: (batch_size, )

        returns:
            dict(
                y_pred_year: (batch_size, num_years)
                y_pred_month: (batch_size, 12)
                y_pred_week: (batch_size, 4)
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

        phase_embedding = self.phase_embedding(phase)  # (bs, c_dim)

        mlp_inputs = torch.cat([c_t, h, phase_embedding], dim=-1)  # (bs, c_dim * 3)

        # Predict phase end date from context vector, hidden state, and phase embedding
        y_pred_year = self.mlp_year(mlp_inputs)
        y_pred_month = self.mlp_month(mlp_inputs)
        y_pred_week = self.mlp_week(mlp_inputs)

        return self.result_tuple(
            cpc_result=output, y_pred_year=y_pred_year, y_pred_month=y_pred_month, y_pred_week=y_pred_week
        )
