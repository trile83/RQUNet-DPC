""" Modified from https://github.com/TengdaHan/MemDPC/blob/master/backbone/convrnn.py """
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """ConvGRU cell"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_tensor: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state is None:
            b, c, h, w = input_tensor.shape
            hidden_state = torch.zeros(b, self.hidden_dim, h, w).to(input_tensor.device)

        combined = torch.cat([input_tensor, hidden_state], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(torch.cat([input_tensor, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    """GRU with internal convolutional layers for 2D inputs/outputs"""

    def __init__(
        self, input_dim: int, hidden_dim: int, kernel_size: int = 3, num_layers: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size))
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        b, t, c, h, w = x.shape

        if hidden_state is None:
            h0 = [None] * len(self.layers)

        h_ts = []

        for i, layer in enumerate(self.layers):
            h_t = h0[i]
            c_ts = []
            for j in range(t):
                h_t = self.dropout(layer(x[:, j, ...], h_t))
                c_ts.append(h_t)

            x = c_ts = torch.stack(c_ts, dim=1)
            h_ts.append(h_t)

        h_ts = torch.stack(h_ts, dim=0)

        return c_ts, h_ts
