from typing import Optional, Tuple

import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    ''' Initialize ConvGRU cell '''
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros([B,self.hidden_size,*spatial_dim]).cuda()
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state], dim=1) #concat in C
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(torch.cat([input_tensor, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout=0.1):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, x, hidden_state=None):
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        # input: image sequences [B, T, C, H, W]
        current_layer_input = x 
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):
                cell_hidden = self.cell_list[idx](current_layer_input[:,t,:], cell_hidden)
                cell_hidden = self.dropout_layer(cell_hidden) # dropout in each time step
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        last_state_list = torch.stack(last_state_list, dim=1)

        return layer_output, last_state_list


class ConvGRU_1(nn.Module):
    """GRU with internal convolutional layers for 2D inputs/outputs"""

    def __init__(
        self, input_size: int, hidden_size: int, kernel_size: int = 3, num_layers: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvGRUCell(input_size if i == 0 else hidden_size, hidden_size, kernel_size))
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


if __name__ == '__main__':
    crnn = ConvGRU(input_size=10, hidden_size=20, kernel_size=3, num_layers=2)
    data = torch.randn(4, 5, 10, 6, 6) # [B, seq_len, C, H, W], temporal axis=1
    output, hn = crnn(data)
    import ipdb; ipdb.set_trace()
