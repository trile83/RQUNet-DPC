import torch
import numpy as np


class Sampler(torch.utils.data.Sampler):
    """
    Base Sampler returns 0, 1 indicators for negative, positive samples
    Note the sample is also balanced return pct_pos positive samples per batch
    """

    def __init__(self, data_source: torch.utils.data.Dataset, pct_pos: float):
        self.data_source = data_source
        self.num_pos = int(len(data_source) * pct_pos)

    def __iter__(self):
        indices = np.zeros(len(self.data_source), dtype=int)
        indices[: self.num_pos] = 1
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.data_source)
