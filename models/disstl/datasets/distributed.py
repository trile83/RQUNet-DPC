from typing import Optional

import torch


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """
    Modified from https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super().__init__(sampler.data_source, num_replicas, rank, shuffle)
        self.sampler = sampler

    def __iter__(self):
        indices = list(self.sampler)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return len(self.sampler)
