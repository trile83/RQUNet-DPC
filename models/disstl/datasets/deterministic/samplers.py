import copy
import random
import numpy as np
import torch

from disstl.datasets.deterministic.datasets import IndexedDataset, FineTuneDataset


class RandomSampler(torch.utils.data.Sampler):
    """
    Random shuffle on the sample indices of this dataset

    Accepts a random seed, and will return python's random module's random state
    back to what it was before this object was built. All randomization is done at init time
    """

    def __init__(self, data_source: IndexedDataset = None, seed: int = 5):
        self.data_source = data_source
        self.sample_indices = copy.deepcopy(self.data_source.sample_indices)
        pre_state = random.getstate()
        random.seed(seed)
        random.shuffle(self.sample_indices)
        random.setstate(pre_state)

    def __iter__(self):
        return iter(self.sample_indices)

    def __len__(self):
        return len(self.data_source)


class BalancedCategoricalSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, data_source: IndexedDataset = None, category: str = None, replacement: bool = True):
        self.data_source = data_source
        self.replacement = replacement

        category = f"{category}_id" if not category.endswith("_id") else category
        targets = np.array(data_source.sample_metadata_by_key[category])
        class_sample_count = np.array([(targets == t).sum() for t in np.unique(targets)])
        weights = 1.0 / class_sample_count
        sample_weights = np.array([weights[t] for t in targets])
        sample_weights = torch.from_numpy(sample_weights)
        super().__init__(sample_weights, len(sample_weights), replacement=replacement)


class FineTuneSampler(torch.utils.data.Sampler):
    """
    Used only for FineTuneDatasets, this sampler builds one of two types of sample index lists:
        1. labeled_samples = True: This option builds a list of sample indices that only point to
           data samples that have fine tune labels.
        2. unlabeled_samples = True: This option builds a list of sample indices that only point to
           data samples that have not been given
        fine-tuned labels.
        * Note: unlabeled and labeled cannot have the same value

    Usage: A good way to use this sampler is to obtain a list of 'training' samples (labeled_samples=True)
    and a list of 'to-be-pseudolabeled' samples (unlabeled_samples=True). The FineTune dataset has a method
    update_pl_labels() which will assign fine-tune labels to samples based on some criteria. Once samples
    have obtained fine tune labels then they will appear in the indices when using labeled_samples=True,
    and will not appear in the samples when using 'unlabeled_samples=True'
    """

    def __init__(
        self, data_source: FineTuneDataset = None, labeled_samples: bool = False, unlabeled_samples: bool = False
    ):
        assert labeled_samples is not unlabeled_samples
        self.data_source = data_source
        self.labeled_samples = labeled_samples
        self.unlabeled_samples = unlabeled_samples

    @property
    def sample_indices(self):
        if self.labeled_samples is True:
            return [
                sample_index
                for sample_index, ft_label in enumerate(self.data_source.ft_labels_by_index)
                if ft_label != self.data_source.sentinel_value
            ]
        elif self.unlabeled_samples is True:
            return [
                sample_index
                for sample_index, ft_label in enumerate(self.data_source.ft_labels_by_index)
                if ft_label == self.data_source.sentinel_value
            ]

    def __iter__(self):
        return iter(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)
