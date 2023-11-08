import torch.nn as nn
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


def get_criterion_loss(class_ammounts: list | None = [1 for x in range(10)]) -> nn.CrossEntropyLoss:
    """ Retrieve CrossEntropyLoss with class weights setup.

        :param class_ammounts: number of speciments in each class. If none is
            passed then weights as one is set.
        :type class_ammounts: list[int]
        :return: CrossEntropyLoss object
    """
    weights = Tensor([1 / x for x in class_ammounts])

    return nn.CrossEntropyLoss(weight=weights)


def retrieve_weighed_dataloader(dataloader: DataLoader):
    """Make a WeightedRandomSampler from data.
       :param dataloader: Dataloader containing imbalanced dataset.
       :type dataloader: DataLoader
       :return: dataloader with prepared WeightedRandomSampler
    """
    targets = dataloader.dataset.targets
    class_sample_count = np.array(
        [len(np.where(targets == t)) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    dataloader.sampler = sampler
