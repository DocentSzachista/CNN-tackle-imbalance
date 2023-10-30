import torchvision
import torchvision.transforms as transforms
import pandas as pd
from utils import IMAGE_PREPROCESSING
from torch.utils.data import DataLoader
import numpy as np

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

SCENARIO_1 = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SCENARIO_2 = [1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]
SCENARIO_3 = [1, 0.01, 0.8, 1, 1, 1, 0.1, 1, 1, 1]

SAVE_PATH = "scenarios"


def download_cifar_set(
        transform_img: transforms.Compose, save: bool, savepath: str, is_train: bool, batch: int) -> DataLoader:
    trainset = torchvision.datasets.CIFAR10(
        root=savepath, train=is_train, download=save, transform=transform_img)
    return DataLoader(
        trainset, batch_size=batch, shuffle=is_train, num_workers=2)


def reduce_dataset_over_strategy(
        data_loader: DataLoader,
        strategy: list,
        filename: str
):
    """Change dataset contents according to strategy and save it to numpy file.

    :param data_loader: data loader of dataset in which should be data and targets changed.
    :type data_loader: DataLoader
    :param strategy: list containing ratio of data should remain in given class.
    :param filename: path where the files are stored with proper scenario name.
    :type filename: str
    :return: returns targets and images lists that were made smaller depending on strategy.
    :rtype: tuple(list, list[ndarray])

    """
    data_ext = [

    ]
    for _, targets in data_loader:
        targets_list = targets.numpy()
        data_ext.extend(targets_list)

    df = pd.DataFrame(data_ext, columns=['original_label'])
    classes_indexes = {k: df.index[df['original_label'] == k].to_list()
                       for k in range(0, len(strategy)) if 1 - strategy[k] != 0}

    images = data_loader.dataset.data
    targets = np.array(data_loader.dataset.targets)

    indexes_to_remove = []
    for k, indexes in classes_indexes.items():
        step = int(100 / (100 * strategy[k]))
        print(step)
        indexes_to_remove += indexes[::step]
    print(len(indexes_to_remove))
    images = np.delete(images, indexes_to_remove, axis=0)
    targets = np.delete(targets, indexes_to_remove, axis=0)

    np.save(
        f"{SAVE_PATH}/{filename}-images", images
    )
    np.save(
        f"./{SAVE_PATH}/{filename}-targets", images
    )
    return targets, images


def load_subset_dataset(data_loader: DataLoader, path: str):
    """ Load chosen scenario from numpy files and store them in dataset.

        :param data_loader: data loader of dataset in which should be data and targets changed.
        :type data_loader: DataLoader
        :param path: path where the files are stored with proper scenario name.
        :type path: str
        :return: None, it only changes contents of dataloader

    """
    images = np.load(f"{path}-images.npy")
    targets = np.load(f"{path}-targets.npy")
    data_loader.dataset.targets = targets
    data_loader.dataset.data = images


if __name__ == "__main__":

    dataset_2 = download_cifar_set(IMAGE_PREPROCESSING, True, "./data", True, 50000)
    reduce_dataset_over_strategy(dataset_2, SCENARIO_1, "strategy_one_class")
    reduce_dataset_over_strategy(dataset_2, SCENARIO_3, "strategy_two_class")
    reduce_dataset_over_strategy(dataset_2, SCENARIO_2, "strategy_many_classes")
