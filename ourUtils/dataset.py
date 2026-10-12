from PIL import Image
import numpy as np
import torch
import torchvision
import os
from ourUtils.funct_band_aids import setup_directory
from torchvision.utils import save_image


def shuffle_dataset(data, labels):
    shuffled_indices = np.random.permutation(len(data))
    shuffled_x = data[shuffled_indices, :]
    shuffled_y = labels[shuffled_indices]
    return shuffled_x, shuffled_y


def save_data_set(data: torch.tensor, labels: torch.tensor, save_dir: str):
    """Simple data set saving utility using torchvision.utils.save_image
    """
    setup_directory(save_dir)
    for idx, image in enumerate(data):
        padded_index = str(idx).rjust(6, '0')
        fname = '{}.png'.format(padded_index)

        label_dir = "{}_{}".format(*labels[idx].tolist())
        img_dir = os.path.join(save_dir, label_dir)
        if not os.path.exists(img_dir):
            setup_directory(img_dir)
        fpath = os.path.join(img_dir, fname)
        save_image(image, fpath)


class MnistVSDataset(torch.utils.data.Dataset):
    """Two mnist images side by side dataset. 
    TODO: Could save paired images to disk to allow for batch loading. This implementation holds everything in memory.
    Args:
        mnist_dataset (string): torchvision.datasets.MNIST object
        left_labels (list): specify which digits should be used on the left
        right_labels (list): specify which digits should be used on the right

        transform (callable, optional): Optional transform to be applied
            on a sample.

    """

    def __init__(self, mnist_dataset: torchvision.datasets.MNIST, left_labels: list, right_labels: list, transform=None, upsample: int = None):

        self.left_labels = left_labels
        self.right_labels = right_labels
        self.upsample = upsample

        self.transform = transform
        self.data, self.labels = self._pair_data(mnist_dataset)
        self.mean = self.data.mean().item()/255
        self.std = self.data.std().item()/255

    @staticmethod
    def _match_samples(left_data, left_labels, right_data, right_labels):
        # Match number of samples on the right with number of samples on the left
        nr_right = right_data.shape[0]
        nr_left = left_data.shape[0]
        print("before matching:")
        print("nr_right: ", nr_right)
        print("nr_left: ", nr_left)

        if nr_right < nr_left:
            diff = nr_left - nr_right
            indices = np.random.choice(nr_right, diff, replace=False)
            right_data = torch.cat([right_data, right_data[indices]], dim=0)
            right_labels = torch.cat(
                [right_labels.unsqueeze(1), right_labels[indices].unsqueeze(1)]).squeeze(1)
        else:
            diff = nr_right - nr_left
            indices = np.random.choice(nr_left, diff, replace=False)
            left_data = torch.cat([left_data, left_data[indices]], dim=0)
            left_labels = torch.cat(
                [left_labels.reshape(-1, 1), left_labels[indices].reshape(-1, 1)])

        return left_data, left_labels, right_data, right_labels

    def _upsample(self, left_data, left_labels, right_data, right_labels):
        nr_right = right_data.shape[0]
        nr_left = left_data.shape[0]

        diff = self.upsample - nr_right
        indices = np.random.choice(nr_right, diff, replace=True)
        right_data = torch.cat([right_data, right_data[indices]], dim=0)
        right_labels = torch.cat(
            [right_labels.unsqueeze(1), right_labels[indices].unsqueeze(1)]).squeeze(1)

        diff = self.upsample - nr_left
        indices = np.random.choice(nr_left, diff, replace=True)
        left_data = torch.cat([left_data, left_data[indices]], dim=0)
        left_labels = torch.cat(
            [left_labels.unsqueeze(1), left_labels[indices].unsqueeze(1)]).squeeze(1)

        return left_data, left_labels, right_data, right_labels

    def _pair_data(self, data_set):
        left_indices = torch.zeros_like(data_set.targets, dtype=torch.uint8)
        for l in self.left_labels:
            left_indices += data_set.targets == l
        left_data = data_set.data[left_indices].float()
        left_labels = data_set.targets[left_indices]

        right_indices = torch.zeros_like(data_set.targets, dtype=torch.uint8)
        for l in self.right_labels:
            right_indices += data_set.targets == l
        right_data = data_set.data[right_indices].float()
        right_labels = data_set.targets[right_indices]

        if right_data.shape[0] != left_data.shape[0]:
            left_data, left_labels, right_data, right_labels = self._match_samples(left_data,
                                                                                   left_labels,
                                                                                   right_data,
                                                                                   right_labels)

        if self.upsample is not None:
            left_data, left_labels, right_data, right_labels = self._upsample(left_data,
                                                                              left_labels,
                                                                              right_data,
                                                                              right_labels)

        left_data, left_labels = shuffle_dataset(left_data, left_labels)
        right_data, right_labels = shuffle_dataset(right_data, right_labels)

        nr_right = right_data.shape[0]
        nr_left = left_data.shape[0]
        print("after matching:")
        print("nr_right: ", nr_right)
        print("nr_left: ", nr_left)

        concat_data = torch.cat([left_data, right_data], 2)
        concat_labels = torch.stack([left_labels, right_labels], dim=1)
        return concat_data, concat_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform is not None:
            image = image.numpy()
            image = self.transform(image)

        return image, self.labels[idx]
