from torch.utils import data
# from skimage import io

import imageio.v2 as io

import numpy as np

import torch

import glob
import os


class Dataset(data.Dataset):

    def __init__(self, path, is_train=False, transform=None, **params):
        super(Dataset, self).__init__()
        self.images_a = []
        self.images_b = []
        self.labels = []
        self.is_train = is_train
        self.transform = transform

        self.dataset_info = params.get('dataset_info', {'a': 'A', 'b': 'B', 'label': 'label', 'ext': 'png'})
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # _image_a = io.imread(self.images_a[idx], plugin='tifffile').astype(np.float32)
        # _image_b = io.imread(self.images_b[idx], plugin='tifffile').astype(np.float32)
        # _label = io.imread(self.labels[idx], plugin='tifffile')
        _image_a = io.imread(self.images_a[idx]).astype(np.float32)
        _image_b = io.imread(self.images_b[idx]).astype(np.float32)
        _label = io.imread(self.labels[idx])

        if self.transform:
            _image_a, _image_b, _label = self.transform((_image_a, _image_b, _label))

        return _image_a, _image_b, _label

    def _load_data(self, path):
        _ext = self.dataset_info['ext']
        self.images_a = glob.glob(os.path.join(path, self.dataset_info['a'], f'*.{_ext}'))
        self.images_b = glob.glob(os.path.join(path, self.dataset_info['b'], f'*.{_ext}'))
        self.labels = glob.glob(os.path.join(path, self.dataset_info['label'], f'*.{_ext}'))

        self.images_a = sorted(self.images_a, key=os.path.basename)
        self.images_b = sorted(self.images_b, key=os.path.basename)
        self.labels = sorted(self.labels, key=os.path.basename)

        # self.images_a = self.images_a[:100]
        # self.images_b = self.images_b[:100]
        # self.labels = self.labels[:100]
        assert len(self.images_a) == len(self.images_b) and len(self.images_a) == len(self.labels) and len(self.labels)
