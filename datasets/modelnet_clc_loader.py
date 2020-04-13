import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    def __init__(self, data_path, npoints=2048, transform=None, train=True):
        self.npoints = npoints
        self.transform = transform
        self.train = train
        if self.train:
            self.list_path = data_path / 'train_files.txt'
        else:
            self.list_path = data_path / 'test_files.txt'

        with open(self.list_path, "r") as f:
            self.fns = [line.strip() for line in f.readlines()]
        self.data = []
        self.label = []

        for i in np.arange(len(self.fns)):
            h5_filename = self.fns[i]
            f = h5py.File(h5_filename)
            self.data.extend(f['data'][:])
            self.label.extend(f['label'][:])
        self.points = np.array(self.data)
        self.label = np.array(self.label)

    def __getitem__(self, index):
        points = self.points[index]
        label = self.label[index]
        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
