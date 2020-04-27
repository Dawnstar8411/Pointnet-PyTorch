import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    def __init__(self, data_path, n_pts=1024, transform=None, train=True):
        self.n_pts = n_pts
        self.transform = transform
        self.train = train
        if self.train:
            self.list_path = data_path / 'train_files.txt'
        else:
            self.list_path = data_path / 'test_files.txt'

        self.fns = [data_path / line.rstrip() for line in open(self.list_path, "r")]
        self.points = []
        self.label = []

        for i in np.arange(len(self.fns)):
            h5_filename = self.fns[i]
            f = h5py.File(h5_filename)
            self.points.append(f['data'][:])
            self.label.append(f['label'][:])
        self.points = np.concatenate(self.points, axis=0)  # (num_samples,n_pts,3)
        self.label = np.concatenate(self.label, axis=0)  # (num_samples,1)

    def __getitem__(self, index):
        points = self.points[index]
        label = self.label[index]
        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points), torch.from_numpy(label)

    def __len__(self):
        return len(self.points)
