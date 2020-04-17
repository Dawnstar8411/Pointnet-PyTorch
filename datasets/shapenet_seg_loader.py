import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    def __init__(self, data_path, npoints=2048, transform=None, train=True):
        self.npoints = npoints
        self.transform = transform
        self.train = train

        if self.train:
            self.list_path = data_path / 'train_hdf5_file_list.txt'
        else:
            self.list_path = data_path / 'val_hdf5_file_list.txt'

        self.fns = [line.strip() for line in open(self.list_path, "r")]

        self.points = []
        self.label = []
        self.seg = []

        for i in np.arange(len(self.fns)):
            h5_filename = self.fns[i]
            f = h5py.File(h5_filename)
            self.points.append(f['data'][:])
            self.label.append(f['label'][:])
            self.seg.append(f['pid'][:])
        self.points = np.concatenate(self.points, axis=0)  # (num_samples, num_pts, 3)
        self.label = np.concatenate(self.label, axis=0)  # (num_samples,1)
        self.seg = np.concatename(self.label, axis=0)  # (num_samples, num_pts)

    def __getitem__(self, index):
        points = self.points[index]
        label = self.label[index]
        seg = self.seg[index]
        label_ont_hot = np.zeros(1, 16)
        label_ont_hot[0, label[0]] = 1
        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points), torch.from_numpy(label), torch.from_numpy(seg), torch.from_numpy(label_ont_hot)

    def __len__(self):
        return len(self.points)
