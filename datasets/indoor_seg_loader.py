import h5py
import numpy as np
import torch
from path import Path
from torch.utils.data import Dataset


class Indoor3dDataset(Dataset):
    def __init__(self, data_path, npoints=4096, test_area=6, transform=None, train=True):
        self.npoints = npoints
        self.transform = transform
        self.train = train

        self.all_files_path = data_path / 'all_files.txt'
        self.room_file_path = data_path / 'room_filelist.txt'

        self.all_file_list = [data_path / line.rstrip() for line in open(self.all_file_path, "r")]

        self.room_file_list = [line.rstrip() for line in open(self.room_file_path, "r")]

        self.points = []
        self.label = []

        for i in np.arange(len(self.all_file_list)):
            h5_filename = self.all_file_list[i]
            f = h5py.File(h5_filename)
            self.points.append(f['data'][:])
            self.label.append(f['label'][:])
        self.points = np.concatenate(self.points, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        self.test_area = 'Area_' + str(test_area)

        self.train_idxs = []
        self.test_idxs = []

        for i, room_name in enumerate(self.room_file_list):
            if self.test_area in room_name:
                self.test_idxs.append(i)
            else:
                self.train_idxs.append(i)

        self.train_points = self.points[self.train_idxs, ...]
        self.train_label = self.label[self.train_idxs, ...]
        self.test_points = self.points[self.test_idxs, ...]
        self.test_label = self.label[self.test_idxs, ...]

    def __getitem__(self, index):
        if self.train:
            points = self.train_points[index]
            label = self.train_label[index]
        else:
            points = self.test_points[index]
            label = self.test_label[index]
        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points), torch.from_numpy(label)

    def __len__(self):
        if self.train:
            return len(self.train_points)
        else:
            return len(self.test_points)
