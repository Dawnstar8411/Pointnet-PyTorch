import torch
import torch.nn as nn
import torch.nn.functional as F

from .transform_nets import Transform_net


class PointNet_cls(nn.Module):
    def __init__(self, K=40, input_transform=True, feature_transform=True):
        super(PointNet_cls, self).__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn = Transform_net(k=3)

        self.conv1_1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1_2 = nn.BatchNorm1d(64)

        if self.feature_transform:
            self.fstn = Transform_net(k=64)

        self.conv2_1 = torch.nn.Conv1d(64, 64, 1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn2_3 = nn.BatchNorm1d(1024)


        self.fc3_1 = nn.Linear(1024, 512)
        self.bn3_1 = nn.BatchNorm1d(512)
        self.dropout3_1 = nn.Dropout(p=0.3)
        self.fc3_2 = nn.Linear(512, 256)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.dropout3_2 = nn.Dropout(p=0.3)
        self.fc3_3 = nn.Linear(256, K)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.input_transform:
            trans = self.stn(x)

            x = x.transpose(2, 1)  # (batchsize, n_pts, 3)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)  # (batchsize, 3, n_pts)
        else:
            trans = None

        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn3_1(self.fc3_1(x)))
        x = self.dropout3_1(x)
        x = self.relu(self.bn3_2(self.fc3_2(x)))
        x = self.dropout3_2(x)
        x = self.fc3_3(x)
        return x, trans_feat

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
