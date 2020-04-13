import torch
import torch.nn as nn
import torch.nn.functional as F

from .transform_nets import Transform_net


class PointNet_seg(nn.Module):
    def __init__(self, K=10, input_transform=True, feature_transform=True):
        super(PointNet_seg, self).__init__()
        self.K = K
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn = Transform_net(k=9)

        self.conv1_1 = torch.nn.conv1d(9, 64, 1)
        self.conv1_2 = torch.nn.conv1d(64,64,1)
        self.conv2_1 = torch.nn.conv1d(64,64,1)
        self.conv2_2 = torch.nn.Conv1d(64, 128, 1)
        self.conv2_3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.fstn = Transform_net(k=64)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)

        self.bn3_1 = nn.BatchNorm1d(512)
        self.bn3_2 = nn.BatchNorm1d(128)


        self.conv4 = torch.nn.Conv1d(2152, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, self.K, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout=nn.Dropout(p=0.3)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        n_pts = x.size()[2]
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
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.relu(self.bn3_1(self.fc1(x)))
        x = self.relu(self.bn3_2(self.fc2(x)))
        x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, pointfeat], 1)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.conv6(x)
        x = x.transpose(2, 1).contiguous()  # (batchsize,n_pts, self.k)
        x = F.log_softmax(x.view(-1, self.K), dim=-1)
        x = x.view(-1, n_pts, self.K)
        return x, trans_feat
