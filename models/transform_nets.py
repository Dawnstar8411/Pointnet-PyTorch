import torch
import torch.nn as nn
import torch.nn.functional as F


class Transform_net(nn.Module):
    def __init__(self, in_channels):
        super(Transform_net, self).__init__()
        self.conv1_1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.conv1_3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1_3 = nn.BatchNorm1d(1024)

        self.fc2_1 = nn.Linear(1024, 512)
        self.bn2_1 = nn.BatchNorm1d(512)
        self.fc2_2 = nn.Linear(512, 256)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.fc2_3 = nn.Linear(256, in_channels * in_channels)

        self.in_channels = in_channels

    def forward(self, x):
        # x:(batch_size, in_channels, n_pts)
        batchsize = x.size()[0]
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x = F.leaky_relu(self.bn1_2(self.conv1_2(x)))
        x = F.leaky_relu(self.bn1_3(self.conv1_3(x)))
        x = torch.max(x, 2)[0]  # 0:value tensor, 1:index tensor
        x = x.view(-1, 1024)

        x = F.leaky_relu(self.bn2_1(self.fc2_1(x)))
        x = F.leaky_relu(self.bn2_2(self.fc2_2(x)))
        x = self.fc2_3(x)

        # torch.repeat(n1,n2,n3): repeat n1 times in 0th channels, repeat n2 times in 1th channels ...
        iden = torch.eye(self.in_channels).view(1, self.in_channels * self.in_channels).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # broadcasting
        x = x.view(-1, self.in_channels, self.in_channels)
        return x
