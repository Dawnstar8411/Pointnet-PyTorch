import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet_sem_seg(nn.Module):
    def __init__(self, num_cls=13):
        super(PointNet_sem_seg, self).__init__()
        self.num_cls = num_cls

        self.conv1_1 = torch.nn.Conv1d(9, 64, 1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_1 = torch.nn.Conv1d(64, 64, 1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn2_3 = nn.BatchNorm1d(1024)

        self.fc3_1 = nn.Linear(1024, 256)
        self.bn3_1 = nn.BatchNorm1d(256)
        self.fc3_2 = nn.Linear(256, 128)
        self.bn3_2 = nn.BatchNorm1d(128)

        self.conv4_1 = torch.nn.Conv1d(1152, 512, 1)
        self.bn4_1 = nn.BatchNorm1d(512)
        self.conv4_2 = torch.nn.Conv1d(512, 256, 1)
        self.bn4_2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.conv4_3 = torch.nn.Conv1d(256, self.num_cls, 1)

    def forward(self, x):
        n_pts = x.size()[2]

        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x = F.leaky_relu(self.bn1_2(self.conv1_2(x)))
        x = F.leaky_relu(self.bn2_1(self.conv2_1(x)))
        x = F.leaky_relu(self.bn2_2(self.conv2_2(x)))
        point_feat = F.leaky_relu(self.bn2_3(self.conv2_3(x)))

        x = torch.max(point_feat, 2)[0]
        x = x.view(-1, 1024)

        x = F.leaky_relu(self.bn3_1(self.fc3_1(x)))
        x = F.leaky_relu(self.bn3_2(self.fc3_2(x)))
        x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
        concat = torch.cat([x, point_feat], 1)

        x = F.leaky_relu(self.bn4_1(self.conv4_1(concat)))
        x = F.leaky_relu(self.bn4_2(self.conv4_2(x)))
        x = self.dropout(x)
        x = self.conv4_3(x)
        return x  # (batch_size,self.num_cls,n_pts)

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
