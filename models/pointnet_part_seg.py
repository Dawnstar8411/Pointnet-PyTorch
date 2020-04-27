import torch
import torch.nn as nn
import torch.nn.functional as F

from .transform_nets import Transform_net


class PointNet_part_seg(nn.Module):
    def __init__(self, cls_num=16, part_num=50, input_transform=True, feature_transform=True):
        super(PointNet_part_seg, self).__init__()
        self.cls_num = cls_num
        self.part_num = part_num
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn = Transform_net(in_channels=3)

        self.conv1_1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.conv1_3 = torch.nn.Conv1d(128, 128, 1)
        self.bn1_3 = nn.BatchNorm1d(128)

        if self.feature_transform:
            self.fstn = Transform_net(in_channels=128)

        self.conv2_1 = torch.nn.Conv1d(128, 512, 1)
        self.bn2_1 = nn.BatchNorm1d(512)
        self.conv2_2 = torch.nn.Conv1d(512, 2048, 1)
        self.bn2_2 = nn.BatchNorm1d(2048)

        self.fc3_1 = nn.Linear(2048, 256)
        self.bn3_1 = nn.BatchNorm1d(256)
        self.fc3_2 = nn.Linear(256, 256)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.dropout3_2 = nn.Dropout(p=0.3)
        self.fc3_3 = nn.Linear(256, self.cls_num)

        self.conv4_1 = torch.nn.Conv1d(4944, 256, 1)
        self.bn4_1 = nn.BatchNorm1d(256)
        self.dropout4_1 = nn.Dropout(p=0.2)
        self.conv4_2 = torch.nn.Conv1d(256, 256, 1)
        self.bn4_2 = nn.BatchNorm1d(256)
        self.dropout4_2 = nn.Dropout(p=0.2)
        self.conv4_3 = torch.nn.Conv1d(256, 128, 1)
        self.bn4_3 = nn.BatchNorm1d(128)
        self.conv4_4 = torch.nn.Conv1d(128, self.part_num, 1)

    def forward(self, x, input_label):
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)  # (batchsize, n_pts, 3)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)  # (batchsize, 3, n_pts)
        else:
            trans = None

        out1 = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        out2 = F.leaky_relu(self.bn1_2(self.conv1_2(out1)))
        out3 = F.leaky_relu(self.bn1_3(self.conv1_3(out2)))
        feature_trans = out3
        if self.feature_transform:
            trans_feat = self.fstn(out3)
            feature_trans = feature_trans.transpose(2, 1)
            feature_trans = torch.bmm(feature_trans, trans_feat)
            feature_trans = feature_trans.transpose(2, 1)
        else:
            trans_feat = None

        out4 = F.leaky_relu(self.bn2_1(self.conv2_1(feature_trans)))
        out5 = F.leaky_relu(self.bn2_2(self.conv2_2(out4)))
        out_max = torch.max(out5, 2)[0]
        x = out_max.view(-1, 2048)

        x = F.leaky_relu(self.bn3_1(self.fc3_1(x)))
        x = F.leaky_relu(self.bn3_2(self.fc3_2(x)))
        x = self.dropout3_2(x)
        out_cls = self.fc3_3(x)

        out_max = torch.cat([out_max, input_label], 1)
        out_max = out_max.view(-1, 2048 + self.cls_num, 1).repeat(1, 1, n_pts)
        concat = torch.cat([out_max, out1, out2, out3, out4, out5], 1)

        x = F.leaky_relu(self.bn4_1(self.conv4_1(concat)))
        x = self.dropout4_1(x)
        x = F.leaky_relu(self.bn4_2(self.conv4_2(x)))
        x = self.dropout4_2(x)
        x = F.leaky_relu(self.bn4_3(self.conv4_3(x)))
        out_seg = self.conv4_4(x)

        return out_cls, out_seg, trans_feat

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
