import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class CriticPointNet(nn.Module):
    def __init__(self, output_dim, action_dim, normal_channel=False, softmax=False, goal_conditioned=False):
        super(CriticPointNet, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_channel)
        self.goal_conditioned = goal_conditioned
        if self.goal_conditioned:
            self.fc1 = nn.Linear(2048+action_dim, 512)
        else:
            self.fc1 = nn.Linear(1024+action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.softmax = softmax

    def forward(self, obs_xyz, action, goal_xyz=None):
        x, trans, trans_feat = self.feat(obs_xyz)
        if self.goal_conditioned and goal_xyz is not None:
            goal_x, goal_trans, goal_trans_feat = self.feat(goal_xyz)
            x = T.cat([x, goal_x.detach()], dim=1)
        x = T.cat([x, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        value = self.fc3(x)

        if not self.softmax:
            return value
        else:
            return F.softmax(value, dim=1)

    def get_features(self, xyz, detach=False):
        x, trans, trans_feat = self.feat(xyz)
        if detach:
            x = x.detach()
        return x


class CriticPointNet2(nn.Module):
    def __init__(self, output_dim, action_dim, normal_channel=False, softmax=False):
        super(CriticPointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024+action_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim)
        self.softmax = softmax

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        value = self.fc3(x)

        if not self.softmax:
            return value
        else:
            return F.softmax(value, dim=1)

    def get_features(self, xyz, detach=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        if detach:
            x = x.detach()
        return x

    def get_action(self, inputs):
        values = self.forward(inputs)
        return T.argmax(values).item()
