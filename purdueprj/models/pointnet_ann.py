import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetBackboneANN(nn.Module):
    def __init__(self, hidden_dims=[128,256,512]):
        super().__init__()
        layers = []
        in_dim = 3
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, pts):
        B, N, _ = pts.shape
        x = pts.reshape(B*N, -1)
        x = self.net(x)
        return x.reshape(B, N, -1)


class TemporalANN(nn.Module):
    def __init__(self, dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x)


class PointNetANN(nn.Module):
    def __init__(self,
                 point_dims=[128,256,512],
                 temporal_dim=512,
                 num_classes=10):
        super().__init__()

        self.backbone = PointNetBackboneANN(point_dims)
        self.temporal = TemporalANN(temporal_dim, num_classes)

    def forward_full(self, pts):
        per_point = self.backbone(pts)
        global_feat = per_point.mean(dim=1)
        return self.temporal(global_feat)

    def forward_step(self, pts_slice):
        # 1. Per-point MLP
        per_point = self.backbone(pts_slice)        # [B, N_slice, 256/512]
        
        # 2. Mean pooling
        slice_feat = per_point.mean(dim=1)          # [B, 256/512]
        
        # 3. Temporal (or MLP)
        return self.temporal(slice_feat)
