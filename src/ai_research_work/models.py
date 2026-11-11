"""Inference model definitions; no fitting utilities."""
from torch import nn

class SensorModel(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(6, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Linear(64, classes)

    def embed(self, x):
        return self.encoder(x.transpose(1, 2))

    def forward(self, x):
        return self.head(self.embed(x))

class VideoModel(nn.Module):
    def __init__(self, classes=8, features=512):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.gru = nn.GRU(features, 64, batch_first=True)
        self.head = nn.Linear(64, classes)

    def embed(self, x):
        _, h = self.gru(self.norm(x))
        return h[-1]

    def forward(self, x):
        return self.head(self.embed(x))
