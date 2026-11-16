"""Compact inference encoders. Constructors initialize untrained weights."""
import torch
from torch import nn


def _classes(classes):
    if type(classes) is not int or classes < 2:
        raise ValueError('classes must be an integer >= 2')


def _input(x, shape):
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or tuple(x.shape[1:]) != shape or not len(x):
        raise ValueError(f'Expected nonempty [batch,{shape[0]},{shape[1]}] tensor')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Input must be finite floating point')


class SensorCNN(nn.Module):
    """Six-channel, 128-step sensor branch; 64-dimensional embedding."""
    def __init__(self, classes=27):
        super().__init__(); _classes(classes)
        self.encoder = nn.Sequential(nn.Conv1d(6, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Linear(64, classes)

    def embed(self, x):
        _input(x, (128, 6))
        return self.encoder(x.transpose(1, 2))

    def forward(self, x):
        return self.head(self.embed(x))


class HOFGRU(nn.Module):
    """16 by 128 HOF branch, single 64-unit GRU."""
    def __init__(self, classes=27):
        super().__init__(); _classes(classes)
        self.norm = nn.LayerNorm(128)
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.head = nn.Linear(64, classes)

    def embed(self, x):
        _input(x, (16, 128))
        _, h = self.gru(self.norm(x))
        return h[-1]

    def forward(self, x):
        return self.head(self.embed(x))


class HOFTransformer(nn.Module):
    """One four-head temporal encoder, 64-dimensional embedding."""
    def __init__(self, classes=27):
        super().__init__(); _classes(classes)
        self.norm = nn.LayerNorm(128)
        self.projection = nn.Linear(128, 64)
        self.position = nn.Parameter(torch.empty(16, 64))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(d_model=64, nhead=4,
            dim_feedforward=96, dropout=0.1, activation='gelu', batch_first=True,
            norm_first=False)
        self.final_norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, classes)

    def embed(self, x):
        _input(x, (16, 128))
        tokens = self.projection(self.norm(x)) + self.position
        return self.final_norm(self.encoder(tokens)).mean(dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


MODELS = {'hof_gru': HOFGRU, 'hof_transformer': HOFTransformer}
