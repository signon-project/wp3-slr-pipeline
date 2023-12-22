import math

import torch
from torch import nn


class KeypointEmbeddingDense(nn.Module):
    """Dense framewise keypoint embedding."""

    def __init__(self, in_features: int, d_embed: int, residual: bool):
        super(KeypointEmbeddingDense, self).__init__()

        self.in_features = in_features
        self.d_embed = d_embed
        self.residual = residual

        self.embedding = nn.Sequential(
            nn.Linear(self.in_features, self.in_features * 4),
            nn.LayerNorm(self.in_features * 4),
            nn.ReLU(),
            nn.Dropout(0.125),
            nn.Linear(self.in_features * 4, self.d_embed * 4),
            nn.LayerNorm(self.d_embed * 4),
            nn.ReLU(),
            nn.Dropout(0.125),
            nn.Linear(self.d_embed * 4, self.d_embed * 2),
            nn.LayerNorm(self.d_embed * 2),
            nn.ReLU(),
            nn.Dropout(0.125),
            nn.Linear(self.d_embed * 2, self.d_embed),
            nn.LayerNorm(self.d_embed),
        )
        if self.residual:
            self.resid = nn.Linear(self.in_features, self.d_embed)
        self.final_activation = nn.Tanh()

    def forward(self, pose_clip: torch.Tensor):
        t, b, c = pose_clip.size()

        pose_minibatch = pose_clip.reshape(t * b, -1)
        if self.residual:
            pose_embedded = self.embedding(pose_minibatch).reshape(t, b, -1) + self.resid(pose_clip)
        else:
            pose_embedded = self.embedding(pose_minibatch).reshape(t, b, -1)
        return self.final_activation(pose_embedded)


class PositionalEncoding(nn.Module):
    """Positional encoding as proposed in the 'Attention is all you need' paper.
    Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        """Initialize the positional encoding.

        :param d_model: Size of the inputs to the self-attention network.
        :param dropout: Dropout probability.
        :param max_len: Maximum sequence length that we can expect.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the positional encoding.

        :param x: The input tensor of shape (time, batch, d_model).
        :return: An output tensor of shape (time, batch, d_model).
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
