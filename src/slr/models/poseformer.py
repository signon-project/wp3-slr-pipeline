import math

import numpy as np
import torch
import torch.nn as nn

from .ptn import _normalize


class FeatureProcessing(nn.Module):
    def __init__(self):
        super(FeatureProcessing, self).__init__()

    def forward(self, pose_clips: torch.Tensor) -> torch.Tensor:
        # Input is a padded batch (pad value: NaN) containing multiple samples.
        # The shape is (length, batch, keypoints, coordinates).
        # Imputation and augmentation (if training) have already been applied.
        # In this module, we perform normalization and feature extraction.

        feature_list = []

        # Normalization: shift and scale.
        pose_clips = _normalize(pose_clips)
        # Selected keypoints: left hand, right hand.
        keypoints = torch.cat([
            torch.arange(0, 25),
            torch.arange(33, 33 + 42),
        ])
        pose_clips = pose_clips[..., keypoints, :]
        # We only keep x and y, and drop z.
        positions = pose_clips[..., :2]
        # Ravel.
        positions = positions.reshape(positions.size(0), positions.size(1), -1)
        feature_list.append(positions)
        # Extract velocity features.
        #velocity = pose_clips[:, 1:] - pose_clips[:, :-1]
        #velocity = torch.cat([torch.zeros_like(pose_clips[:, :1, ...]), velocity], dim=1)
        # Drop z.
        #velocity = velocity[..., :2]
        # Ravel.
        #velocity = velocity.reshape(velocity.size(0), velocity.size(1), -1)
        #feature_list.append(velocity)
        # If any features are needed other than x and y coordinates, here is where you'd extract them.
        # Or, if they need to be extracted based on 3D coordinates, then above the previous line.
        features = torch.cat(feature_list, dim=-1)
        return features


@torch.jit.script
def gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GeLU(nn.Module):
    def forward(self, x):
        return gelu(x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, mask=None):
        b, t, e = x.shape
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)
        q = q.view(b, t, self.n_heads, -1).transpose(1, 2)
        k = k.view(b, t, self.n_heads, -1).transpose(1, 2)
        v = v.view(b, t, self.n_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (e / self.n_heads) ** 0.5)
        if mask is not None:
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float("-inf"))
            mask = mask.view(b, 1, 1, t).expand(-1, self.n_heads, -1, -1)
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, t, e)
        x = self.proj(x)
        x = self.resid_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout2(self.fc2(gelu(self.dropout1(self.fc1(x)))))
        return x


class TransformerBlock(nn.Module):
    """Equivalent to nn.TransformerEncoderLayer with batch_first and norm_first set to True, GeLU
    but with operations supported for the tflite conversion."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.4, drop_path=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = MLP(d_model, dim_feedforward, dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), src_key_padding_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Pooling(nn.Module):
    def __init__(self, pool_type="mean", prefix_tokens=1):
        super().__init__()
        self.pool_type = pool_type
        self.prefix_tokens = prefix_tokens

    def forward(self, x):
        if self.pool_type == "mean":
            return torch.nanmean(x[:, self.prefix_tokens:, ...], dim=1)
        elif self.pool_type == "max":
            return x.max(dim=1)[0]
        elif self.pool_type == "min":
            return x.min(dim=1)[0]
        elif self.pool_type == "cls":
            return x[:, 0]
        else:
            raise ValueError(f"Invalid pool type: {self.pool_type}")


class FrameEmbed(nn.Module):
    def __init__(self, feature_size, embed_dim, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_size, feature_size * 4),
            nn.LayerNorm(feature_size * 4),
            GeLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size * 4, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            GeLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            GeLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.upsample = nn.Linear(feature_size, embed_dim)

    def forward(self, x):
        b, t, f = x.shape
        x = x.reshape(b * t, f)
        up_x = self.upsample(x)
        x = up_x + self.proj(x)
        x = x.reshape(b, t, -1)
        return x


class PoseFormer(nn.Module):
    def __init__(
            self,
            feature_size=248,
            embed_dim=256,
            n_layers=4,
            n_heads=8,
            dim_feedforward=512,
            dropout=0.2,
            layer_norm_eps=1e-5,
            pooling="cls",
            num_classes=[250],
            lang_embedding_size=4,
            l1_label_norm=False,
            multilingual_conditioned=False
    ):
        super().__init__()
        self.feature_extractor = FeatureProcessing()

        self.embed_dim = embed_dim
        self.frame_embed = FrameEmbed(feature_size, embed_dim, dropout)
        self.pos_embed = PositionalEncoding(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pooler = Pooling(pooling)
        self.multilingual_conditioned = multilingual_conditioned

        if self.multilingual_conditioned:
            self.language_embedding = nn.Embedding(len(num_classes), lang_embedding_size)
        else:  # multilingual joint or monolingual
            self.language_embedding = None

        self.input_convs = nn.ModuleList(
            [
                nn.Conv1d(feature_size, feature_size, 3, groups=feature_size, padding=1),
                nn.Conv1d(feature_size, feature_size, 5, groups=feature_size, padding=2),
                nn.Conv1d(feature_size, feature_size, 7, groups=feature_size, padding=3),
                nn.Conv1d(feature_size, feature_size, 9, groups=feature_size, padding=4),
            ]
        )

        self.filters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, 3, groups=embed_dim, padding="same"),
                    GeLU(),
                    nn.Conv1d(embed_dim, embed_dim, 3, groups=embed_dim, padding="same"),
                ),
                nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, 5, groups=embed_dim, padding="same"),
                    GeLU(),
                    nn.Conv1d(embed_dim, embed_dim, 5, groups=embed_dim, padding="same"),
                ),
                nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, 7, groups=embed_dim, padding="same"),
                    GeLU(),
                    nn.Conv1d(embed_dim, embed_dim, 7, groups=embed_dim, padding="same"),
                ),
                nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, 9, groups=embed_dim, padding="same"),
                    GeLU(),
                    nn.Conv1d(embed_dim, embed_dim, 9, groups=embed_dim, padding="same"),
                ),
            ]
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    drop_path=0.1,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        if self.multilingual_conditioned:
            self.head = nn.Linear(embed_dim + lang_embedding_size, int(np.sum(np.array(num_classes))))
        else:  # multilingual joint or monolingual
            self.head = nn.Linear(embed_dim, int(np.sum(np.array(num_classes))))

        self.logit_norm = None
        if l1_label_norm:
            self.logit_norm = nn.LayerNorm(num_classes, eps=0.03)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, model_inputs, language_ids):
        pose_clip = self.feature_extractor(model_inputs)
        pose_clip = torch.nan_to_num(pose_clip, 0)
        x = pose_clip.permute(1, 0, 2)  # PoseFormer works batch first.

        b, s, f = x.shape
        mask = torch.sum(x, dim=-1) == 0
        reverse_mask = (~mask).unsqueeze(-1).repeat(1, 1, f).float()
        for conv in self.input_convs:
            x = x + conv(x.transpose(1, 2)).transpose(1, 2)
            x = x * reverse_mask
        x = self.frame_embed(x)
        b, s, f = x.shape
        reverse_mask = (~mask).unsqueeze(-1).repeat(1, 1, f).float()
        x = x * reverse_mask
        for f in self.filters:
            x = x + f(x.transpose(1, 2)).transpose(1, 2)
            x = x * reverse_mask
        x = self.pos_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        mask = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=x.device), mask), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=mask)
        x = self.norm(x)
        x = x[:, 0]

        if self.language_embedding is not None:
            emb_lang = self.language_embedding(language_ids)
            x = torch.cat((x, emb_lang), dim=-1)

        x = self.head(x)
        if self.logit_norm is not None:
            x = self.logit_norm(x)

        return x

    def setup_inference_hook(self, embedding_kind: str, hook):
        if embedding_kind == 'spatial':
            self.frame_embed.register_forward_hook(hook)
        elif embedding_kind == 'temporal' or embedding_kind == 'CLS':
            self.blocks[-1].register_forward_hook(hook)
        else:
            raise ValueError(f'Unsupported embedding kind {embedding_kind} for inference hook.')
