"""
Conditioning (Timestep & Class Embeddings) and Positional Embeddings for Dual-Level DiT.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar diffusion timesteps into vector representations via
    sinusoidal positional encoding followed by an MLP + SiLU.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: 1-D Tensor of `B` indices, one per batch element (may be fractional).
            dim: Dimension of the output embedding.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            Tensor of shape `(B, dim)`.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout
    for classifier-free guidance (CFG).
    """

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding), hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Drops labels to enable classifier-free guidance during training.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: tuple[int, int] | int
) -> np.ndarray:
    """
    Generates 2D sine-cosine positional embeddings of shape `(grid_h * grid_w, embed_dim)`.
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size

    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sin-cos pos embed.")

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class RotaryEmbedding2D(nn.Module):
    """
    2D Axial Rotary Position Embedding (RoPE) for 2D spatial token grids `(gh, gw)`.

    Splits `head_dim` into two halves:
      - First `head_dim // 2` channels encode row (height) coordinates `0 ... gh - 1`
      - Second `head_dim // 2` channels encode column (width) coordinates `0 ... gw - 1`
    Requires `head_dim % 4 == 0`.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 4 for 2D Axial RoPE."
            )
        self.head_dim = head_dim
        self.theta = theta

        half_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_grid: tuple[int, int] | None = None
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _get_cos_sin(
        self, grid_size: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_grid == grid_size
            and self._cached_cos is not None
            and self._cached_sin is not None
            and self._cached_cos.device == device
        ):
            return self._cached_cos, self._cached_sin

        gh, gw = grid_size
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        pos_h = torch.arange(gh, device=device, dtype=torch.float32)
        pos_w = torch.arange(gw, device=device, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing="ij")

        freqs_h = torch.outer(grid_h.reshape(-1), inv_freq)  # (gh * gw, head_dim // 4)
        freqs_w = torch.outer(grid_w.reshape(-1), inv_freq)  # (gh * gw, head_dim // 4)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # (gh * gw, head_dim // 2)

        cos = freqs.cos()[None, None, :, :]  # (1, 1, N, head_dim // 2)
        sin = freqs.sin()[None, None, :, :]  # (1, 1, N, head_dim // 2)

        self._cached_grid = grid_size
        self._cached_cos = cos
        self._cached_sin = sin
        return cos, sin

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # Pairwise rotation on (B, num_heads, N, head_dim // 2, 2) in fp32
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2).to(dtype=x.dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, grid_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D RoPE to query and key tensors.

        Args:
            q: `(B, num_heads, gh * gw, head_dim)`
            k: `(B, num_heads, gh * gw, head_dim)`
            grid_size: `(gh, gw)` spatial grid dimensions.

        Returns:
            Rotated `(q, k)` tensors preserving input shape and dtype.
        """
        cos, sin = self._get_cos_sin(grid_size, device=q.device)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
