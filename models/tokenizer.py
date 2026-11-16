"""
Image Tokenizer (Patchify & Unpatchify) modules for Dual-Level DiT.

Provides an MLP + SiLU based image tokenizer that splits an image into
non-overlapping `patch_size x patch_size` patches:
- Detail branch (`patch_size = 16`): patch-level tokens `(H/16) * (W/16)`
- Global branch (`patch_size = 1`): pixel-level tokens `H * W`
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageTokenizer(nn.Module):
    """
    Patchifies an image into sequence tokens using non-overlapping patch unfolding
    followed by an `MLP + SiLU` projection.

    Args:
        img_size: Input spatial resolution `(H, W)` or int `S` for square images.
        patch_size: Spatial size `P` of each square patch (`16` for patch-level, `1` for pixel-level).
        in_channels: Number of input image channels `C`.
        embed_dim: Output token embedding dimension `D`.
        mlp_ratio: Hidden dimension multiplier inside the tokenizer MLP.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        mlp_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_h, img_w = img_size, img_size
        else:
            img_h, img_w = img_size

        if img_h % patch_size != 0 or img_w % patch_size != 0:
            raise ValueError(
                f"img_size ({img_h}, {img_w}) must be divisible by patch_size ({patch_size})."
            )

        self.img_size = (img_h, img_w)
        self.patch_size = patch_size
        self.grid_size = (img_h // patch_size, img_w // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        patch_dim = in_channels * patch_size * patch_size
        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim, bias=True),
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearranges `(B, C, H, W)` into flattened patches `(B, N, P * P * C)`.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Input spatial size ({H}, {W}) must be divisible by patch_size ({p})."
            )

        gh, gw = H // p, W // p
        # (B, C, gh, p, gw, p) -> (B, gh, gw, p, p, C) -> (B, gh * gw, p * p * C)
        x = x.reshape(B, C, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(B, gh * gw, p * p * C)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape `(B, C, H, W)`.

        Returns:
            Token embeddings of shape `(B, (H/P)*(W/P), embed_dim)`.
        """
        patches = self.patchify(x)
        tokens = self.mlp(patches)
        return tokens


def unpatchify(
    tokens: torch.Tensor,
    patch_size: int,
    out_channels: int,
    hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Converts projected patch tokens `(B, N, P * P * out_channels)` back to image `(B, out_channels, H, W)`.

    Args:
        tokens: Tensor of shape `(B, gh * gw, P * P * out_channels)`.
        patch_size: Spatial size `P` of each patch (`1` for pixel-level output).
        out_channels: Number of output image channels.
        hw: Optional `(H, W)` image spatial resolution. If `None`, assumes square grid.

    Returns:
        Reconstructed image tensor of shape `(B, out_channels, H, W)`.
    """
    B, N, D = tokens.shape
    p = patch_size
    expected_dim = p * p * out_channels
    if D != expected_dim:
        raise ValueError(
            f"Token channel dim ({D}) does not match patch_size^2 * out_channels ({expected_dim})."
        )

    if hw is not None:
        gh, gw = hw[0] // p, hw[1] // p
    else:
        gh = gw = int(N**0.5)

    if gh * gw != N:
        raise ValueError(
            f"Number of tokens ({N}) does not match patch grid ({gh} x {gw})."
        )

    # (B, gh, gw, p, p, C) -> (B, C, gh, p, gw, p) -> (B, C, H, W)
    x = tokens.reshape(B, gh, gw, p, p, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, out_channels, gh * p, gw * p)
    return x
