"""
Transformer Blocks for Dual-Level DiT.

1. `PatchDiTBlock`:
   - Standard DiT block for the `16x16` patch-level (detail) branch.
   - Inputs: patch token embeddings `x` `(B, N_detail, D_detail)` and condition `c` `(B, D_detail)`.
   - Output: updated patch token embeddings `(B, N_detail, D_detail)`.

2. `GlobalDiTBlock`:
   - Pixel-level (`1x1`) DiT block conditioned on the paired `PatchDiTBlock` output `cond`:
     * Conditioner MLP expands patch-level `cond` into pixel-wise `(alpha1, alpha2, beta1, beta2, gamma1, gamma2)`.
     * Attention branch: `RMSNorm` -> pixel-wise scale/shift (`gamma1, beta1`) -> spatial token compaction + Linear downsample
       -> 2D RoPE + Multi-Head Self-Attention -> Linear upsample + spatial unpack -> gate (`alpha1`) + residual.
     * FFN branch: `RMSNorm` -> pixel-wise scale/shift (`gamma2, beta2`) -> FFN -> gate (`alpha2`) + residual.

3. `DualLevelPairBlock`:
   - Couples one `PatchDiTBlock` and one `GlobalDiTBlock` per layer.

4. `SpatialAdaLNFinalLayer`:
   - Final unpatchify projection head conditioned on the final `detail_emb`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding2D


def modulate_1d(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Applies global condition modulation across sequence dimension.
    Args:
        x: `(B, N, D)`
        shift: `(B, D)`
        scale: `(B, D)`
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_spatial(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Applies per-token (pixel-wise) modulation where `shift` (`beta`) and `scale` (`gamma`)
    match `(B, N, D)`.
    """
    return x * (1.0 + scale) + shift


def pack_pixels_to_patches(
    x: torch.Tensor,
    patch_grid: tuple[int, int],
    scale_factor: int,
) -> torch.Tensor:
    """
    Packs pixel-level tokens `(B, H * W, D)` into patch-grouped tokens
    `(B, gh * gw, s * s * D)`, aligned with `ImageTokenizer(patch_size=s)` spatial order.

    Args:
        x: Pixel-level tensor of shape `(B, (gh * s) * (gw * s), D)`.
        patch_grid: `(gh, gw)` spatial grid of patches (`(H // s, W // s)`).
        scale_factor: Spatial ratio `s` (`detail_patch_size // global_patch_size`, e.g., `16`).

    Returns:
        Packed patch tensor of shape `(B, gh * gw, s * s * D)`.
    """
    if scale_factor == 1:
        return x

    B, N_global, D = x.shape
    gh, gw = patch_grid
    s = scale_factor
    expected_n = gh * s * gw * s
    if N_global != expected_n:
        raise ValueError(
            f"Global token length ({N_global}) does not match patch_grid {patch_grid} "
            f"with scale_factor ({s}) -> expected {expected_n}."
        )

    # (B, gh * s, gw * s, D) -> (B, gh, s, gw, s, D) -> (B, gh, gw, s, s, D) -> (B, gh * gw, s * s * D)
    x = x.view(B, gh, s, gw, s, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, gh * gw, s * s * D)


def unpack_patches_to_pixels(
    x: torch.Tensor,
    patch_grid: tuple[int, int],
    scale_factor: int,
) -> torch.Tensor:
    """
    Inverse of `pack_pixels_to_patches`: unpacks patch-grouped tensor `(B, gh * gw, s * s * C)`
    back into row-major pixel-level token order `(B, (gh * s) * (gw * s), C)`.

    Args:
        x: Patch-grouped tensor of shape `(B, gh * gw, s * s * C)`.
        patch_grid: `(gh, gw)` spatial grid of patches (`(H // s, W // s)`).
        scale_factor: Spatial ratio `s` (`detail_patch_size // global_patch_size`, e.g., `16`).

    Returns:
        Pixel-level tensor of shape `(B, (gh * s) * (gw * s), C)`.
    """
    if scale_factor == 1:
        return x

    B, N_patch, packed_dim = x.shape
    gh, gw = patch_grid
    s = scale_factor
    if N_patch != gh * gw:
        raise ValueError(
            f"Patch token length ({N_patch}) does not match patch_grid {patch_grid}."
        )
    if packed_dim % (s * s) != 0:
        raise ValueError(
            f"Packed channel dim ({packed_dim}) must be divisible by scale_factor^2 ({s * s})."
        )

    C = packed_dim // (s * s)
    # (B, gh, gw, s, s, C) -> (B, gh, s, gw, s, C) -> (B, gh * s * gw * s, C)
    x = x.view(B, gh, gw, s, s, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, gh * s * gw * s, C)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (computed in fp32 for numerical stability).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(
            x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        normed = normed.to(dtype=x.dtype)
        if self.weight is not None:
            normed = normed * self.weight
        return normed


class SelfAttention(nn.Module):
    """
    Multi-head Self-Attention using `F.scaled_dot_product_attention`,
    with optional 2D Axial RoPE injection on `Q` and `K`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rope = (
            RotaryEmbedding2D(self.head_dim, theta=rope_theta) if use_rope else None
        )
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)

        if self.rope is not None:
            if grid_size is None:
                gh = gw = int(N**0.5)
                if gh * gw != N:
                    raise ValueError(
                        f"grid_size must be provided for non-square token count ({N}) when using 2D RoPE."
                    )
                grid_size = (gh, gw)
            q, k = self.rope(q, k, grid_size)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    Transformer Feed-Forward Network (FFN) with GELU (tanh approx) activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchDiTBlock(nn.Module):
    """
    Standard DiT Block with adaptive layer norm zero (adaLN-Zero) conditioning
    for the `16x16` patch-level branch.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, use_rope=False
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch token embeddings `(B, N_detail, D_detail)`.
            c: Combined `timestep + class` condition `(B, D_detail)`.

        Returns:
            Updated patch token embeddings `(B, N_detail, D_detail)`.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate_1d(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate_1d(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class GlobalDiTBlock(nn.Module):
    """
    Global (`1x1` pixel-level) DiT Block conditioned on the paired `PatchDiTBlock` output.

    Forward pipeline:
      1. Conditioner MLP outputs pixel-wise `(alpha1, alpha2, beta1, beta2, gamma1, gamma2)`:
         - `gamma1, beta1` / `gamma2, beta2`: pixel-wise scale & shift
         - `alpha1, alpha2`: residual gate ratios
      2. `RMSNorm(x)` -> scale & shift (`gamma1, beta1`) -> spatial compaction + Linear downsample
         -> 2D RoPE + Multi-Head Self-Attention -> Linear upsample + spatial unpack -> gate (`alpha1`) -> `attn_out`
      3. `x = x + attn_out` -> `RMSNorm(x)` -> scale & shift (`gamma2, beta2`) -> FFN -> gate (`alpha2`) -> `x + ffn_out`
    """

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        num_heads: int,
        attn_dim: int | None = None,
        scale_factor: int = 16,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        attn_dim = attn_dim or cond_dim
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor

        packed_pixel_dim = scale_factor * scale_factor * hidden_size

        # Conditioner MLP: expands each patch cond into 16x16 pixel-wise (alpha1, alpha2, beta1, beta2, gamma1, gamma2)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * packed_pixel_dim, bias=True),
        )

        # Attention branch: RMSNorm -> scale/shift -> Linear downsample -> 2D RoPE + MHSA -> Linear upsample -> gate
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.downsample = nn.Linear(packed_pixel_dim, attn_dim, bias=True)
        self.attn = SelfAttention(
            attn_dim,
            num_heads=num_heads,
            qkv_bias=True,
            use_rope=True,
            rope_theta=rope_theta,
        )
        self.upsample = nn.Linear(attn_dim, packed_pixel_dim, bias=True)

        # FFN branch: RMSNorm -> scale/shift -> FFN -> gate
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        detail_grid: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Pixel-level global token embeddings `(B, H * W, D_global)`.
            cond: Patch token embeddings `(B, N_detail, D_detail)` from paired `PatchDiTBlock`.
            detail_grid: Optional `(H // 16, W // 16)` spatial dimensions of `cond`.

        Returns:
            Updated global token embeddings `(B, H * W, D_global)`.
        """
        if detail_grid is None:
            n_patch = cond.shape[1]
            gh = gw = int(n_patch**0.5)
            if gh * gw != n_patch:
                raise ValueError(
                    f"detail_grid must be provided for non-square patch token count ({n_patch})."
                )
            detail_grid = (gh, gw)

        # 1. Conditioner MLP -> pixel-wise alpha1, alpha2, beta1, beta2, gamma1, gamma2
        packed_mod = self.cond_mlp(cond)  # (B, N_detail, s^2 * 6 * D_global)
        pixel_mod = unpack_patches_to_pixels(
            packed_mod, patch_grid=detail_grid, scale_factor=self.scale_factor
        )  # (B, H * W, 6 * D_global)
        alpha1, alpha2, beta1, beta2, gamma1, gamma2 = pixel_mod.chunk(6, dim=-1)

        # 2. RMSNorm -> scale & shift -> Linear downsample -> RoPE + MHSA -> Linear upsample -> gate
        h = modulate_spatial(self.norm1(x), shift=beta1, scale=gamma1)
        h = pack_pixels_to_patches(
            h, patch_grid=detail_grid, scale_factor=self.scale_factor
        )  # (B, N_detail, s^2 * D_global)
        h = self.downsample(h)  # (B, N_detail, attn_dim)
        h = self.attn(h, grid_size=detail_grid)  # (B, N_detail, attn_dim)
        h = self.upsample(h)  # (B, N_detail, s^2 * D_global)
        h = unpack_patches_to_pixels(
            h, patch_grid=detail_grid, scale_factor=self.scale_factor
        )  # (B, H * W, D_global)
        attn_out = alpha1 * h

        # 3. Residual add -> RMSNorm -> scale & shift -> FFN -> gate -> Residual add
        x = x + attn_out
        h2 = modulate_spatial(self.norm2(x), shift=beta2, scale=gamma2)
        ffn_out = self.mlp(h2)
        x = x + alpha2 * ffn_out
        return x


class DualLevelPairBlock(nn.Module):
    """
    Paired layer coupling one `PatchDiTBlock` and one `GlobalDiTBlock`:
      1. `detail_emb = patch_block(detail_emb, c)`
      2. `global_emb = global_block(global_emb, detail_emb, detail_grid)`
    """

    def __init__(
        self,
        detail_hidden_size: int,
        detail_num_heads: int,
        global_hidden_size: int,
        global_num_heads: int,
        global_attn_dim: int | None = None,
        scale_factor: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_block = PatchDiTBlock(
            hidden_size=detail_hidden_size,
            num_heads=detail_num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.global_block = GlobalDiTBlock(
            hidden_size=global_hidden_size,
            cond_dim=detail_hidden_size,
            num_heads=global_num_heads,
            attn_dim=global_attn_dim or detail_hidden_size,
            scale_factor=scale_factor,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        detail_emb: torch.Tensor,
        global_emb: torch.Tensor,
        c: torch.Tensor,
        detail_grid: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail_emb = self.patch_block(detail_emb, c)
        global_emb = self.global_block(global_emb, detail_emb, detail_grid=detail_grid)
        return detail_emb, global_emb


class SpatialAdaLNFinalLayer(nn.Module):
    """
    Final output layer for the Global DiT stream.
    Uses pixel-wise RMSNorm modulation conditioned on the final `detail_emb` to project
    pixel-level tokens back to image channels (`patch_size_global^2 * out_channels`).
    """

    def __init__(
        self,
        global_hidden_size: int,
        detail_hidden_size: int,
        patch_size_global: int,
        out_channels: int,
        scale_factor: int = 16,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        packed_pixel_dim = scale_factor * scale_factor * global_hidden_size

        self.norm_final = RMSNorm(global_hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(
            global_hidden_size,
            patch_size_global * patch_size_global * out_channels,
            bias=True,
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(detail_hidden_size, 2 * packed_pixel_dim, bias=True),
        )

    def forward(
        self,
        global_emb: torch.Tensor,
        detail_cond: torch.Tensor,
        detail_grid: tuple[int, int],
    ) -> torch.Tensor:
        packed_mod = self.cond_mlp(detail_cond)
        pixel_mod = unpack_patches_to_pixels(
            packed_mod, patch_grid=detail_grid, scale_factor=self.scale_factor
        )
        beta, gamma = pixel_mod.chunk(2, dim=-1)
        x = modulate_spatial(self.norm_final(global_emb), shift=beta, scale=gamma)
        x = self.linear(x)
        return x
