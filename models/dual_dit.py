"""
Dual-Level Diffusion Transformer (DualLevelDiT) Main Architecture.

Data flow summary:
  - Inputs:
      * `x`: Noised image `(B, C, H, W)`
      * `t`: Diffusion timesteps `(B,)`
      * `y`: Class labels `(B,)`
  - Condition `c`:
      * `c = t_embedder(t) + y_embedder(y)` of shape `(B, D_detail)`
  - Dual Tokenization (`MLP + SiLU`):
      * `detail_emb = detail_tokenizer(x) + detail_pos_embed`  [16x16 patchify -> `(B, (H/16)*(W/16), D_detail)`]
      * `global_emb = global_tokenizer(x) + global_pos_embed`  [1x1 patchify   -> `(B, H*W, D_global)`]
  - Layer-wise Paired Blocks (`i = 0 ... depth - 1`):
      * `detail_emb = detail_block_i(detail_emb, c)`           (passes `detail_emb` to next detail block & paired global block)
      * `global_emb = global_block_i(global_emb, detail_emb)`  (spatial adaLN-Zero conditioned on `detail_emb`)
  - Final Layer & Unpatchify:
      * `out_tokens = final_layer(global_emb, detail_emb)`
      * `pred = unpatchify(out_tokens)` -> `(B, out_channels, H, W)`
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DualLevelPairBlock, SpatialAdaLNFinalLayer
from .embeddings import LabelEmbedder, TimestepEmbedder, get_2d_sincos_pos_embed
from .tokenizer import ImageTokenizer, unpatchify


class DualLevelDiT(nn.Module):
    """
    Dual-Level Diffusion Transformer simultaneously modeling patch-level (`16x16`)
    and pixel-level (`1x1`) representations.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 256,
        in_channels: int = 3,
        detail_patch_size: int = 16,
        global_patch_size: int = 1,
        detail_hidden_size: int = 768,
        global_hidden_size: int = 16,
        global_attn_dim: int | None = None,
        depth: int = 12,
        detail_num_heads: int = 12,
        global_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        tokenizer_mlp_ratio: float = 1.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()
        if detail_patch_size % global_patch_size != 0:
            raise ValueError(
                f"detail_patch_size ({detail_patch_size}) must be divisible by "
                f"global_patch_size ({global_patch_size})."
            )

        global_attn_dim = global_attn_dim or detail_hidden_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.detail_patch_size = detail_patch_size
        self.global_patch_size = global_patch_size
        self.scale_factor = detail_patch_size // global_patch_size
        self.detail_hidden_size = detail_hidden_size
        self.global_hidden_size = global_hidden_size
        self.global_attn_dim = global_attn_dim
        self.num_classes = num_classes
        self.depth = depth

        # 1. Dual Image Tokenizers (MLP + SiLU)
        # Detail branch: 16x16 patchify
        self.detail_tokenizer = ImageTokenizer(
            img_size=img_size,
            patch_size=detail_patch_size,
            in_channels=in_channels,
            embed_dim=detail_hidden_size,
            mlp_ratio=tokenizer_mlp_ratio,
        )
        # Global branch: 1x1 pixel-level patchify
        self.global_tokenizer = ImageTokenizer(
            img_size=img_size,
            patch_size=global_patch_size,
            in_channels=in_channels,
            embed_dim=global_hidden_size,
            mlp_ratio=tokenizer_mlp_ratio,
        )

        # 2. Condition Embedders (Timestep + Class Label -> Detail DiT condition)
        self.t_embedder = TimestepEmbedder(detail_hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes=num_classes,
            hidden_size=detail_hidden_size,
            dropout_prob=class_dropout_prob,
        )

        # 3. Fixed 2D Sin-Cos Positional Embeddings for both levels
        self.detail_pos_embed = nn.Parameter(
            torch.zeros(1, self.detail_tokenizer.num_patches, detail_hidden_size),
            requires_grad=False,
        )
        self.global_pos_embed = nn.Parameter(
            torch.zeros(1, self.global_tokenizer.num_patches, global_hidden_size),
            requires_grad=False,
        )

        # 4. Layer-wise Paired Patch & Global DiT Blocks
        self.blocks = nn.ModuleList(
            [
                DualLevelPairBlock(
                    detail_hidden_size=detail_hidden_size,
                    detail_num_heads=detail_num_heads,
                    global_hidden_size=global_hidden_size,
                    global_num_heads=global_num_heads,
                    global_attn_dim=global_attn_dim,
                    scale_factor=self.scale_factor,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # 5. Final Output Head (Global stream conditioned on final detail_emb)
        self.final_layer = SpatialAdaLNFinalLayer(
            global_hidden_size=global_hidden_size,
            detail_hidden_size=detail_hidden_size,
            patch_size_global=global_patch_size,
            out_channels=self.out_channels,
            scale_factor=self.scale_factor,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Follows standard DiT initialization:
          - Xavier uniform for all Linear layers
          - Fixed 2D sin-cos positional embeddings
          - Normal(0, 0.02) for label & timestep embedders
          - Zero-init for all adaLN-Zero modulation projections and final linear head
        """
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fixed 2D sin-cos positional embeddings
        detail_pos = get_2d_sincos_pos_embed(
            self.detail_hidden_size, self.detail_tokenizer.grid_size
        )
        self.detail_pos_embed.data.copy_(
            torch.from_numpy(detail_pos).float().unsqueeze(0)
        )

        global_pos = get_2d_sincos_pos_embed(
            self.global_hidden_size, self.global_tokenizer.grid_size
        )
        self.global_pos_embed.data.copy_(
            torch.from_numpy(global_pos).float().unsqueeze(0)
        )

        # Initialize label embedding table & timestep MLP
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN-Zero / conditioner MLP modulation layers in Patch and Global DiT blocks
        for pair_block in self.blocks:
            nn.init.constant_(pair_block.patch_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(pair_block.patch_block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(pair_block.global_block.cond_mlp[-1].weight, 0)
            nn.init.constant_(pair_block.global_block.cond_mlp[-1].bias, 0)

        # Zero-out final output projection layer
        nn.init.constant_(self.final_layer.cond_mlp[-1].weight, 0)
        nn.init.constant_(self.final_layer.cond_mlp[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of DualLevelDiT.

        Args:
            x: `(B, C, H, W)` noised image tensor.
            t: `(B,)` diffusion timesteps.
            y: `(B,)` class labels.
            force_drop_ids: Optional `(B,)` mask to force class dropout for CFG.

        Returns:
            Predicted noise (and optionally variance) of shape `(B, out_channels, H, W)`.
        """
        _, _, H, W = x.shape
        detail_grid = (H // self.detail_patch_size, W // self.detail_patch_size)

        # 1. Tokenize image at both granularities + add 2D positional embeddings
        detail_emb = self.detail_tokenizer(x)  # (B, (H/16)*(W/16), D_detail)
        detail_emb = detail_emb + self.detail_pos_embed.to(dtype=detail_emb.dtype)

        global_emb = self.global_tokenizer(x)  # (B, H*W, D_global)
        global_emb = global_emb + self.global_pos_embed.to(dtype=global_emb.dtype)

        # 2. Build condition vector for Detail DiT: timestep + class embedding
        c = self.t_embedder(t) + self.y_embedder(y, self.training, force_drop_ids)  # (B, D_detail)

        # 3. Pass through paired Patch & Global DiT blocks
        for pair_block in self.blocks:
            detail_emb, global_emb = pair_block(
                detail_emb=detail_emb,
                global_emb=global_emb,
                c=c,
                detail_grid=detail_grid,
            )

        # 4. Final modulation + unpatchify back to image space
        global_tokens_out = self.final_layer(
            global_emb=global_emb,
            detail_cond=detail_emb,
            detail_grid=detail_grid,
        )
        out = unpatchify(
            global_tokens_out,
            patch_size=self.global_patch_size,
            out_channels=self.out_channels,
            hw=(H, W),
        )
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Batched Classifier-Free Guidance (CFG) inference helper, compatible with standard DiT sampling.
        Expects `x` to have duplicated conditional & unconditional halves `(2*B, C, H, W)`.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ---------------------------------------------------------------------------
# Standard Model Configurations
# ---------------------------------------------------------------------------


def DualLevelDiT_S(**kwargs) -> DualLevelDiT:
    return DualLevelDiT(
        depth=12,
        detail_hidden_size=384,
        detail_num_heads=6,
        global_hidden_size=16,
        global_attn_dim=384,
        global_num_heads=6,
        **kwargs,
    )


def DualLevelDiT_B(**kwargs) -> DualLevelDiT:
    return DualLevelDiT(
        depth=12,
        detail_hidden_size=768,
        detail_num_heads=12,
        global_hidden_size=16,
        global_attn_dim=768,
        global_num_heads=12,
        **kwargs,
    )


def DualLevelDiT_L(**kwargs) -> DualLevelDiT:
    return DualLevelDiT(
        depth=24,
        detail_hidden_size=1024,
        detail_num_heads=16,
        global_hidden_size=16,
        global_attn_dim=1024,
        global_num_heads=16,
        **kwargs,
    )


def DualLevelDiT_XL(**kwargs) -> DualLevelDiT:
    return DualLevelDiT(
        depth=28,
        detail_hidden_size=1152,
        detail_num_heads=16,
        global_hidden_size=16,
        global_attn_dim=1152,
        global_num_heads=16,
        **kwargs,
    )


DUAL_DIT_MODELS = {
    "DualLevelDiT-S": DualLevelDiT_S,
    "DualLevelDiT-B": DualLevelDiT_B,
    "DualLevelDiT-L": DualLevelDiT_L,
    "DualLevelDiT-XL": DualLevelDiT_XL,
}


def build_model_from_config(
    model_cfg: dict,
    num_classes_override: int | None = None,
) -> DualLevelDiT:
    """
    Instantiates `DualLevelDiT` from a YAML `model` configuration dictionary.

    Supports either:
      - `preset`: one of `"DualLevelDiT-S"`, `"DualLevelDiT-B"`, `"DualLevelDiT-L"`, `"DualLevelDiT-XL"`
        with any additional keys in `model_cfg` overriding the preset defaults; or
      - Full explicit architectural hyperparameters directly in `model_cfg`.
    """
    cfg = dict(model_cfg)
    preset = cfg.pop("preset", None)

    if num_classes_override is not None:
        cfg["num_classes"] = max(int(cfg.get("num_classes", 1)), int(num_classes_override))

    if isinstance(cfg.get("img_size"), list):
        cfg["img_size"] = tuple(cfg["img_size"])

    if preset is not None:
        if preset not in DUAL_DIT_MODELS:
            raise ValueError(
                f"Unknown model preset '{preset}'. Available: {list(DUAL_DIT_MODELS.keys())}"
            )
        return DUAL_DIT_MODELS[preset](**cfg)

    return DualLevelDiT(**cfg)
