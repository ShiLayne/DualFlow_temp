"""
Rectified Flow / Flow Matching training loss and Euler ODE sampler for Dual-Level DiT.

Formulation:
  - Clean data: `x_0` in `[-1, 1]`
  - Prior noise: `x_1 ~ N(0, I)`
  - Interpolation (`t in [0, 1]`): `x_t = (1 - t) * x_0 + t * x_1`
  - Target velocity: `v_target = dx_t / dt = x_1 - x_0`
  - Reverse sampling (`t: 1 -> 0`): `x_{t + dt} = x_t + dt * v_theta(x_t, t * scale, y)` with `dt < 0`
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class RectifiedFlow:
    """
    Continuous-time Rectified Flow / Flow Matching trainer & Euler ODE sampler.
    """

    def __init__(
        self,
        timestep_scale: float = 1000.0,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        if timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                f"Unsupported timestep_sampling '{timestep_sampling}'. "
                "Choose 'uniform' or 'logit_normal'."
            )
        self.timestep_scale = timestep_scale
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Samples continuous timesteps `t` in `(0, 1)` of shape `(B,)`.
        """
        if self.timestep_sampling == "logit_normal":
            u = (
                torch.randn(batch_size, device=device, dtype=torch.float32)
                * self.logit_std
                + self.logit_mean
            )
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device, dtype=torch.float32)

        return t.clamp(1e-5, 1.0 - 1e-5)

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes Flow Matching velocity MSE loss on a batch `(x_0, y)`.

        Args:
            model: `DualLevelDiT` (or DDP-wrapped `DualLevelDiT`).
            x_0: Clean images `(B, C, H, W)` normalized in `[-1, 1]`.
            y: Class/prompt IDs `(B,)`.
            noise: Optional Gaussian noise `(B, C, H, W)`.

        Returns:
            Scalar MSE loss tensor.
        """
        b, c, _, _ = x_0.shape
        x_1 = torch.randn_like(x_0) if noise is None else noise
        t = self.sample_timesteps(b, device=x_0.device)

        t_expand = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0

        v_pred = model(x_t, t * self.timestep_scale, y)
        if v_pred.shape[1] > c:
            v_pred = v_pred[:, :c]

        return F.mse_loss(v_pred.float(), v_target.float())

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        y_cond: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        null_class_idx: int | None = None,
        generator: torch.Generator | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Generates images from pure Gaussian noise (`t = 1 -> 0`) via Euler ODE integration.

        Args:
            model: Unwrapped `DualLevelDiT` in eval mode.
            shape: `(B, C, H, W)` target output tensor shape.
            y_cond: Conditional prompt class IDs `(B,)`.
            num_steps: Number of Euler integration steps.
            cfg_scale: Classifier-Free Guidance scale (`1.0` disables CFG).
            null_class_idx: Unconditional class token index (`num_classes` when `class_dropout_prob > 0`).
            generator: Optional RNG generator for reproducible noise initialization.
            autocast_dtype: Optional AMP dtype (`torch.bfloat16` or `torch.float16`).

        Returns:
            Sampled images `(B, C, H, W)` clamped to `[-1, 1]`.
        """
        device = y_cond.device
        b, c, _, _ = shape

        x = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)

        use_cfg = cfg_scale > 1.0 and null_class_idx is not None
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
            if (autocast_dtype is not None and device.type == "cuda")
            else nullcontext()
        )

        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # negative step size (-1 / num_steps)

            t_batch = torch.full(
                (b,), t_curr * self.timestep_scale, device=device, dtype=torch.float32
            )

            with amp_ctx:
                if use_cfg:
                    x_in = torch.cat([x, x], dim=0)
                    t_in = torch.cat([t_batch, t_batch], dim=0)
                    y_null = torch.full_like(y_cond, null_class_idx)
                    y_in = torch.cat([y_cond, y_null], dim=0)

                    v_both = model(x_in, t_in, y_in)[:, :c].float()
                    v_cond, v_uncond = v_both.chunk(2, dim=0)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(x, t_batch, y_cond)[:, :c].float()

            x = x + dt * v

        return x.clamp(-1.0, 1.0)
