"""
Distributed Training Script (`torchrun` DDP) for Dual-Level DiT controlled via YAML.

Example usage:
  Single/Multi-GPU via `run.sh`:
    bash run.sh
  Direct `torchrun`:
    torchrun --nproc_per_node=1 train.py --config configs/train_dog.yaml
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PromptImageDataset, normalize_path, save_tensor_image
from diffusion import RectifiedFlow
from models import DualLevelDiT, build_model_from_config


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    """
    Initializes `torch.distributed` when launched with `torchrun`.
    Also supports single-process execution when `RANK` is not set.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        return True, rank, local_rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, device


def set_seed(seed: int, rank: int = 0) -> None:
    actual_seed = seed + rank
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


@torch.no_grad()
def update_ema(ema_model: DualLevelDiT, model: DualLevelDiT, decay: float, step: int) -> None:
    """
    Updates EMA parameters with warmup-scheduled decay `min(decay, (1 + step) / (10 + step))`
    so early training steps converge quickly while later steps remain smooth.
    """
    effective_decay = min(decay, (1.0 + step) / (10.0 + step))
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(effective_decay).add_(param.data, alpha=1.0 - effective_decay)


def compute_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if schedule == "constant" or total_steps <= warmup_steps:
        return base_lr

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def resolve_amp_dtype(
    mixed_precision: str, device: torch.device, is_main: bool
) -> tuple[torch.dtype | None, bool]:
    """
    Returns `(autocast_dtype, use_grad_scaler)`.
    """
    mp = mixed_precision.lower()
    if device.type != "cuda" or mp in ("no", "fp32", "none"):
        return None, False

    if mp == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        if is_main:
            print("[Warn] CUDA device does not support bf16; falling back to fp16.")
        return torch.float16, True

    if mp == "fp16":
        return torch.float16, True

    raise ValueError(f"Unsupported mixed_precision '{mixed_precision}'.")


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = normalize_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dual-Level DiT with YAML & torchrun")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_dog.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Optional override for train.total_steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for experiment.output_dir.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path override to resume training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    if args.total_steps is not None:
        cfg.setdefault("train", {})["total_steps"] = args.total_steps
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.resume is not None:
        cfg.setdefault("train", {})["resume_from"] = args.resume

    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    try:
        exp_cfg = cfg.get("experiment", {})
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        diff_cfg = cfg.get("diffusion", {})
        train_cfg = cfg.get("train", {})
        sample_cfg = cfg.get("sample", {})

        seed = int(exp_cfg.get("seed", 42))
        set_seed(seed, rank=rank)

        output_dir = normalize_path(exp_cfg.get("output_dir", "outputs/dual_dit_dog"))
        ckpt_dir = output_dir / "checkpoints"
        sample_dir = output_dir / "samples"
        if is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # 1. Dataset & DataLoader
        img_size = model_cfg.get("img_size", 256)
        dataset = PromptImageDataset(data_cfg=data_cfg, img_size=img_size)
        sampler = (
            DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
            )
            if is_distributed
            else None
        )
        batch_size = int(train_cfg.get("batch_size_per_gpu", 4))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        if is_main:
            ref_img, _, ref_prompt = dataset[0]
            save_tensor_image(ref_img, output_dir / "target_dog.jpeg")
            print(
                f"[Init] World size: {world_size} | Device: {device} | "
                f"Prompt vocab: {dataset.vocab.to_list()} | Target saved: {output_dir / 'target_dog.jpeg'}"
            )

        # 2. Model & EMA
        raw_model = build_model_from_config(
            model_cfg, num_classes_override=len(dataset.vocab)
        ).to(device)
        ema_model = copy.deepcopy(raw_model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        if is_main:
            total_params = sum(p.numel() for p in raw_model.parameters())
            print(f"[Model] DualLevelDiT parameters: {total_params / 1e6:.2f}M")

        # 3. Optimizer & AMP Scaler
        base_lr = float(train_cfg.get("lr", 2e-4))
        min_lr = float(train_cfg.get("min_lr", 1e-5))
        lr_schedule = str(train_cfg.get("lr_scheduler", "cosine"))
        warmup_steps = int(train_cfg.get("warmup_steps", 100))
        total_steps = int(train_cfg.get("total_steps", 2000))
        grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        ema_decay = float(train_cfg.get("ema_decay", 0.995))

        betas_cfg = train_cfg.get("betas", [0.9, 0.999])
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=base_lr,
            betas=(float(betas_cfg[0]), float(betas_cfg[1])),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        amp_dtype, use_scaler = resolve_amp_dtype(
            str(train_cfg.get("mixed_precision", "bf16")), device=device, is_main=is_main
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        # Optional checkpoint resume
        start_step = 0
        resume_path = train_cfg.get("resume_from")
        if resume_path:
            ckpt_file = normalize_path(resume_path)
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model"])
            ema_model.load_state_dict(ckpt["ema"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0))
            if is_main:
                print(f"[Resume] Loaded checkpoint from {ckpt_file} at step {start_step}.")

        model: torch.nn.Module = (
            DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None)
            if is_distributed
            else raw_model
        )

        # 4. Rectified Flow Trainer / Sampler
        flow = RectifiedFlow(
            timestep_scale=float(diff_cfg.get("timestep_scale", 1000.0)),
            timestep_sampling=str(diff_cfg.get("timestep_sampling", "logit_normal")),
            logit_mean=float(diff_cfg.get("logit_mean", 0.0)),
            logit_std=float(diff_cfg.get("logit_std", 1.0)),
        )

        log_every = int(train_cfg.get("log_every", 20))
        sample_every = int(train_cfg.get("sample_every", 200))
        ckpt_every = int(train_cfg.get("ckpt_every", 500))

        def run_validation_sample(curr_step: int) -> None:
            if is_main:
                ema_model.eval()
                val_prompt = str(sample_cfg.get("prompt", data_cfg.get("prompt", "小狗")))
                num_samples = int(sample_cfg.get("num_samples", 1))
                y_cond = dataset.vocab.encode_batch(
                    [val_prompt] * num_samples, device=device
                )
                img_h, img_w = raw_model.detail_tokenizer.img_size
                shape = (num_samples, raw_model.in_channels, img_h, img_w)

                gen = torch.Generator(device=device)
                gen.manual_seed(int(sample_cfg.get("seed", seed)))

                null_idx = (
                    raw_model.num_classes
                    if raw_model.y_embedder.dropout_prob > 0.0
                    else None
                )
                samples = flow.sample(
                    model=ema_model,
                    shape=shape,
                    y_cond=y_cond,
                    num_steps=int(sample_cfg.get("num_steps", 50)),
                    cfg_scale=float(sample_cfg.get("cfg_scale", 1.5)),
                    null_class_idx=null_idx,
                    generator=gen,
                    autocast_dtype=amp_dtype,
                )
                step_img_path = save_tensor_image(
                    samples, sample_dir / f"step_{curr_step:06d}.jpeg"
                )
                latest_img_path = save_tensor_image(
                    samples, output_dir / "result_dog.jpeg"
                )
                print(
                    f"[Sample] Step {curr_step:06d} | Prompt='{val_prompt}' -> "
                    f"{step_img_path} (latest: {latest_img_path})"
                )

            if is_distributed:
                dist.barrier()

        def save_checkpoint(curr_step: int, is_final: bool = False) -> None:
            if is_main:
                payload = {
                    "step": curr_step,
                    "model": raw_model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "prompt_vocab": dataset.vocab.to_list(),
                }
                torch.save(payload, ckpt_dir / "latest.pt")
                if not is_final:
                    torch.save(payload, ckpt_dir / f"step_{curr_step:06d}.pt")
                print(f"[Ckpt] Saved checkpoint at step {curr_step:06d} -> {ckpt_dir}")

            if is_distributed:
                dist.barrier()

        # 5. Main Training Loop
        model.train()
        epoch = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(loader)

        running_loss = 0.0
        t0 = time.perf_counter()

        for step in range(start_step, total_steps):
            lr_now = compute_lr(
                step=step,
                total_steps=total_steps,
                base_lr=base_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                schedule=lr_schedule,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            step_loss_val = 0.0

            for micro in range(grad_accum_steps):
                try:
                    x_0, y_id, _ = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    data_iter = iter(loader)
                    x_0, y_id, _ = next(data_iter)

                x_0 = x_0.to(device, non_blocking=True)
                y_id = y_id.to(device, non_blocking=True)

                sync_ctx = (
                    model.no_sync()
                    if (is_distributed and micro < grad_accum_steps - 1)
                    else nullcontext()
                )
                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                    if (amp_dtype is not None and device.type == "cuda")
                    else nullcontext()
                )

                with sync_ctx:
                    with amp_ctx:
                        loss = flow.compute_loss(model, x_0, y_id) / grad_accum_steps

                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                step_loss_val += loss.detach().item()

            if use_scaler:
                scaler.unscale_(optimizer)
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            update_ema(ema_model, raw_model, decay=ema_decay, step=step)
            running_loss += step_loss_val

            curr_step = step + 1
            if curr_step % log_every == 0 or curr_step == 1:
                avg_loss = running_loss / (1 if curr_step == 1 else log_every)
                running_loss = 0.0

                loss_tensor = torch.tensor(avg_loss, device=device)
                if is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                if is_main:
                    elapsed = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    print(
                        f"[Train] Step {curr_step:06d}/{total_steps:06d} | "
                        f"Loss: {loss_tensor.item():.6f} | LR: {lr_now:.2e} | "
                        f"Time: {elapsed:.2f}s"
                    )

            if sample_every > 0 and curr_step % sample_every == 0:
                run_validation_sample(curr_step)
                model.train()

            if ckpt_every > 0 and curr_step % ckpt_every == 0:
                save_checkpoint(curr_step)

        # Final sample & checkpoint
        run_validation_sample(total_steps)
        save_checkpoint(total_steps, is_final=True)

    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
