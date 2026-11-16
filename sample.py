"""
Standalone Inference / Sampling Script for Dual-Level DiT.

Example:
  python sample.py \
    --ckpt outputs/dual_dit_dog/checkpoints/latest.pt \
    --prompt "小狗" \
    --out outputs/dual_dit_dog/result_dog.jpeg
"""

from __future__ import annotations

import argparse

import torch

from dataset import PromptVocab, normalize_path, save_tensor_image
from diffusion import RectifiedFlow
from models import build_model_from_config
from train import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from trained Dual-Level DiT")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/dual_dit_dog/checkpoints/latest.pt",
        help="Path to saved checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path (defaults to config stored inside checkpoint).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="小狗",
        help="Text prompt to generate (must exist in prompt_vocab).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/dual_dit_dog/sampled_dog.jpeg",
        help="Output image file path.",
    )
    parser.add_argument("--num_steps", type=int, default=None, help="Euler ODE steps.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use_raw_weights",
        action="store_true",
        help="Use raw training weights instead of EMA weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = normalize_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = load_yaml_config(args.config) if args.config else ckpt["config"]
    vocab_list = ckpt.get("prompt_vocab") or cfg.get("data", {}).get("prompt_vocab", ["小狗"])
    vocab = PromptVocab(vocab_list)

    model = build_model_from_config(
        cfg["model"], num_classes_override=len(vocab)
    ).to(device)
    state_key = "model" if args.use_raw_weights else "ema"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    diff_cfg = cfg.get("diffusion", {})
    sample_cfg = cfg.get("sample", {})

    flow = RectifiedFlow(
        timestep_scale=float(diff_cfg.get("timestep_scale", 1000.0)),
        timestep_sampling=str(diff_cfg.get("timestep_sampling", "logit_normal")),
        logit_mean=float(diff_cfg.get("logit_mean", 0.0)),
        logit_std=float(diff_cfg.get("logit_std", 1.0)),
    )

    y_cond = vocab.encode_batch([args.prompt], device=device)
    img_h, img_w = model.detail_tokenizer.img_size
    shape = (1, model.in_channels, img_h, img_w)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    null_idx = model.num_classes if model.y_embedder.dropout_prob > 0.0 else None
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else None
    )

    samples = flow.sample(
        model=model,
        shape=shape,
        y_cond=y_cond,
        num_steps=args.num_steps or int(sample_cfg.get("num_steps", 50)),
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else float(sample_cfg.get("cfg_scale", 1.5)),
        null_class_idx=null_idx,
        generator=gen,
        autocast_dtype=amp_dtype,
    )

    saved = save_tensor_image(samples, args.out)
    print(f"[Done] Prompt='{args.prompt}' -> saved image to {saved}")


if __name__ == "__main__":
    main()
