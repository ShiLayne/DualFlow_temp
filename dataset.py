"""
Prompt Vocabulary Mapper and Image Dataset for Dual-Level DiT training.

Supports YAML-configured prompt-to-image pairs (e.g. `prompt: "小狗"` -> `data/dog.jpeg`),
cross-platform path normalization (`\\` and `/`), aspect-preserving center cropping,
and virtual epoch repetition for multi-GPU `torchrun` DistributedSampler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def normalize_path(path_str: str | Path) -> Path:
    """
    Normalizes Windows (`\\`) and POSIX (`/`) path separators so YAML configs
    work identically across Linux (`run.sh`) and Windows environments.
    """
    return Path(str(path_str).replace("\\", "/"))


class PromptVocab:
    """
    Deterministic mapping between text prompts (e.g., `"小狗"`) and integer class IDs
    consumed by `DualLevelDiT.y_embedder` (`LabelEmbedder`).
    """

    def __init__(self, prompts: list[str]) -> None:
        unique_prompts: list[str] = []
        for p in prompts:
            p_clean = str(p).strip()
            if p_clean and p_clean not in unique_prompts:
                unique_prompts.append(p_clean)

        if not unique_prompts:
            raise ValueError("PromptVocab requires at least one non-empty prompt.")

        self.prompts = unique_prompts
        self.prompt_to_id = {p: idx for idx, p in enumerate(self.prompts)}
        self.id_to_prompt = {idx: p for idx, p in enumerate(self.prompts)}

    def __len__(self) -> int:
        return len(self.prompts)

    def encode(self, prompt: str) -> int:
        key = prompt.strip()
        if key not in self.prompt_to_id:
            raise KeyError(
                f"Prompt '{key}' not found in prompt_vocab: {self.prompts}"
            )
        return self.prompt_to_id[key]

    def encode_batch(
        self, prompts: list[str], device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        ids = [self.encode(p) for p in prompts]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode(self, idx: int) -> str:
        return self.id_to_prompt[int(idx)]

    def to_list(self) -> list[str]:
        return list(self.prompts)


def load_and_preprocess_image(
    image_path: str | Path,
    img_size: int | tuple[int, int] = 256,
    center_crop: bool = True,
) -> torch.Tensor:
    """
    Loads an RGB image from disk, resizes / center-crops to `(H, W)`, and normalizes
    pixel values from `[0, 255]` to `[-1.0, 1.0]`.
    """
    path = normalize_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target image not found at: {path.resolve()}")

    if isinstance(img_size, int):
        target_h, target_w = img_size, img_size
    else:
        target_h, target_w = int(img_size[0]), int(img_size[1])

    pil_img = Image.open(path).convert("RGB")

    if center_crop:
        # Scale so the shorter side matches target size, then center-crop
        w, h = pil_img.size
        scale = max(target_h / h, target_w / w)
        new_h = max(target_h, int(round(h * scale)))
        new_w = max(target_w, int(round(w * scale)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        pil_img = pil_img.crop((left, top, left + target_w, top + target_h))
    else:
        pil_img = pil_img.resize(
            (target_w, target_h), resample=Image.Resampling.BICUBIC
        )

    arr = np.asarray(pil_img, dtype=np.float32) / 127.5 - 1.0  # (H, W, 3) in [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    return tensor


def save_tensor_image(tensor: torch.Tensor, save_path: str | Path) -> Path:
    """
    Saves a `(3, H, W)` or `(B, 3, H, W)` image tensor in `[-1, 1]` to disk.
    If `B > 1`, tiles images horizontally into a single strip.
    """
    out_path = normalize_path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = tensor.detach().cpu().float().clamp(-1.0, 1.0)
    if x.ndim == 4:
        if x.shape[0] == 1:
            x = x[0]
        else:
            # Concatenate batch horizontally: (B, C, H, W) -> (C, H, B * W)
            b, c, h, w = x.shape
            x = x.permute(1, 2, 0, 3).reshape(c, h, b * w)

    arr = ((x.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)
    return out_path


class PromptImageDataset(Dataset):
    """
    Dataset of `(image_tensor, class_id, prompt_str)` built from YAML `data` config.

    Supports either:
      1. Single target mode:
         `data.prompt: "小狗"`, `data.image_path: "data/dog.jpeg"`
      2. Multi-sample list mode:
         `data.samples: [{prompt: "小狗", image_path: "data/dog.jpeg"}, ...]`
    """

    def __init__(
        self,
        data_cfg: dict[str, Any],
        img_size: int | tuple[int, int] = 256,
    ) -> None:
        super().__init__()
        raw_samples: list[dict[str, str]] = []
        if "samples" in data_cfg and data_cfg["samples"]:
            for item in data_cfg["samples"]:
                raw_samples.append(
                    {
                        "prompt": str(item["prompt"]),
                        "image_path": str(item["image_path"]),
                    }
                )
        elif "prompt" in data_cfg and "image_path" in data_cfg:
            raw_samples.append(
                {
                    "prompt": str(data_cfg["prompt"]),
                    "image_path": str(data_cfg["image_path"]),
                }
            )
        else:
            raise ValueError(
                "data config must specify either (`prompt` + `image_path`) or `samples`."
            )

        vocab_list = list(data_cfg.get("prompt_vocab", []))
        for s in raw_samples:
            if s["prompt"] not in vocab_list:
                vocab_list.append(s["prompt"])

        self.vocab = PromptVocab(vocab_list)
        self.img_size = (
            tuple(img_size) if isinstance(img_size, list) else img_size
        )
        self.center_crop = bool(data_cfg.get("center_crop", True))
        self.repeat_per_epoch = max(1, int(data_cfg.get("repeat_per_epoch", 64)))
        self.samples = raw_samples

        # Cache preprocessed tensors in memory when dataset has few unique images
        self._cache: list[tuple[torch.Tensor, int, str]] | None = None
        if len(self.samples) <= 256:
            self._cache = []
            for s in self.samples:
                img_tensor = load_and_preprocess_image(
                    s["image_path"],
                    img_size=self.img_size,
                    center_crop=self.center_crop,
                )
                label_id = self.vocab.encode(s["prompt"])
                self._cache.append((img_tensor, label_id, s["prompt"]))

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        base_idx = index % len(self.samples)
        if self._cache is not None:
            img_tensor, label_id, prompt = self._cache[base_idx]
            return img_tensor.clone(), label_id, prompt

        s = self.samples[base_idx]
        img_tensor = load_and_preprocess_image(
            s["image_path"],
            img_size=self.img_size,
            center_crop=self.center_crop,
        )
        label_id = self.vocab.encode(s["prompt"])
        return img_tensor, label_id, s["prompt"]
