"""IDLoraGGUFPatcher — applies ID-LoRA weights to a GGUF model.

Returns a patched MODEL that can be used with standard ComfyUI nodes
(MultimodalGuider, KSampler, etc.) for lip-synced audio+video generation.

This is the lightweight alternative to IDLoraGGUFModelLoader — instead of
a custom pipeline with its own denoising loop, the patched model goes
through ComfyUI's standard sampling path which handles cross-modal
attention and lip sync properly via MultimodalGuider.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import comfy.lora
import comfy.lora_convert
import comfy.utils
from comfy_api.latest import io


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> str:
    """Return *p* unchanged if absolute, otherwise resolve relative to repo root."""
    if not p or os.path.isabs(p):
        return p
    resolved = _REPO_ROOT / p
    return str(resolved) if resolved.exists() else p


class IDLoraGGUFPatcher(io.ComfyNode):
    """Apply ID-LoRA weights to a GGUF model for identity-preserving generation.

    Use with the standard MultimodalGuider workflow for lip-synced output.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraGGUFPatcher",
            display_name="ID-LoRA GGUF Patcher",
            category="ID-LoRA-GGUF",
            description=(
                "Applies ID-LoRA weights to a GGUF model. Returns a patched MODEL "
                "for use with standard ComfyUI sampling nodes (MultimodalGuider, etc.)."
            ),
            inputs=[
                io.Custom("MODEL").Input("model", tooltip="GGUF model from UnetLoaderGGUF."),
                io.String.Input(
                    "lora_path",
                    default="",
                    tooltip="Path to ID-LoRA weights (.safetensors).",
                ),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05),
            ],
            outputs=[
                io.Custom("MODEL").Output(display_name="MODEL"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        lora_path: str,
        lora_strength: float,
    ) -> io.NodeOutput:
        t0 = time.time()
        print("[ID-LoRA-GGUF] === Patcher starting ===", flush=True)

        # Clone model patcher so we don't mutate the original
        model_patcher = model.clone()

        if not lora_path.strip():
            print("[ID-LoRA-GGUF] WARNING: No LoRA path provided, returning unpatched model.", flush=True)
            return io.NodeOutput(model_patcher)

        lora_path_resolved = _resolve_path(lora_path.strip())
        print(f"[ID-LoRA-GGUF] Loading LoRA: {lora_path_resolved}", flush=True)
        lora_sd = comfy.utils.load_torch_file(lora_path_resolved, safe_load=True)

        # ComfyUI standard LoRA loading pipeline:
        # 1. Build key map from model state dict
        # 2. Convert LoRA format (handles diffusers <-> ComfyUI key differences)
        # 3. Load LoRA through key map to create properly-formatted patches
        # 4. Apply patches to model patcher
        key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
        lora_sd = comfy.lora_convert.convert_lora(lora_sd)
        loaded = comfy.lora.load_lora(lora_sd, key_map)
        if loaded:
            applied = model_patcher.add_patches(loaded, strength_patch=lora_strength)
            print(f"[ID-LoRA-GGUF] Applied {len(applied)} LoRA patches (strength={lora_strength})", flush=True)
        else:
            print("[ID-LoRA-GGUF] WARNING: No LoRA keys matched. LoRA may not be applied.", flush=True)

        print(f"[ID-LoRA-GGUF] === Patcher complete in {time.time()-t0:.1f}s ===", flush=True)
        return io.NodeOutput(model_patcher)
