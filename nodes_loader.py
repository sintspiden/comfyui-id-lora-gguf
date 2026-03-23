"""IDLoraGGUFModelLoader — accepts a pre-loaded GGUF MODEL, applies ID-LoRA,
and creates a pipeline backed by separate component files."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import comfy.lora
import comfy.lora_convert
import comfy.utils
from comfy_api.latest import io

from .component_loader import SplitComponentLoader
from .pipeline_gguf import GGUFIDLoraOneStagePipeline


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> str:
    """Return *p* unchanged if absolute, otherwise resolve relative to repo root."""
    if not p or os.path.isabs(p):
        return p
    resolved = _REPO_ROOT / p
    return str(resolved) if resolved.exists() else p


class IDLoraGGUFModelLoader(io.ComfyNode):
    """Load a GGUF LTXAV model, apply ID-LoRA weights, and create a
    generation pipeline that uses separate VAE / text-encoder files."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraGGUFModelLoader",
            display_name="ID-LoRA GGUF Model Loader",
            category="ID-LoRA-GGUF",
            description=(
                "Takes a pre-loaded GGUF MODEL (from UnetLoaderGGUF), applies "
                "ID-LoRA weights, and creates a pipeline using separate component files."
            ),
            inputs=[
                io.Custom("MODEL").Input("model", tooltip="GGUF model from UnetLoaderGGUF."),
                io.String.Input(
                    "video_vae_path",
                    default="models/vae/LTX23_video_vae_bf16.safetensors",
                    tooltip="Path to video VAE .safetensors file.",
                ),
                io.String.Input(
                    "audio_vae_path",
                    default="models/vae/LTX23_audio_vae_bf16.safetensors",
                    tooltip="Path to audio VAE .safetensors file.",
                ),
                io.String.Input(
                    "gemma_path",
                    default="/media/me/little_monster/models/gemma",
                    tooltip="Path to Gemma text encoder directory.",
                ),
                io.String.Input(
                    "text_projection_path",
                    default="models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
                    tooltip="Path to text projection .safetensors file.",
                ),
                io.String.Input(
                    "lora_path",
                    default="",
                    tooltip="Path to ID-LoRA weights (.safetensors).",
                ),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("stg_scale", default=0.0, min=0.0, max=10.0, step=0.1,
                               tooltip="STG scale. 0 disables (recommended for v1 GGUF bridge)."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("av_bimodal_scale", default=0.0, min=0.0, max=20.0, step=0.1,
                               tooltip="AV bimodal CFG scale. 0 disables (recommended for v1)."),
            ],
            outputs=[
                io.Custom("ID_LORA_GGUF_PIPELINE").Output(
                    display_name="Pipeline",
                    tooltip="GGUF-backed ID-LoRA pipeline.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        video_vae_path: str,
        audio_vae_path: str,
        gemma_path: str,
        text_projection_path: str,
        lora_path: str,
        lora_strength: float,
        stg_scale: float,
        identity_guidance_scale: float,
        av_bimodal_scale: float,
    ) -> io.NodeOutput:
        t0 = time.time()
        print("[ID-LoRA-GGUF] === Model Loader starting ===", flush=True)
        device = torch.device("cuda")

        # Resolve relative paths
        video_vae_path = _resolve_path(video_vae_path)
        audio_vae_path = _resolve_path(audio_vae_path)
        gemma_path = _resolve_path(gemma_path)
        text_projection_path = _resolve_path(text_projection_path)

        # Clone model patcher so we don't mutate the original
        model_patcher = model.clone()

        # Apply ID-LoRA if provided
        if lora_path.strip():
            lora_path_resolved = _resolve_path(lora_path.strip())
            print(f"[ID-LoRA-GGUF] Loading LoRA: {lora_path_resolved}")
            lora_sd = comfy.utils.load_torch_file(lora_path_resolved, safe_load=True)

            # Use ComfyUI's standard LoRA loading pipeline:
            # 1. Build key map from model state dict
            # 2. Convert LoRA format (handles diffusers ↔ ComfyUI key differences)
            # 3. Load LoRA through key map to create properly-formatted patches
            # 4. Apply patches to model patcher
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
            lora_sd = comfy.lora_convert.convert_lora(lora_sd)
            loaded = comfy.lora.load_lora(lora_sd, key_map)
            if loaded:
                applied = model_patcher.add_patches(loaded, strength_patch=lora_strength)
                print(f"[ID-LoRA-GGUF] Applied {len(applied)} LoRA patches (strength={lora_strength})")
            else:
                print("[ID-LoRA-GGUF] WARNING: No LoRA keys matched. LoRA may not be applied.")

        # Find the GGUF model path (needed for extracting connector weights)
        import folder_paths
        gguf_path = None
        unet_dir = os.path.join(_REPO_ROOT, "models", "unet")
        for fname in os.listdir(unet_dir):
            if fname.endswith(".gguf") and "dev" in fname:
                gguf_path = os.path.join(unet_dir, fname)
                break
        if gguf_path is None:
            # Fallback: try any GGUF in the unet folder
            for fname in os.listdir(unet_dir):
                if fname.endswith(".gguf"):
                    gguf_path = os.path.join(unet_dir, fname)
                    break

        # Create component loader
        print(f"[ID-LoRA-GGUF] Creating component loader (GGUF: {os.path.basename(gguf_path) if gguf_path else 'none'})...", flush=True)
        component_loader = SplitComponentLoader(
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            gemma_root_path=gemma_path,
            text_projection_path=text_projection_path,
            gguf_path=gguf_path,
            dtype=torch.bfloat16,
            device=device,
        )

        # Create pipeline
        pipeline = GGUFIDLoraOneStagePipeline(
            model_patcher=model_patcher,
            component_loader=component_loader,
            device=device,
            stg_scale=stg_scale,
            identity_guidance=True,
            identity_guidance_scale=identity_guidance_scale,
            av_bimodal_cfg=(av_bimodal_scale > 0),
            av_bimodal_scale=av_bimodal_scale,
        )

        print(f"[ID-LoRA-GGUF] === Model Loader complete in {time.time()-t0:.1f}s ===", flush=True)
        return io.NodeOutput(pipeline)
