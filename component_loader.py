"""Split component loader — duck-types ModelLedger for separate model files.

Instead of loading everything from a single 46GB checkpoint, each component
loads from its own file: video VAE, audio VAE, text projection, Gemma encoder.
The transformer is handled separately via ComfyUI's GGUF model patcher.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import ModuleOps

# SDOps for the separate video VAE file — strips 'encoder.' / 'decoder.' prefix
# since the VideoEncoder/VideoDecoder models don't have that prefix in their
# parameter names, but the standalone safetensors file does.
_VIDEO_ENC_SD_OPS = (
    SDOps("VIDEO_ENC_SPLIT")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("encoder.", "")
)
_VIDEO_DEC_SD_OPS = (
    SDOps("VIDEO_DEC_SPLIT")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("decoder.", "")
)
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessor,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoder,
    GemmaTextEncoderConfigurator,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    module_ops_from_gemma_root,
)


class SplitComponentLoader:
    """Duck-types the ``ModelLedger`` interface used by ``encode_prompts``
    and the ID-LoRA pipeline, loading each component from its own file.

    Factory methods intentionally create a **new** model instance on every
    call (same as ``ModelLedger``) so callers can load → use → delete to
    manage GPU memory.
    """

    def __init__(
        self,
        video_vae_path: str,
        audio_vae_path: str,
        gemma_root_path: str,
        text_projection_path: str,
        gguf_path: str | None = None,
        upsampler_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        self._dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device

        self._video_vae_path = video_vae_path
        self._audio_vae_path = audio_vae_path
        self._gemma_root_path = gemma_root_path
        self._text_projection_path = text_projection_path
        self._gguf_path = gguf_path
        self._upsampler_path = upsampler_path

        # Pre-build module ops for Gemma (tokenizer + processor loaders)
        self._gemma_module_ops = module_ops_from_gemma_root(gemma_root_path)

        # Build tuple of weight paths for Gemma (Builder needs individual files,
        # not a directory — same pattern as ModelLedger.build_model_builders)
        gemma_root = Path(gemma_root_path)
        sft_files = sorted(gemma_root.rglob("model*.safetensors"))
        if not sft_files:
            sft_files = sorted(gemma_root.rglob("*.safetensors"))
        self._gemma_weight_paths = tuple(str(p) for p in sft_files)

        # Model patcher reference (set later by pipeline, needed for
        # embeddings processor to extract connector weights from GGUF)
        self._model_patcher = None

        # Read raw text_config from config.json (transformers >=5.x may
        # restructure rope_scaling and drop rope_local_base_freq from
        # Gemma3TextConfig, but ltx-core's encoder_configurator needs them
        # in the original flat format)
        import json
        config_json = gemma_root / "config.json"
        self._gemma_raw_text_config = {}
        if config_json.exists():
            with open(config_json) as f:
                raw = json.load(f)
            self._gemma_raw_text_config = raw.get("text_config", raw)

    # ------------------------------------------------------------------
    # Video VAE
    # ------------------------------------------------------------------

    def video_encoder(self) -> VideoEncoder:
        # Separate VAE file uses 'encoder.*' prefix — SDOps strips it.
        # .to(dtype) ensures registered buffers match parameter dtype.
        builder = Builder(
            model_class_configurator=VideoEncoderConfigurator,
            model_path=self._video_vae_path,
            model_sd_ops=_VIDEO_ENC_SD_OPS,
        )
        return builder.build(device=self._device, dtype=self._dtype).to(self._dtype).eval()

    def video_decoder(self) -> VideoDecoder:
        builder = Builder(
            model_class_configurator=VideoDecoderConfigurator,
            model_path=self._video_vae_path,
            model_sd_ops=_VIDEO_DEC_SD_OPS,
        )
        return builder.build(device=self._device, dtype=self._dtype).to(self._dtype).eval()

    # ------------------------------------------------------------------
    # Audio VAE + Vocoder
    # ------------------------------------------------------------------

    def audio_encoder(self) -> AudioEncoder:
        # Audio VAE file uses 'audio_vae.*' prefix — SDOps strips it.
        builder = Builder(
            model_class_configurator=AudioEncoderConfigurator,
            model_path=self._audio_vae_path,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        return builder.build(device=self._device, dtype=self._dtype).to(self._dtype).eval()

    def audio_decoder(self) -> AudioDecoder:
        builder = Builder(
            model_class_configurator=AudioDecoderConfigurator,
            model_path=self._audio_vae_path,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        )
        return builder.build(device=self._device, dtype=self._dtype).to(self._dtype).eval()

    def vocoder(self) -> Vocoder:
        builder = Builder(
            model_class_configurator=VocoderConfigurator,
            model_path=self._audio_vae_path,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        )
        return builder.build(device=self._device, dtype=self._dtype).to(self._dtype).eval()

    # ------------------------------------------------------------------
    # Text encoder (Gemma 3)
    # ------------------------------------------------------------------

    def text_encoder(self) -> GemmaTextEncoder:
        # Patch: transformers >=5.x restructures rope_scaling into nested
        # sliding_attention/full_attention dicts and may drop
        # rope_local_base_freq entirely.  ltx-core's create_and_populate
        # expects the original flat format from config.json.
        raw_tc = self._gemma_raw_text_config

        def _patch_gemma_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
            """Combined config patch + create_and_populate for transformers >=5.x.

            transformers 5.x changes:
            - rope_local_base_freq may be absent from config (restructured)
            - Gemma3TextModel has single `rotary_emb` with per-layer-type buffers
              (sliding_attention_inv_freq, full_attention_inv_freq) instead of
              separate `rotary_emb` + `rotary_emb_local` modules
            """
            from ltx_core.text_encoders.gemma.encoders.encoder_configurator import ROPE_INIT_FUNCTIONS

            model = module.model
            v_model = model.model.vision_tower.vision_model
            l_model = model.model.language_model
            cfg = model.config.text_config

            # 1. Patch config: restore values transformers 5.x restructured
            if not hasattr(cfg, "rope_local_base_freq"):
                cfg.rope_local_base_freq = raw_tc.get("rope_local_base_freq", 10000.0)
            if not hasattr(cfg, "rope_theta"):
                cfg.rope_theta = raw_tc.get("rope_theta", 1000000.0)
            rs = cfg.rope_scaling
            if isinstance(rs, dict) and "rope_type" not in rs:
                raw_rs = raw_tc.get("rope_scaling", {})
                if "rope_type" in raw_rs:
                    cfg.rope_scaling = raw_rs

            # 2. Compute rope frequencies (same as original create_and_populate)
            dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            base = cfg.rope_local_base_freq
            local_rope_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
            inv_freqs, _ = ROPE_INIT_FUNCTIONS[cfg.rope_scaling["rope_type"]](cfg)

            # 3. Register position_ids and embed_scale
            positions_length = len(v_model.embeddings.position_ids[0])
            position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
            v_model.embeddings.register_buffer("position_ids", position_ids)
            embed_scale = torch.tensor(cfg.hidden_size**0.5, device="cpu")
            l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

            # 4. Register rope inv_freq — handle both old and new transformers
            if hasattr(l_model, "rotary_emb_local"):
                # transformers <5: separate modules
                l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
                l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)
            else:
                # transformers >=5: unified Gemma3RotaryEmbedding with per-layer buffers
                rotary = l_model.rotary_emb
                rotary.register_buffer("sliding_attention_inv_freq", local_rope_freqs)
                rotary.register_buffer("full_attention_inv_freq", inv_freqs)
                # Also register the _original_inv_freq copies (used for dynamic NTK
                # scaling); without these they stay on meta and block .to(device)
                rotary.register_buffer("sliding_attention_original_inv_freq", local_rope_freqs.clone())
                rotary.register_buffer("full_attention_original_inv_freq", inv_freqs.clone())

            return module

        gemma_model_ops = ModuleOps(
            name="GemmaModelCompat",
            matcher=GEMMA_MODEL_OPS.matcher,
            mutator=_patch_gemma_and_populate,
        )

        builder = Builder(
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_path=self._gemma_weight_paths,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(gemma_model_ops,) + self._gemma_module_ops,
        )
        # Build on CPU first — Gemma 12B bf16 is ~18GB, won't fit in VRAM
        # alongside the GGUF model.  encode_prompts() will run on CPU.
        cpu = torch.device("cpu")
        return builder.build(device=cpu, dtype=self._dtype).to(cpu).eval()

    # ------------------------------------------------------------------
    # Embeddings processor (text projection)
    # ------------------------------------------------------------------

    def gemma_embeddings_processor(self, model_patcher=None) -> EmbeddingsProcessor:  # noqa: N802
        if model_patcher is None:
            model_patcher = self._model_patcher
        """Build EmbeddingsProcessor from text projection file + GGUF connector weights.

        The processor needs weights from two sources:
        - text_embedding_projection.* (from text projection file)
        - video_embeddings_connector.* / audio_embeddings_connector.* (from GGUF model)

        We merge these into a temporary safetensors file for the Builder.
        """
        import os
        import tempfile
        import safetensors.torch

        # 1. Load text projection weights
        tp_sd = safetensors.torch.load_file(self._text_projection_path, device="cpu")

        # 2. Extract connector weights from the GGUF file directly
        #    (can't use model.state_dict() because GGUF tensors are on meta device)
        if self._gguf_path is not None:
            tp_sd = self._extract_gguf_connector_weights(tp_sd)

        # 3. Read config from text projection metadata
        meta = {}
        with safetensors.safe_open(self._text_projection_path, framework="pt") as f:
            raw_meta = f.metadata()
            if raw_meta:
                meta = dict(raw_meta)

        # 4. Write merged safetensors to temp file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
        safetensors.torch.save_file(tp_sd, tmp_path, metadata=meta)

        try:
            builder = Builder(
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_path=tmp_path,
                model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            )
            # Build on CPU — the text encoder outputs are on CPU (Gemma is too
            # large for VRAM alongside the GGUF model), so the embeddings
            # processor must also run on CPU.  The prompt encoder node moves
            # the final results to CUDA.
            cpu = torch.device("cpu")
            return builder.build(device=cpu, dtype=self._dtype).to(cpu).eval()
        finally:
            os.unlink(tmp_path)

    def _extract_gguf_connector_weights(self, sd: dict) -> dict:
        """Extract embeddings connector weights from GGUF file and add to state dict.

        Uses ComfyUI-GGUF's loader to properly handle quantized tensors, then
        dequantizes connector weights to bf16 for the Builder.

        GGUF keys:  video_embeddings_connector.*, audio_embeddings_connector.*
        SDOps expects: model.diffusion_model.video_embeddings_connector.*, etc.
        """
        import sys
        import importlib

        # Import dequantization from ComfyUI-GGUF
        gguf_node_path = str(Path(__file__).resolve().parents[1] / "ComfyUI-GGUF")
        if gguf_node_path not in sys.path:
            sys.path.insert(0, gguf_node_path)

        from gguf import GGUFReader, GGMLQuantizationType
        dequant_mod = importlib.import_module("dequant", package=None)
        # Fallback: import from ComfyUI-GGUF directly
        try:
            from custom_nodes import ComfyUI_GGUF  # noqa: N811
            dequantize_tensor = ComfyUI_GGUF.ops.dequantize_tensor
        except (ImportError, AttributeError):
            dequantize_tensor = dequant_mod.dequantize_tensor

        reader = GGUFReader(self._gguf_path)
        count = 0

        # F32 type IDs that can be directly converted
        F32_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}

        for tensor_info in reader.tensors:
            if "embeddings_connector" not in tensor_info.name:
                continue

            shape = tuple(reversed(tensor_info.shape.tolist()))
            qtype = GGMLQuantizationType(tensor_info.tensor_type)

            if qtype in F32_TYPES:
                # Simple conversion for unquantized types
                import numpy as np
                data = np.array(tensor_info.data, copy=True)
                t = torch.from_numpy(data).reshape(shape)
            else:
                # For quantized types, create GGMLTensor and dequantize
                import numpy as np
                data = np.array(tensor_info.data, copy=True)
                raw = torch.from_numpy(data)
                # Dequantize using ComfyUI-GGUF's dequant functions
                t = dequant_mod.dequantize(raw, qtype, shape, dtype=torch.bfloat16)

            # Add with the prefix that EMBEDDINGS_PROCESSOR_KEY_OPS expects
            key = f"model.diffusion_model.{tensor_info.name}"
            sd[key] = t.to(torch.bfloat16)
            count += 1

        if sys.path[0] == gguf_node_path:
            sys.path.pop(0)

        print(f"[ID-LoRA-GGUF] Extracted {count} connector weights from GGUF")
        return sd

    # ------------------------------------------------------------------
    # Spatial upsampler (optional, for two-stage)
    # ------------------------------------------------------------------

    def spatial_upsampler(self) -> LatentUpsampler:
        if self._upsampler_path is None:
            raise RuntimeError("No upsampler path provided")
        builder = Builder(
            model_class_configurator=LatentUpsamplerConfigurator,
            model_path=self._upsampler_path,
        )
        return builder.build(device=self._device, dtype=self._dtype)
