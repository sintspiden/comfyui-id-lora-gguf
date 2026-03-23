"""IDLoraGGUFPromptEncoder — encode text prompts using the split component loader."""

from __future__ import annotations

import time

import torch
import comfy.model_management
from comfy_api.latest import io

from ltx_pipelines.utils import encode_prompts, cleanup_memory
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

from .pipeline_gguf import GGUFIDLoraOneStagePipeline


class IDLoraGGUFPromptEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraGGUFPromptEncoder",
            display_name="ID-LoRA GGUF Prompt Encoder",
            category="ID-LoRA-GGUF",
            description="Encode positive and negative prompts for GGUF-backed ID-LoRA generation.",
            inputs=[
                io.Custom("ID_LORA_GGUF_PIPELINE").Input("pipeline", tooltip="GGUF ID-LoRA pipeline."),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default=DEFAULT_NEGATIVE_PROMPT,
                ),
            ],
            outputs=[
                io.Custom("ID_LORA_GGUF_CONDITIONING").Output(
                    display_name="Conditioning",
                    tooltip="Encoded video/audio conditioning tensors.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: GGUFIDLoraOneStagePipeline,
        prompt: str,
        negative_prompt: str,
    ) -> io.NodeOutput:
        device = pipeline.device

        t0 = time.time()
        # Offload the GGUF model to make room for the 23GB Gemma text encoder.
        # The sampler will reload it later.
        print("[ID-LoRA-GGUF] Offloading GGUF model for text encoding...", flush=True)
        pipeline.model_patcher.unpatch_model()
        comfy.model_management.soft_empty_cache()

        # encode_prompts uses model_ledger.text_encoder() and
        # model_ledger.gemma_embeddings_processor() — our SplitComponentLoader
        # duck-types this interface.  Internally it loads → encodes → deletes
        # each component sequentially to manage memory.
        print("[ID-LoRA-GGUF] Encoding prompts (Gemma 12B on CPU — this takes ~4 min)...", flush=True)
        results = encode_prompts(
            prompts=[prompt, negative_prompt],
            model_ledger=pipeline.model_ledger,
        )
        ctx_p, ctx_n = results
        print(f"[ID-LoRA-GGUF]   Prompt encoding complete in {time.time()-t0:.1f}s", flush=True)

        conditioning = {
            "v_context_p": ctx_p.video_encoding.to(device),
            "a_context_p": ctx_p.audio_encoding.to(device),
            "v_context_n": ctx_n.video_encoding.to(device),
            "a_context_n": ctx_n.audio_encoding.to(device),
        }
        print(f"[ID-LoRA-GGUF]   Context shapes: video={ctx_p.video_encoding.shape}, audio={ctx_p.audio_encoding.shape}", flush=True)

        cleanup_memory()
        return io.NodeOutput(conditioning)
