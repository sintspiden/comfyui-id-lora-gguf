"""IDLoraGGUFSampler — core generation: denoise + decode video + decode audio."""

from __future__ import annotations

from fractions import Fraction

import torch
from comfy_api.latest import io, Input, InputImpl, Types

from .pipeline_gguf import GGUFIDLoraOneStagePipeline, compute_resolution_match_aspect


class IDLoraGGUFSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraGGUFSampler",
            display_name="ID-LoRA GGUF Sampler",
            category="ID-LoRA-GGUF",
            description=(
                "Generate audio+video with speaker identity transfer "
                "using the GGUF-backed ID-LoRA pipeline."
            ),
            inputs=[
                io.Custom("ID_LORA_GGUF_PIPELINE").Input("pipeline"),
                io.Custom("ID_LORA_GGUF_CONDITIONING").Input("conditioning"),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("height", default=512, min=64, max=2048, step=32),
                io.Int.Input("width", default=512, min=64, max=2048, step=32),
                io.Int.Input("num_frames", default=121, min=1, max=1000, step=1),
                io.Int.Input("num_inference_steps", default=30, min=1, max=200, step=1),
                io.Float.Input("frame_rate", default=25.0, min=1.0, max=120.0, step=0.1),
                io.Float.Input("video_guidance_scale", default=3.0, min=0.0, max=30.0, step=0.1),
                io.Float.Input("audio_guidance_scale", default=7.0, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("auto_resolution", default=True,
                                 tooltip="Auto-detect resolution from first-frame aspect ratio."),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32),
                io.Image.Input("first_frame", optional=True),
                io.Audio.Input("reference_audio", optional=True),
            ],
            outputs=[
                io.Video.Output(display_name="Video"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: GGUFIDLoraOneStagePipeline,
        conditioning: dict,
        first_frame: Input.Image | None,
        reference_audio: Input.Audio | None,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        frame_rate: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        auto_resolution: bool,
        max_resolution: int,
    ) -> io.NodeOutput:
        # Load heavy models now (after prompt encoding freed the text encoder)
        if pipeline._transformer is None:
            pipeline.load_models()

        # Convert ComfyUI IMAGE [B,H,W,C] → pipeline [C,H,W]
        condition_image = None
        if first_frame is not None:
            condition_image = first_frame[0].permute(2, 0, 1)

            if auto_resolution:
                src_h, src_w = first_frame.shape[1], first_frame.shape[2]
                height, width = compute_resolution_match_aspect(src_h, src_w, max_long=max_resolution)
                print(f"[ID-LoRA-GGUF] Auto-resolution: {src_w}x{src_h} → {width}x{height}")

        # Convert ComfyUI AUDIO → pipeline [C,S]
        ref_audio = None
        ref_sr = 16000
        if reference_audio is not None:
            ref_audio = reference_audio["waveform"][0]
            ref_sr = reference_audio["sample_rate"]

        v_context_p = conditioning["v_context_p"]
        a_context_p = conditioning["a_context_p"]
        v_context_n = conditioning["v_context_n"]
        a_context_n = conditioning["a_context_n"]

        video_tensor, audio_output = pipeline(
            v_context_p=v_context_p,
            a_context_p=a_context_p,
            v_context_n=v_context_n,
            a_context_n=a_context_n,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            reference_audio=ref_audio,
            reference_audio_sample_rate=ref_sr,
            condition_image=condition_image,
        )

        # Convert outputs to ComfyUI types
        images = video_tensor.permute(1, 2, 3, 0)  # [C,F,H,W] → [F,H,W,C]
        vocoder_sr = getattr(pipeline, "_vocoder_sr", 24000)
        audio_dict = {
            "waveform": audio_output.unsqueeze(0),
            "sample_rate": vocoder_sr,
        }
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio_dict,
                frame_rate=Fraction(frame_rate),
            )
        )

        return io.NodeOutput(video)
