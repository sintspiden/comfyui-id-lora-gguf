"""GGUF-backed ID-LoRA one-stage pipeline.

Reuses the guidance logic (CFG, identity guidance, STG, bimodal) from the
original ``pipeline_wrapper.py`` but replaces the ltx-core ``X0Model``
transformer with ``GGUFTransformerBridge`` which routes through ComfyUI's
GGUF-dequantised model.

The component loader (``SplitComponentLoader``) provides VAEs, text encoder,
and embeddings processor from separate files — no monolithic 46 GB checkpoint.
"""

from __future__ import annotations

import gc
import time

import torch

import comfy.model_management

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from ltx_core.model.audio_vae import AudioProcessor, decode_audio as vae_decode_audio
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape

from ltx_pipelines.utils import cleanup_memory, euler_denoising_loop
from ltx_pipelines.utils.helpers import modality_from_latent_state

from .component_loader import SplitComponentLoader
from .gguf_bridge import GGUFTransformerBridge


# ---------------------------------------------------------------------------
# Resolution helpers (duplicated from original pipeline_wrapper for isolation)
# ---------------------------------------------------------------------------

RESOLUTION_DIVISOR = 32
MAX_LONG_SIDE = 512
MAX_PIXELS = 576 * 1024


def snap_to_divisor(value: int, divisor: int = RESOLUTION_DIVISOR) -> int:
    return max(int(round(value / divisor)) * divisor, divisor)


def compute_resolution_match_aspect(
    src_h: int, src_w: int,
    max_long: int = MAX_LONG_SIDE,
    max_pixels: int = MAX_PIXELS,
    divisor: int = RESOLUTION_DIVISOR,
) -> tuple[int, int]:
    scale = max_long / max(src_h, src_w)
    pixel_scale = (max_pixels / (src_h * src_w)) ** 0.5
    scale = min(scale, pixel_scale)
    return (
        snap_to_divisor(int(round(src_h * scale)), divisor),
        snap_to_divisor(int(round(src_w * scale)), divisor),
    )


# ---------------------------------------------------------------------------
# Shared base (cherry-picked from original _IDLoraBase)
# ---------------------------------------------------------------------------

class _GGUFIDLoraBase:
    """Shared helpers for the GGUF-backed ID-LoRA pipeline."""

    dtype: torch.dtype
    device: torch.device
    _stg_scale: float
    _stg_blocks: list[int]
    _stg_mode: str
    _av_bimodal_cfg: bool
    _av_bimodal_scale: float
    _video_patchifier: VideoLatentPatchifier
    _audio_patchifier: AudioPatchifier

    # -- guidance configs --------------------------------------------------

    def _stg_config(self) -> BatchedPerturbationConfig:
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=self._stg_blocks)
        ]
        if self._stg_mode == "stg_av":
            perturbations.append(
                Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=self._stg_blocks)
            )
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])

    def _av_bimodal_config(self) -> BatchedPerturbationConfig:
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=[
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ])])

    # -- state creation helpers -------------------------------------------

    def _create_video_state(
        self,
        output_shape: VideoPixelShape,
        condition_image: torch.Tensor | None,
        noiser: GaussianNoiser,
        frame_rate: float,
    ) -> tuple[LatentState, VideoLatentTools, int]:
        video_tools = VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(output_shape),
            fps=frame_rate,
        )
        target_state = video_tools.create_initial_state(device=self.device, dtype=self.dtype)

        if condition_image is not None:
            target_state = self._apply_image_conditioning(target_state, condition_image, output_shape)

        video_state = noiser(latent_state=target_state, noise_scale=1.0)
        return video_state, video_tools, 0

    def _create_audio_state(
        self,
        output_shape: VideoPixelShape,
        reference_audio: torch.Tensor | None,
        reference_audio_sample_rate: int,
        noiser: GaussianNoiser,
    ) -> tuple[LatentState, AudioLatentTools, int]:
        duration = output_shape.frames / output_shape.fps
        audio_tools = AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=duration),
        )
        target_state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)
        ref_seq_len = 0

        if reference_audio is not None:
            ref_latent, ref_pos = self._encode_audio(reference_audio, reference_audio_sample_rate)
            ref_seq_len = ref_latent.shape[1]

            hop_length = 160
            downsample = 4
            sr = 16000
            time_per_latent = hop_length * downsample / sr
            aud_dur = ref_pos[:, :, -1, 1].max().item()
            ref_pos = ref_pos - aud_dur - time_per_latent

            ref_mask = torch.zeros(1, ref_seq_len, 1, device=self.device, dtype=torch.float32)
            combined = LatentState(
                latent=torch.cat([ref_latent, target_state.latent], dim=1),
                denoise_mask=torch.cat([ref_mask, target_state.denoise_mask], dim=1),
                positions=torch.cat([ref_pos, target_state.positions], dim=2),
                clean_latent=torch.cat([ref_latent, target_state.clean_latent], dim=1),
            )
            audio_state = noiser(latent_state=combined, noise_scale=1.0)
        else:
            audio_state = noiser(latent_state=target_state, noise_scale=1.0)

        return audio_state, audio_tools, ref_seq_len

    # -- image / audio conditioning helpers --------------------------------

    @staticmethod
    def _center_crop_resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        import torch.nn.functional as F
        src_h, src_w = image.shape[1], image.shape[2]
        img = image.unsqueeze(0)
        if src_h != height or src_w != width:
            ar, tar = src_w / src_h, width / height
            rh, rw = (height, int(height * ar)) if ar > tar else (int(width / ar), width)
            img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
            h0, w0 = (rh - height) // 2, (rw - width) // 2
            img = img[:, :, h0:h0 + height, w0:w0 + width]
        return img

    def _apply_image_conditioning(
        self, video_state: LatentState, image: torch.Tensor, output_shape: VideoPixelShape
    ) -> LatentState:
        image = self._center_crop_resize(image, output_shape.height, output_shape.width)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(2).to(device=self.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = self._video_encoder(image)
        patchified = self._video_patchifier.patchify(encoded)
        n = patchified.shape[1]

        new_latent = video_state.latent.clone()
        new_latent[:, :n] = patchified.to(new_latent.dtype)
        new_clean = video_state.clean_latent.clone()
        new_clean[:, :n] = patchified.to(new_clean.dtype)
        new_mask = video_state.denoise_mask.clone()
        new_mask[:, :n] = 0.0
        return LatentState(
            latent=new_latent, denoise_mask=new_mask,
            positions=video_state.positions, clean_latent=new_clean,
        )

    def _encode_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ltx_core.types import Audio
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        mel = self._audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sample_rate))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_raw = self._audio_encoder(mel.to(torch.float32))
        latent_raw = latent_raw.to(self.dtype)
        B, C, T, Fq = latent_raw.shape
        latent = self._audio_patchifier.patchify(latent_raw)
        positions = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(batch=B, channels=C, frames=T, mel_bins=Fq),
            device=self.device,
        ).to(self.dtype)
        return latent, positions


# ---------------------------------------------------------------------------
# One-stage GGUF-backed ID-LoRA pipeline
# ---------------------------------------------------------------------------

class GGUFIDLoraOneStagePipeline(_GGUFIDLoraBase):
    """One-stage pipeline: audio+video identity transfer using a GGUF model.

    Same guidance logic as the original ``IDLoraOneStagePipeline``, but the
    transformer call routes through ``GGUFTransformerBridge`` which uses
    ComfyUI's GGUF-backed model (15 GB Q5_0 instead of 46 GB bf16).

    Parameters
    ----------
    model_patcher : ModelPatcher
        ComfyUI model patcher wrapping the GGUF LTXAV model.
    component_loader : SplitComponentLoader
        Provides VAEs, text encoder, and embeddings processor.
    stg_scale, identity_guidance_scale, av_bimodal_scale : float
        Guidance scales. Set to 0 to disable each.
    """

    def __init__(
        self,
        model_patcher,
        component_loader: SplitComponentLoader,
        device: torch.device = torch.device("cuda"),
        stg_scale: float = 1.0,
        stg_blocks: list[int] | None = None,
        stg_mode: str = "stg_av",
        identity_guidance: bool = True,
        identity_guidance_scale: float = 3.0,
        av_bimodal_cfg: bool = True,
        av_bimodal_scale: float = 3.0,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_patcher = model_patcher
        self.model_ledger = component_loader  # duck-typed for encode_prompts
        # Give the component loader access to the model patcher so it can
        # extract connector weights for the embeddings processor.
        component_loader._model_patcher = model_patcher

        self._stg_scale = stg_scale
        self._stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self._stg_mode = stg_mode
        self._identity_guidance = identity_guidance
        self._identity_guidance_scale = identity_guidance_scale
        self._av_bimodal_cfg = av_bimodal_cfg
        self._av_bimodal_scale = av_bimodal_scale

        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

        self._transformer = None   # GGUFTransformerBridge, set in load_models
        self._video_encoder = None
        self._video_encoder_on_cpu = None
        self._audio_encoder = None
        self._audio_processor = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self):
        """Load VAE encoders for use in __call__().

        Called by the sampler node after prompt encoding is complete.
        The GGUF model is NOT loaded here — it's loaded in __call__()
        after the VAE encoders are freed, to avoid exceeding VRAM.
        """
        t0 = time.time()
        print("[ID-LoRA-GGUF] Loading video encoder...", flush=True)
        self._video_encoder = self.model_ledger.video_encoder()
        print(f"[ID-LoRA-GGUF]   Video encoder loaded in {time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        print("[ID-LoRA-GGUF] Loading audio encoder...", flush=True)
        self._audio_encoder = self.model_ledger.audio_encoder()
        print(f"[ID-LoRA-GGUF]   Audio encoder loaded in {time.time()-t1:.1f}s", flush=True)

        print("[ID-LoRA-GGUF] Creating audio processor...", flush=True)
        self._audio_processor = AudioProcessor(
            target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(self.device)
        print(f"[ID-LoRA-GGUF] All encoders loaded in {time.time()-t0:.1f}s", flush=True)

    def _ensure_video_encoder(self):
        if self._video_encoder is None and self._video_encoder_on_cpu is not None:
            self._video_encoder = self._video_encoder_on_cpu.to(self.device)
            self._video_encoder_on_cpu = None
        elif self._video_encoder is None:
            print("[ID-LoRA-GGUF] Re-loading video encoder...")
            self._video_encoder = self.model_ledger.video_encoder()

    def _stash_video_encoder(self):
        if self._video_encoder is not None:
            self._video_encoder_on_cpu = self._video_encoder.cpu()
            self._video_encoder = None
            cleanup_memory()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        v_context_p: torch.Tensor,
        a_context_p: torch.Tensor,
        v_context_n: torch.Tensor,
        a_context_n: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        reference_audio: torch.Tensor | None = None,
        reference_audio_sample_rate: int = 16000,
        condition_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_start = time.time()
        print(f"[ID-LoRA-GGUF] === Starting generation ===", flush=True)
        print(f"[ID-LoRA-GGUF]   Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}", flush=True)
        print(f"[ID-LoRA-GGUF]   CFG: video={video_guidance_scale}, audio={audio_guidance_scale}", flush=True)
        print(f"[ID-LoRA-GGUF]   Seed: {seed}, Frame rate: {frame_rate}", flush=True)

        self._ensure_video_encoder()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        video_cfg = CFGGuider(video_guidance_scale)
        audio_cfg = CFGGuider(audio_guidance_scale)
        stg_guider = STGGuider(self._stg_scale)
        av_bimodal_guider = CFGGuider(self._av_bimodal_scale if self._av_bimodal_cfg else 0.0)

        stg_pcfg = self._stg_config() if stg_guider.enabled() else None
        av_pcfg = self._av_bimodal_config() if av_bimodal_guider.enabled() else None

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device,
        )

        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        t_enc = time.time()
        print("[ID-LoRA-GGUF] Encoding video state (image conditioning)...", flush=True)
        video_state, video_tools, ref_vid_len = self._create_video_state(
            output_shape=output_shape,
            condition_image=condition_image,
            noiser=noiser,
            frame_rate=frame_rate,
        )
        print(f"[ID-LoRA-GGUF]   Video state created in {time.time()-t_enc:.1f}s", flush=True)

        t_aud = time.time()
        print("[ID-LoRA-GGUF] Encoding audio state...", flush=True)
        audio_state, audio_tools, ref_aud_len = self._create_audio_state(
            output_shape=output_shape,
            reference_audio=reference_audio,
            reference_audio_sample_rate=reference_audio_sample_rate,
            noiser=noiser,
        )
        print(f"[ID-LoRA-GGUF]   Audio state created in {time.time()-t_aud:.1f}s (ref_len={ref_aud_len})", flush=True)

        # Free ALL VAE/audio components to make room for GGUF model (~14GB)
        print("[ID-LoRA-GGUF] Freeing VAE encoders for GGUF denoising...", flush=True)
        if self._video_encoder is not None:
            del self._video_encoder
            self._video_encoder = None
        self._video_encoder_on_cpu = None
        if self._audio_encoder is not None:
            del self._audio_encoder
            self._audio_encoder = None
        if self._audio_processor is not None:
            self._audio_processor = self._audio_processor.cpu()
        cleanup_memory()

        # Create the bridge with the correct video latent shape
        t_gguf = time.time()
        print("[ID-LoRA-GGUF] Loading GGUF model to GPU...", flush=True)
        video_latent_shape = VideoLatentShape.from_pixel_shape(output_shape)
        bridge = GGUFTransformerBridge(
            model_patcher=self.model_patcher,
            video_latent_shape=video_latent_shape,
            frame_rate=frame_rate,
        )
        bridge.load_to_gpu()
        print(f"[ID-LoRA-GGUF]   GGUF model loaded in {time.time()-t_gguf:.1f}s", flush=True)

        total_steps = len(sigmas) - 1
        print(f"[ID-LoRA-GGUF] Starting denoising ({total_steps} steps)...", flush=True)
        t_denoise = time.time()

        # -- Denoising function (same logic as original) ------------------
        _step_times = []
        def denoising_func(video_state, audio_state, sigmas, step_idx):
            sigma = sigmas[step_idx]
            t_step = time.time()
            elapsed = time.time() - t_denoise
            eta = ""
            if _step_times:
                avg = sum(_step_times) / len(_step_times)
                remaining = avg * (total_steps - step_idx)
                eta = f", ETA {remaining:.0f}s"
            print(f"  Step {step_idx + 1}/{total_steps} (σ={sigma.item():.4f}, {elapsed:.0f}s elapsed{eta})", flush=True)

            pv = modality_from_latent_state(video_state, v_context_p, sigma)
            pa = modality_from_latent_state(audio_state, a_context_p, sigma)
            dv_pos, da_pos = bridge(video=pv, audio=pa, perturbations=None)

            delta_v = torch.zeros_like(dv_pos)
            delta_a = torch.zeros_like(da_pos) if da_pos is not None else None

            # CFG
            if video_cfg.enabled() or audio_cfg.enabled():
                nv = modality_from_latent_state(video_state, v_context_n, sigma)
                na = modality_from_latent_state(audio_state, a_context_n, sigma)
                dv_neg, da_neg = bridge(video=nv, audio=na, perturbations=None)
                delta_v = delta_v + video_cfg.delta(dv_pos, dv_neg)
                if delta_a is not None:
                    delta_a = delta_a + audio_cfg.delta(da_pos, da_neg)

            # Identity guidance
            if self._identity_guidance and self._identity_guidance_scale > 0 and ref_aud_len > 0:
                tgt_aud = LatentState(
                    latent=audio_state.latent[:, ref_aud_len:],
                    denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                    positions=audio_state.positions[:, :, ref_aud_len:],
                    clean_latent=audio_state.clean_latent[:, ref_aud_len:],
                )
                nrv = modality_from_latent_state(video_state, v_context_p, sigma)
                nra = modality_from_latent_state(tgt_aud, a_context_p, sigma)
                _, da_noref = bridge(video=nrv, audio=nra, perturbations=None)
                if delta_a is not None and da_noref is not None:
                    id_delta = self._identity_guidance_scale * (da_pos[:, ref_aud_len:] - da_noref)
                    full_id = torch.zeros_like(delta_a)
                    full_id[:, ref_aud_len:] = id_delta
                    delta_a = delta_a + full_id

            # STG (v1: perturbations not yet mapped, so stg_guider.enabled()
            #       effectively does nothing unless perturbation mapping is added)
            if stg_guider.enabled() and stg_pcfg is not None:
                pv_s, pa_s = bridge(video=pv, audio=pa, perturbations=stg_pcfg)
                delta_v = delta_v + stg_guider.delta(dv_pos, pv_s)
                if delta_a is not None and pa_s is not None:
                    delta_a = delta_a + stg_guider.delta(da_pos, pa_s)

            # AV bimodal CFG
            if av_bimodal_guider.enabled() and av_pcfg is not None:
                pv_b, pa_b = bridge(video=pv, audio=pa, perturbations=av_pcfg)
                delta_v = delta_v + av_bimodal_guider.delta(dv_pos, pv_b)
                if delta_a is not None and pa_b is not None:
                    delta_a = delta_a + av_bimodal_guider.delta(da_pos, pa_b)

            out_v = dv_pos + delta_v
            out_a = (da_pos + delta_a) if (da_pos is not None and delta_a is not None) else da_pos
            _step_times.append(time.time() - t_step)
            return out_v, out_a

        # -- Run denoising loop -------------------------------------------
        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=denoising_func,
        )
        denoise_elapsed = time.time() - t_denoise
        avg_step = sum(_step_times) / len(_step_times) if _step_times else 0
        print(f"[ID-LoRA-GGUF] Denoising complete: {denoise_elapsed:.1f}s total, {avg_step:.1f}s/step avg", flush=True)

        # -- Strip reference tokens ---------------------------------------
        if ref_vid_len > 0:
            video_state = LatentState(
                latent=video_state.latent[:, ref_vid_len:],
                denoise_mask=video_state.denoise_mask[:, ref_vid_len:],
                positions=video_state.positions[:, :, ref_vid_len:],
                clean_latent=video_state.clean_latent[:, ref_vid_len:],
            )
        if ref_aud_len > 0:
            audio_state = LatentState(
                latent=audio_state.latent[:, ref_aud_len:],
                denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                positions=audio_state.positions[:, :, ref_aud_len:],
                clean_latent=audio_state.clean_latent[:, ref_aud_len:],
            )

        # Unpatchify while still on GPU (needs patchifier)
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # -- Offload transformer, decode ----------------------------------
        print("[ID-LoRA-GGUF] Offloading GGUF model...", flush=True)
        bridge.offload_to_cpu()
        del bridge
        # Aggressively free ALL GPU memory — the GGUF model and any cached
        # weights must be fully evicted before the video decoder can fit.
        comfy.model_management.unload_all_models()
        cleanup_memory()

        # Move latents to CPU to free GPU for decoder
        video_latent_cpu = video_state.latent.cpu()
        audio_latent_cpu = audio_state.latent.cpu()
        del video_state, audio_state
        cleanup_memory()

        t_dec = time.time()
        print("[ID-LoRA-GGUF] Loading video decoder...", flush=True)
        video_decoder = self.model_ledger.video_decoder()
        video_latent = video_latent_cpu.to(device=self.device, dtype=torch.bfloat16)
        del video_latent_cpu
        print("[ID-LoRA-GGUF] Decoding video latents...", flush=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded_video = video_decoder(video_latent)
        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        video_tensor = decoded_video[0].float().cpu()
        del video_latent, decoded_video, video_decoder
        cleanup_memory()
        print(f"[ID-LoRA-GGUF]   Video decoded in {time.time()-t_dec:.1f}s", flush=True)

        t_adec = time.time()
        print("[ID-LoRA-GGUF] Loading audio decoder + vocoder...", flush=True)
        audio_decoder = self.model_ledger.audio_decoder()
        vocoder = self.model_ledger.vocoder()
        audio_latent_cuda = audio_latent_cpu.to(device=self.device, dtype=torch.bfloat16)
        del audio_latent_cpu
        print("[ID-LoRA-GGUF] Decoding audio latents...", flush=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded_audio = vae_decode_audio(audio_latent_cuda, audio_decoder, vocoder)
        audio_output = decoded_audio.waveform.cpu()
        self._vocoder_sr = decoded_audio.sampling_rate
        print(f"[ID-LoRA-GGUF]   Audio decoded in {time.time()-t_adec:.1f}s", flush=True)

        del audio_decoder, vocoder, audio_latent_cuda
        gc.collect()
        torch.cuda.empty_cache()

        # Reload GGUF model for potential next run
        print("[ID-LoRA-GGUF] Reloading GGUF model...", flush=True)
        comfy.model_management.load_models_gpu([self.model_patcher])

        total_time = time.time() - t_start
        print(f"[ID-LoRA-GGUF] === Generation complete in {total_time:.1f}s ===", flush=True)
        return video_tensor, audio_output
