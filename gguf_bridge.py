"""Bridge between ltx-core's Modality/X0Model interface and ComfyUI's GGUF model.

Provides a callable with the same signature as ``X0Model.__call__()`` so the
ID-LoRA denoising loop can be used unmodified. Internally converts between
patchified Modality tensors and ComfyUI's 5-D spatial convention, calls the
GGUF-backed model through ``BaseModel._apply_model``, and converts back.

Denoising equivalence
---------------------
LTXAV uses ``CONST`` prediction type in ComfyUI:
  calculate_input  = identity (no scaling)
  calculate_denoised = input − velocity × σ

ltx-core X0Model computes:
  denoised = sample − velocity × timesteps   (per-token)

For non-conditioned tokens (mask=1): timesteps = σ → identical.
For conditioned tokens (mask=0): model sees timestep=0, outputs ≈ zero
velocity; ``post_process_latent`` in the denoising loop replaces them with
``clean_latent`` anyway.  Numerically equivalent.
"""

from __future__ import annotations

import math
import time

import torch
import comfy.model_management
import comfy.utils

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.types import AudioLatentShape, VideoLatentShape


class GGUFTransformerBridge:
    """Drop-in replacement for ``X0Model`` that routes through a ComfyUI
    ``GGUFModelPatcher``-wrapped ``LTXAV`` base model.

    Parameters
    ----------
    model_patcher : GGUFModelPatcher
        The GGUF model patcher with the LTX-AV model loaded.
    video_latent_shape : VideoLatentShape
        Shape of the target video latent (used for un-/re-patchify).
    frame_rate : float
        Frame rate passed to the diffusion model.
    """

    def __init__(
        self,
        model_patcher,
        video_latent_shape: VideoLatentShape,
        frame_rate: float = 25.0,
    ):
        self.model_patcher = model_patcher
        self.base_model = model_patcher.model          # LTXAV (BaseModel)
        self.video_latent_shape = video_latent_shape
        self.frame_rate = frame_rate

        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

        # Cache the model dtype for casting
        self._dtype = self.base_model.get_dtype_inference()

    # ------------------------------------------------------------------
    # Public interface (same as X0Model.__call__)
    # ------------------------------------------------------------------

    def __call__(
        self,
        video,   # Modality
        audio,   # Modality
        perturbations=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the GGUF model, returning denoised patchified
        latents exactly as ``X0Model`` would.

        Parameters
        ----------
        video, audio : Modality
            Patchified input modalities from ``modality_from_latent_state``.
        perturbations : BatchedPerturbationConfig | None
            Currently ignored (v1); STG / bimodal perturbations will require
            ``transformer_options`` mapping in a future iteration.
        """
        sigma = video.sigma  # scalar or (B,) tensor

        # Ensure sigma is 1D (B,) — process_timestep indexes timestep.shape[0]
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], device=video.latent.device, dtype=torch.float32)
        elif sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)

        # 1) Unpatchify latents to spatial format --------------------------
        v_latent_5d = self._video_patchifier.unpatchify(
            video.latent, self.video_latent_shape
        )  # (B, C, F, H, W)

        a_latent_shape = self._infer_audio_shape(audio.latent)
        a_latent_4d = self._audio_patchifier.unpatchify(
            audio.latent, a_latent_shape
        )  # (B, C_a, T_mel, F_mel)

        # 2) Reconstruct spatial denoise masks ----------------------------
        v_mask_5d = self._unpatchify_mask_video(video, sigma)
        a_mask_4d = self._unpatchify_mask_audio(audio, sigma, a_latent_shape)

        # 3) Pack video + audio into a single tensor ----------------------
        packed, shapes = comfy.utils.pack_latents([v_latent_5d, a_latent_4d])

        # 4) Build transformer_options ------------------------------------
        transformer_options = self.model_patcher.model_options.get(
            "transformer_options", {}
        ).copy()
        # v1: perturbations are not mapped to transformer_options yet.
        # The denoising loop can still run CFG + identity guidance (which
        # don't need perturbations).  STG/bimodal guidance will be added
        # in a follow-up.

        # 5) Concatenate video + audio context along last dim ----------------
        #    LTXAV._prepare_context splits context into [v_dim, a_dim] along
        #    the last dimension.  The embeddings processor produces separate
        #    video (B, S, 4096) and audio (B, S, 2048) encodings.
        context = torch.cat([video.context, audio.context], dim=-1)

        # 6) Call _apply_model --------------------------------------------
        #    This handles: dtype casting → process_timestep (per-token
        #    timesteps from spatial mask) → diffusion_model.forward →
        #    calculate_denoised.
        denoised_packed = self.base_model._apply_model(
            packed,
            sigma,
            c_crossattn=context,
            transformer_options=transformer_options,
            denoise_mask=v_mask_5d,
            audio_denoise_mask=a_mask_4d,
            latent_shapes=shapes,
            frame_rate=self.frame_rate,
        )

        # 6) Unpack -------------------------------------------------------
        unpacked = comfy.utils.unpack_latents(denoised_packed, shapes)
        denoised_v_5d = unpacked[0]
        denoised_a_4d = unpacked[1] if len(unpacked) > 1 else None

        # 7) Patchify back to Modality format -----------------------------
        denoised_v = self._video_patchifier.patchify(denoised_v_5d)
        denoised_a = (
            self._audio_patchifier.patchify(denoised_a_4d)
            if denoised_a_4d is not None
            else None
        )

        return denoised_v, denoised_a

    # ------------------------------------------------------------------
    # Model lifecycle helpers
    # ------------------------------------------------------------------

    def load_to_gpu(self):
        """Ensure the GGUF model is loaded and resident on the GPU."""
        t0 = time.time()
        comfy.model_management.load_models_gpu([self.model_patcher])
        print(f"[GGUF-Bridge]   Model loaded to GPU in {time.time()-t0:.1f}s", flush=True)

    def offload_to_cpu(self):
        """Move model weights to CPU and free VRAM."""
        self.model_patcher.unpatch_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def to(self, device):
        """Compatibility shim so pipeline can call ``self._transformer.to(device)``."""
        if str(device) == "cpu":
            self.offload_to_cpu()
        else:
            self.load_to_gpu()
        return self

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_audio_shape(audio_latent: torch.Tensor) -> AudioLatentShape:
        """Infer ``AudioLatentShape`` from a patchified tensor.

        For ``AudioPatchifier(patch_size=1)``:
          patchify:   (B, C, T, F_mel) → (B, T, C·F_mel)
          unpatchify: (B, T, C·F_mel) → (B, C, T, F_mel)  given C and F_mel.

        LTX-2.3 audio VAE: channels=8, mel_bins=16 → D_a = 8*16 = 128.
        """
        B, T_a, D_a = audio_latent.shape
        C_audio = 8
        F_mel = D_a // C_audio  # 128 // 8 = 16
        return AudioLatentShape(batch=B, channels=C_audio, frames=T_a, mel_bins=F_mel)

    def _unpatchify_mask_video(self, video, sigma) -> torch.Tensor:
        """Reconstruct spatial video denoise mask from Modality timesteps.

        ``Modality.timesteps = denoise_mask × σ`` (per-token, patchified).
        We recover: ``spatial_mask = timesteps / σ`` then unpatchify.

        Returns shape ``(B, 1, F, H, W)`` for ``process_timestep``.
        """
        sigma_val = sigma if isinstance(sigma, (int, float)) else sigma.item()
        if sigma_val == 0:
            # At sigma=0 the loop is finished; return all-ones
            B = video.latent.shape[0]
            vls = self.video_latent_shape
            return torch.ones(
                B, 1, vls.frames, vls.height, vls.width,
                device=video.latent.device, dtype=torch.float32,
            )

        # timesteps: (B, T_v) or (B, T_v, 1) → (B, T_v, 1)
        ts = video.timesteps
        if ts.ndim == 2:
            ts = ts.unsqueeze(-1)
        mask_patchified = (ts / sigma_val).clamp(0, 1).float()

        # Unpatchify as if channels=1: (B, T_v, 1) → (B, 1, F, H, W)
        mask_shape = VideoLatentShape(
            batch=mask_patchified.shape[0],
            channels=1,
            frames=self.video_latent_shape.frames,
            height=self.video_latent_shape.height,
            width=self.video_latent_shape.width,
        )
        return self._video_patchifier.unpatchify(mask_patchified, mask_shape)

    def _unpatchify_mask_audio(self, audio, sigma, a_shape: AudioLatentShape) -> torch.Tensor:
        """Reconstruct spatial audio denoise mask.

        Returns shape ``(B, 1, T_mel, F_mel)`` for ``process_timestep``.
        """
        sigma_val = sigma if isinstance(sigma, (int, float)) else sigma.item()
        if sigma_val == 0:
            B = audio.latent.shape[0]
            return torch.ones(
                B, 1, a_shape.frames, a_shape.mel_bins,
                device=audio.latent.device, dtype=torch.float32,
            )

        ts = audio.timesteps
        if ts.ndim == 2:
            ts = ts.unsqueeze(-1)
        # For audio patchifier (patch_size=1): (B, T, C*F) with C=1
        # We need (B, T, F_mel) to unpatchify to (B, 1, T, F_mel)
        # But the mask is (B, T, 1), so we need to expand it to (B, T, F_mel)
        mask_per_token = (ts / sigma_val).clamp(0, 1).float()  # (B, T, 1)
        # Expand to match audio patchifier: (B, T, 1*F_mel)
        mask_expanded = mask_per_token.expand(-1, -1, a_shape.mel_bins)
        mask_shape = AudioLatentShape(
            batch=mask_per_token.shape[0],
            channels=1,
            frames=a_shape.frames,
            mel_bins=a_shape.mel_bins,
        )
        return self._audio_patchifier.unpatchify(mask_expanded, mask_shape)
