"""ComfyUI custom node package for GGUF-backed ID-LoRA-2.3 inference.

Uses a 15 GB GGUF model instead of the 46 GB bf16 checkpoint, saving 31 GB
of disk and preserving GGUF's lazy-dequant VRAM efficiency (~18 GB VRAM).
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes_loader import IDLoraGGUFModelLoader
from .nodes_prompt_encoder import IDLoraGGUFPromptEncoder
from .nodes_sampler import IDLoraGGUFSampler


class IDLoraGGUFExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IDLoraGGUFModelLoader,
            IDLoraGGUFPromptEncoder,
            IDLoraGGUFSampler,
        ]


async def comfy_entrypoint() -> IDLoraGGUFExtension:
    return IDLoraGGUFExtension()
