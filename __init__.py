"""ComfyUI nodes for the popular Roformer music source separation checkpoints.

The whole pack registers through the V3 entrypoint below. ComfyUI's loader registers a custom
node module as either V1 (NODE_CLASS_MAPPINGS) or V3 (comfy_entrypoint) - it checks V1 first
and returns, so a module with both would silently lose its V3 nodes; this pack therefore has no
NODE_CLASS_MAPPINGS of its own and registers everything here: the six io.ComfyNode classes in
nodes.py.
"""
from comfy_api.latest import ComfyExtension

from . import nodes

__all__ = ["comfy_entrypoint"]


class RoformerPack(ComfyExtension):
    async def get_node_list(self):
        return [
            nodes.RoformerLoader,
            nodes.RoformerSeparate,
            nodes.RoformerToAudio,
            nodes.CustomRoformerToAudio,
            nodes.RoformerSaveAudio,
            nodes.RoformerInfo,
        ]


async def comfy_entrypoint() -> RoformerPack:
    return RoformerPack()
