# Vendored Qwen3-VL modeling code with starVLA dual-stream cross-view
# interaction (spec: qwen3_vl_2b_dual_stream_design.md).
#
# Copied from transformers==4.57.6 so the vision tower can be modified in
# the project without touching the installed transformers package. The
# dual-stream extension activates only when
# `vision_config.dual_stream["enabled"]` is truthy; otherwise behavior and
# parameter names match upstream.

from .modeling_qwen3_vl import (
    Qwen3VLDualStreamGateMLP,
    Qwen3VLDualStreamInteractionStage,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLVisionModel,
)

__all__ = [
    "Qwen3VLDualStreamGateMLP",
    "Qwen3VLDualStreamInteractionStage",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLVisionModel",
]
