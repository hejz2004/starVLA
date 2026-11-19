"""DualStream bottleneck vision with the current QwenPI training/action path."""
from dataclasses import dataclass, field

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI, QwenPIDefaultConfig
from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class DualStreamQwenPIDefaultConfig(QwenPIDefaultConfig):
    name: str = "DualStreamQwenPI"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-2B-Instruct",
        "load_weights": True,
        "attn_implementation": "sdpa",
        "vl_hidden_dim": 2048,
        "cross_view_attention": None,
        "dual_stream": {
            "enabled": True,
            "interaction_indexes": [5, 11, 17, 23],
            "num_shared_tokens": 32,
            "num_heads": 16,
            "intermediate_size": 4096,
            "gate_hidden_size": 256,
            "gate_init_value": 0.01,
            "gate_output_weight_std": 0.001,
            "attention_dropout": 0.0,
        },
    })


@FRAMEWORK_REGISTRY.register("DualStreamPI")
@FRAMEWORK_REGISTRY.register("DualStreamQwenPI")
@FRAMEWORK_REGISTRY.register("DualStreamQwenFM")
class DualStreamQwen_PI(Qwen_PI):
    default_config = DualStreamQwenPIDefaultConfig

    def _build_vlm(self):
        cross_view = self.config.framework.qwenvl.get("cross_view_attention")
        if cross_view and cross_view.get("enabled", False):
            raise ValueError("DualStream and late alternating cross-view attention cannot both be enabled")
        return _QWen3DualStream_VL_Interface(config=self.config)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(".register_tokens." in key for key in state_dict):
            raise ValueError("MV2 register checkpoints cannot initialize DualStreamPI")
        # DualStream checkpoints keep their historical names and strict schema.
        return baseframework.load_state_dict(self, state_dict, strict=strict, assign=assign)
