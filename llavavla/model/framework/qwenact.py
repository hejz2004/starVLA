"""
cogactvla.py

"""
from __future__ import annotations
from typing import Union, List

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from types import SimpleNamespace
import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
import re
from prismatic.overwatch import initialize_overwatch

from llavavla.model.action_model.action_model import ActionModel
import torch.distributed as dist
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

# get QWen2.5
from llavavla.model.vlm import _QWen_VL_Interface #不应该强依赖于这个，应该是一个接口类，而不是一个具体的类, TODO 不要实现 hard 接口类， 使用 **kwargs
from llavavla.model.tools import auto_get_module_keys, auto_get_trainable_modules # 后续应该是trainer 的职责范围
from llavavla.model.vlm.QWen2_5 import get_qwen2_5_interface
from llavavla.model.projector.QFormer import get_layerwise_qformer

class QwenQFormerDiT(nn.Module):
    def __init__(
        self,
        qwen_model_name:str = './playground/Pretrained_models/Qwen2.5-VL-3B-Instruct', # 这是不好的实现， 一定不能是互相依赖
        action_model_type: str = 'DiT-B', 
        vl_token_dim: int = 2048,
        action_hidden_dim: int = 768,  # @Jinhui # 这个 应该是和DiT-B
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        config: Optional[dict] = None,  # @Jinhui TODO 这里应该是config, 但是现在是直接传入参数
        **kwargs,
    ) -> None:
        super().__init__()
        
        # TODO 全部转 全局config, 要面向对象编程
        self.qwen_vl_interface = get_qwen2_5_interface(qwen_model_name, config) 
        self.layer_qformer = get_layerwise_qformer(input_hidden_dim=vl_token_dim, output_hidden_dim=action_hidden_dim,config=config) # @Jinhui 需要逻辑从QWen 中对齐 hidden
        self.action_model = ActionModel(model_type = action_model_type,  # TODO @Jinhui 应该写到 get_action_model()
                                            action_hidden_dim = action_hidden_dim, # 这些参数关系要 TODO集中 设置到config
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size) # 也应该用 函数封装
        
        # TODO ActionModel 需要和qformer 一起设计
        self.config = config
        # self.qwen_processor = vlm.processor # 要面向对象编程， 不要 属性外泄
        # 这些是 action chunck 的参数
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size

        # self.all_module_keys = auto_get_module_keys(self) #  TODO 这个是trainer的 funx， 或许是多余的
        self.norm_stats = norm_stats # 这个是 inference 时候用到的， 不应该是放到这个位置？

        # if we need some pretrain prameters, we can load them here
        # TODO 需要考虑这个是谁的职责 --> 按照扁平管理，切实应该在内部做条件判断
        self.load_pretrained_backbones(self.config) # 只是提示作用

    @property
    def trainable_module_keys(self) -> List[str]:

        # TODO check, 原版返回的死 vlm.model, 新的实现是vlm --> 看一下保存逻辑是否发上变化
        keys = auto_get_trainable_modules(self, max_depth=1)# auto 去判断哪些module是trainable的
        return keys
    

    def forward( # TODO 需要将 loss 计算分离出来
        self, # 只面对最原始的 data exmaples, 为了可读性，这里还是要写成显示的参数
        examples: List[dict] = None,  # 这里的 examples 是指原始的输入数据
        repeated_diffusion_steps: int = 4,
        **kwargs,  # 👈 敏捷代码的灵活性， 允许任何形式的传参数
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # @Jinhui TBD TODO 
        images = [example["image"] for example in examples]  #  TODO check 是什么
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples] #label
        
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions = instructions) # @Jinhui TODO add instruction to qwenvl inputs
        
        with torch.autocast("cuda", dtype=torch.float16):
            # dist.barrier()  # 确保所有进程都加载完毕
            qwenvl_outputs = self.qwen_vl_interface( # 都是local的参数变化， 不要写到config, 但是为了保持可复现，应该有个默认的 yaml
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, # [512, 1176] 石斛没有 B,  
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                labels= qwen_inputs.input_ids.clone(),
                # use_cache=use_cache,
                output_attentions=False, # Flash attention 还不确定是否支持返回attention， 官方代码有bug
                output_hidden_states=True,
                return_dict=True,
                # past_key_values=past_key_values,
                # **kwargs
                )
        
        vlm_loss = qwenvl_outputs.loss # @Jinhui TODO 这里是可以study 的地方， 是否 training lang
        with torch.autocast("cuda", dtype=torch.bfloat16):
            start_layer = self.config.vla.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.vla.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer]) # [B, 64, D_action]
    
        # actions = torch.stack([torch.tensor(a) for a in actions], dim=0).to(action_latent_feature.device)  # [B, chunk, 7] @Jinhui TODO to tensor 的逻辑可以放到 transform 里面
        # 先将 actions 转换为单个 NumPy 数组，再转换为 PyTorch 张量
        actions = torch.tensor(np.array(actions), device=action_latent_feature.device)  # [B, chunk, 7] TODO to tensor 的逻辑可以放到 transform 里面
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        action_latent_feature = action_latent_feature.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, T, D_action]
        # Action model forward and compute loss # 这里功能有点 越俎代庖 TODO 将loss 集中到 main module中统一处理
        action_loss = self.action_model.loss(actions_repeated, action_latent_feature) # TODO loss 应该放到另一个函数
        return action_loss, qwenvl_outputs

    # @torch.inference_mode() # @Jinhui DEBUG 临时取消
    def predict_action( # 
        self, image: Union[Image, List[Image]],
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """

        # @之后写入模型内部， 变成私有化方法
        if not isinstance(image, list):
            imgs = [image.resize((224, 224))]  # list of PIL RGB for one instruction
        else:
            imgs = [img.resize((224, 224)) for img in image]
        
        lang = instruction.lower() 

        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=[imgs], instructions = [lang]) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

        
        # add by Jinhui

        # end add by Jinhui
        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fmt: off
            qwenvl_outputs = self.qwen_vl_interface( # TODO 这里之后要用generation func
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, # [512, 1176] 石斛没有 B,  
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                labels= qwen_inputs.input_ids.clone(),
                output_hidden_states=True, 
                return_dict=True,
            ) # generation 拿不到前面token 的信息，考虑使用 forward?

        with torch.autocast("cuda", dtype=torch.bfloat16):
            start_layer = self.config.vla.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.vla.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer]) # [B, 64, D_action]
            
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = action_latent_feature.shape[0]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=action_latent_feature.device).to(model_dtype)  #[B, T, D]

        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0) #[2,16,7]
            uncondition = self.action_model.net.z_embedder.uncondition # [64, 768]
            uncondition_shape = uncondition.shape
            uncondition = uncondition.unsqueeze(0)  #[1, 64, D]
            uncondition = uncondition.expand(B, uncondition_shape[0], uncondition_shape[1]) #[B, n_qformer_token, D] # 
            z = torch.cat([action_latent_feature, uncondition], 0) # [2, 64, 768] TODO check 看看 training的时候是剁手
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=action_latent_feature)
            sample_fn = self.action_model.net.forward
        
        # if os.environ.get("DEBUG"):
        #     print(z .shape)
        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None: #@JinhuiYE =TODO check, shape 上没问题， 就不知道traine / infer 和内部操作是否有问题了
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=action_latent_feature.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=action_latent_feature.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        # actions max 1, min -0.05 # 感觉不再一个 scale
        return actions, normalized_actions


    def freeze_backbones(self):
        """
        根据相对模块路径列表（patterns）直接冻结指定子模块，不再递归查找所有子模块名称：
          - patterns: 从 config.vla.freeze_modules 中读取，用逗号分隔得到的“相对路径”列表
            例如 "qwen_vl_interface, action_model.net"，
            就意味着冻结 self.qwen_vl_interface 和 self.action_model.net。
        返回值：
          - frozen: 实际找到并冻结的模块路径列表
        """
        freeze_modules = ( # 我觉得全局就应该只有一个config， 使用没必要相对路径
            self.config.vla.freeze_modules
            if (self.config and hasattr(self.config.vla, "freeze_modules"))
            else None
        )
        # 拆分并去除空白
        patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

        frozen = []
        for path in patterns:
            # 将“相对路径”按点拆分，例如 "action_model.net" → ["action_model", "net"]
            attrs = path.split(".")
            module = self
            try:
                for attr in attrs:
                    module = getattr(module, attr)
                # 如果成功 get 到 module，就把它和它的所有子模块参数都 freeze
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(path)
            except AttributeError:
                # 如果某一级属性不存在，就跳过并打印警告
                print(f"⚠️ 模块路径不存在，无法冻结：{path}")
                continue

        dist.barrier()  # 分布式训练时同步
        print(f"🔒 Frozen modules (by relative path): {frozen}")
        return frozen
    
    def load_pretrained_backbones(self, config): # TODO Jinhui 这在哪里被调用还是需要商量
        """
        加载 checkpoint：
        - 如果设置了 config.vla.reload_modules（逗号分隔的模块路径）→ 按路径部分加载
        - 否则 → 加载整个模型参数（覆盖 self）

        返回：
            替换，loaded_modules: 成功加载参数的模块路径列表；若全局加载则为 ["<full_model>"]
        """
        checkpoint_path = getattr(self.config, "pretrained_checkpoint", None)
        reload_module_name = getattr(self.config, "reload_modules", None)

        if not checkpoint_path:
            return []  

        print(f"📦 正在加载 checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ 加载 checkpoint 失败: {e}")

        loaded_modules = []

        if reload_module_name:  # 部分加载
            module_paths = [p.strip() for p in reload_module_name.split(",") if p.strip()]
            for path in module_paths:
                reload_module_name = path.split(".")
                module = self
                try:
                    for module_name in reload_module_name: # 这里 top2down 的找到 要修改的module
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {
                        k[len(prefix):]: v
                        for k, v in checkpoint.items()
                        if k.startswith(prefix)
                    }

                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        print(f"✅ 参数已加载到模块 '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ checkpoint 中未找到 '{path}' 相关参数")
                except AttributeError:
                    print(f"❌ 无法找到模块路径：{path}")
        else:  # 全部加载
            try:
                self.load_state_dict(checkpoint, strict=True)
                print("✅ 已加载完整模型参数")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ 加载完整模型失败: {e}")

        return loaded_modules
    def print_freeze_status(self): # 这个是 工具类方法。 可以考虑移动
        for name, param in self.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @classmethod
    def from_pretrained( # @Jinhui TODO 这里要写如何resume checkpoints
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint) # 读取 config 和 norm_stats
        # Initialize CogACT
        # model_config TODO DEBUE @JinhuiYE 这里应该保证training infer 的参数和模型🔗是一致的 （特别是 QFormer)
        # TODO 
        config = dict_to_namespace(model_config)
        model_config = config # TODO 不要使用相对变量 model_config， 需要换名字
        qwenQFormerACT = build_model_framework(model_config) 
        # set for action un-norm
        qwenQFormerACT.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu") #["model"]
        
        model_keys = set(qwenQFormerACT.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())

        # ✅ 1. 加载匹配的权重
        for key in checkpoint_keys:
            if key in model_keys:
                try:
                    qwenQFormerACT.state_dict()[key].copy_(model_state_dict[key])
                    # overwatch.info(f"✅ Loaded: {key}")
                except Exception as e:
                    overwatch.warning(f"⚠️ Failed to copy weight for key '{key}': {e}")
            else:
                overwatch.warning(f"⚠️ Checkpoint has unknown key '{key}' (not in model). Ignoring.")

        # ✅ 2. 反向检查：模型中有但 checkpoint 中缺失的
        missing_keys = model_keys - checkpoint_keys # TODO 这里之后要考虑 nontrainable params --> 我觉得没必要省存储空间
        for key in sorted(missing_keys):
                overwatch.warning(f"⚠️ Model expects key '{key}' but it's missing in checkpoint.")
                
        return qwenQFormerACT
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

# TODO 写一个build model 函数

def build_model_framework(model_config: dict = {}) -> QwenQFormerDiT:
    # TODO  实现和 config 对应的 load 逻辑

    model = QwenQFormerDiT(
    qwen_model_name='/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct',
    action_model_type='DiT-B',
    vl_token_dim=2048,
    action_dim=model_config.vla.action_dim if hasattr(model_config.vla, 'action_dim') else 7,  # @Jinhui TODO 这里应该是config
    future_action_window_size=15,
    past_action_window_size=0,
    # use_ema=False,
    config=model_config
    )
        
    return model


def read_mode_config(pretrained_checkpoint):
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt")
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)

        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f) #["vla"]
            # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    return vla_cfg, norm_stats

def load_from_pretrained(pretrained_checkpoint):
    """Load a pretrained QwenQFormerDiT model from a checkpoint."""

    # TODO 这里应该是从config中加载
    
    model = QwenQFormerDiT.from_pretrained(
        pretrained_checkpoint=pretrained_checkpoint)
    return model


if __name__ == "__main__":
    from omegaconf import OmegaConf
    # 模型参数
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("🔍 Rank 0 waiting for debugger attach on port 5878...")
    debugpy.wait_for_client()
    samples = {}

    config_yaml = "llavavla/conf/qwenvla_cotrain_v2.yaml"
    cfg = OmegaConf.load(config_yaml)

    model_framework = build_model_framework(cfg)
    model_framework(samples)
    pass

    # git remote add gitee https://gitee.pjlab.org.cn/L2/MultimodalVLA/llavavla.git
    # git push -u gitee master