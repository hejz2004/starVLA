"""
cogactvla.py

"""
from __future__ import annotations
from typing import Union, List
import torchvision
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from types import SimpleNamespace
import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
import re


from accelerate.logging import get_logger
logger = get_logger(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from llavavla.model.framework.share_tools import dict_to_namespace


# get QWen2.5
from llavavla.model.tools import auto_get_module_keys, auto_get_trainable_modules # 后续应该是trainer 的职责范围
from llavavla.model.vlm.QWen2_5 import get_qwen2_5_interface
from llavavla.model.projector.QFormer import get_layerwise_qformer
from llavavla.model.action_model.action_model import get_action_model
from llavavla.training.metrics import resize_images
from llavavla.model.vlm.dino import get_dino_model


class QwenQFormerDiT(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,  # @Jinhui TODO 这里应该是config, 但是现在是直接传入参数
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO 全部转 全局config, 要面向对象编程
        self.qwen_vl_interface = get_qwen2_5_interface(model_id=config.framework.qwenvl.base_vlm, config=self.config) 
        self.layer_qformer = get_layerwise_qformer(config=self.config) # @Jinhui 一般来说 人们喜欢总分结构， 但是有讨厌递归， 实验framework 下面就不能太总分了
        self.action_model = get_action_model(config=self.config)
        self.dino_encoder = get_dino_model(backone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14"))
        # map dino feature to self.qwen_vl_interface.model.config.hidden_size
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,  # DINO 输出的特征维度  
            out_features=self.qwen_vl_interface.model.config.hidden_size)
        
        # get dino
        
        
        # TODO 为什么要在这个位置开始 看到 这些？--> 去思考， framework level 用户其他看到什么， 需要看到什么
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size

        # self.all_module_keys = auto_get_module_keys(self) #  TODO 这个是trainer的 funx， 或许是多余的
        self.norm_stats = norm_stats # 这个是 inference 时候用到的， 不应该是放到这个位置？
        self.use_ema = config.framework.action_model.use_ema


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
        images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples] #label [B， len, 7]
        if "solution" in examples[0]:  # @Jinhui TODO 这里是为了兼容旧的格式
            solutions = [example["solution"] for example in examples]  # [B, dict]
        else: #  还有if else 和模型可阅读性的 trade off
            solutions = None

        # Step 1: 使用 DINO imaga processing 是什么？
        # 将 PIL 图片转换为张量并 **归一化**
        image_tensors = self.dino_encoder.prepare_dino_input(images) # 
        B = len(images)
        dino_features = self.dino_encoder(image_tensors)  # DINO 输出为 [B*num_view, token, dim]

        # 提取 DINO 的特征
        dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]

        dino_encoded_features = self.dino_pro(dino_encoded_features) # [B, num_view * token, hidden_size]
        # Step 2: QWenVL 输入格式
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions = instructions, solutions=solutions) # @Jinhui TODO 再考虑一下这里的分支分流应该有.py控制还是由 if else
        
        with torch.autocast("cuda", dtype=torch.bfloat16): # @Jinhui TODO 这里的 dtype 是不是应该是config里面的
            # dist.barrier()  # 确保所有进程都加载完毕
            qwenvl_outputs = self.qwen_vl_interface( # 都是local的参数变化， 不要写到config, 但是为了保持可复现，应该有个默认的 yaml
                **qwen_inputs, # 兼容性和可读性的 trade off
                output_attentions=False, # Flash attention 还不确定是否支持返回attention， 官方代码有bug
                output_hidden_states=True,
                return_dict=True,
                )
            pass
            # dist.barrier()
        action_cot_loss = qwenvl_outputs.loss # @Jinhui TODO 这里是可以study 的地方， 是否 training lang
        
        if action_cot_loss is None or torch.isnan(action_cot_loss): # TODO 将不同逻辑的 forward 罗杰写成 if else 会破坏可读性
            action_cot_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            
            # 给 DiT 添加来自dino 的输入
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
                cat_conditions.append(layer_features)

            action_latent_feature = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]
    
            # actions = torch.stack([torch.tensor(a) for a in actions], dim=0).to(action_latent_feature.device)  # [B, chunk, 7] @Jinhui TODO to tensor 的逻辑可以放到 transform 里面
            # 先将 actions 转换为单个 NumPy 数组，再转换为 PyTorch 张量
            actions = torch.tensor(np.array(actions), device=action_latent_feature.device)  # [B, chunk, 7] TODO to tensor 的逻辑可以放到 transform 里面
            actions_future = actions[:, -(self.future_action_window_size+1):, :]
            
            # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            action_latent_feature = action_latent_feature.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, T, D_action]
            # Action model forward and compute loss # 这里功能有点 越俎代庖 TODO 将loss 集中到 main module中统一处理
            action_loss = self.action_model.loss(actions_repeated, action_latent_feature) # TODO loss 应该放到另一个函数
        return action_loss, action_cot_loss

    @torch.inference_mode() # @Jinhui DEBUG 临时取消
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
        images = [imgs]

        lang = instruction.lower() 
        instructions = [lang]  # @Jinhui TODO 这里是为了兼容旧的格式， 需要考虑是否要将lang 变成 list
        
        

        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions = instructions) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

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

        # get dino feature
        image_tensors = self.dino_encoder.prepare_dino_input(images)
        dino_features = self.dino_encoder(image_tensors)  # DINO 输出为 OrderedDict

        # 提取 DINO 的特征
        B = dino_features.shape[0]
        dino_encoded_features = dino_features.view(B, -1,dino_features.shape[-1])  # [B, num_view * token, dim]
        dino_encoded_features = self.dino_pro(dino_encoded_features) # B, 256, D

        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            
            # 给 DiT 添加来自dino 的输入
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
                cat_conditions.append(layer_features)

            action_latent_feature = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]
    
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
            
            # Un-normalize Actions --> 这个信息应该集成在哪里，能够能够取消动态
            action_norm_stats = self.get_action_stats(unnorm_key)
            raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=action_norm_stats) 
        return raw_actions, normalized_actions # actions, normalized_actions #TODO 得想清楚， Un-normalize Actions  到底是谁控制的。 我觉得必须是模型， 因为减少相对变化， 扁平管理。 但是要单独写成函数

    @torch.inference_mode() # @Jinhui DEBUG 临时取消
    def predict_action_withCoT( # TODO 和 predict action 是有差别了，之后要去解那边为情况# 那边暂时是for windox
        self, images: Union[Image, List[Image]], # 变成可以 batch infer
        instructions: List[str], 
        solutions: Union[Dict, List[Dict]] = None, # @Jinhui TODO 这里是为了兼容旧的格式, 可以用于出中间表征的评测？
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """

        # @之后写入模型内部， 变成私有化方法
        batch_images = resize_images(images, target_size=(224, 224))  # list of PIL RGB for one instruction
        
        # instructions = [instruction.lower()  for instruction in instructions] # @Jinhui TODO 这里是为了兼容旧的格式， 需要考虑是否要将lang 变成 list
        
        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

        # Generate feature through vlm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fmt: off # TODO --> 高阶功能， 直接train 不要train结束符号
            # TODO Qwen 背后会多一个 终止符号， 需要 [:, :-2] 去掉。是需要思考是否使用 v1 版本来产生 对应的token, v2 不会去尝试学习闭嘴
            qwenvl_outputs = self.qwen_vl_interface.model.generate(
                input_ids=qwen_inputs.input_ids[:, :],
                attention_mask=qwen_inputs.attention_mask[:, :],
                pixel_values=qwen_inputs.pixel_values, 
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                output_hidden_states=True,
                max_new_tokens=256,
                return_dict_in_generate=True,
            )
            # for check output format
            decoded_sequences = self.qwen_vl_interface.processor.tokenizer.batch_decode(
            qwenvl_outputs.sequences, 
            skip_special_tokens=True 
            )
            # print(decoded_sequences[0])

            hidden_states = qwenvl_outputs.hidden_states  # [1 + new token, num_layers, batch_size, attention token, hidden_dim]

            if len(hidden_states) == 1: # 表明没有新的token 残生
                # 如果生成的 token 为 0，仅保留 prefix_hidden_states
                prefix_hidden_states = hidden_states[0]  # Shape: [num_layers, B, prefix_len, hidden_dim]
                prefix_hidden_states = torch.stack(prefix_hidden_states, dim=0)  # Shape: [num_layers, B, prefix_len, hidden_dim]
                combined_hidden_states = prefix_hidden_states  # Shape: [num_layers, B, prefix_len, hidden_dim]
            else: # 为了逻辑清晰而使用了 if else, 
                # 正常处理生成的 token
                prefix_hidden_states = hidden_states[0]  # Shape: [num_layers, B, prefix_len, hidden_dim]
                prefix_hidden_states = torch.stack(prefix_hidden_states, dim=0)  # Shape: [num_layers, B, prefix_len, hidden_dim]

                # Step 1: Convert list of lists to a tensor [num_new_tokens, num_layers, B, 1, hidden_dim]
                new_hidden_states = torch.stack([
                    torch.stack(layer_hiddens, dim=0)
                    for layer_hiddens in hidden_states[1:]
                ], dim=0) # 

                # Step 2: Remove singleton dimension and transpose to [num_layers, B, num_new_tokens, hidden_dim]
                new_hidden_states = new_hidden_states.squeeze(3).permute(1, 2, 0, 3)  # [num_layers, B, num_new_tokens, hidden_dim]

                # Concatenate prefix and new tokens
                combined_hidden_states = torch.cat([
                    prefix_hidden_states,  # [num_layers, B, prefix_len, hidden_dim]
                    new_hidden_states      # [num_layers, B, num_new_tokens, hidden_dim]
                ], dim=2)  # Shape: [num_layers, B, total_len, hidden_dim]
        
        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer
            end_layer = self.config.framework.layer_qformer.qformer_end_layer
            latent_features = []
            # TODO 上面为可读性，牺牲了速度, 稳定后可以考虑 只转换需要用的feature
            for i in range(start_layer, end_layer):
                latent_features.append(combined_hidden_states[i]) # 
            action_latent_feature = self.layer_qformer(latent_features) # [B, 64, D_action]
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
            normalized_actions = samples.cpu().numpy()
            # Un-normalize Actions --> 这个信息应该集成在哪里，能够能够取消动态
            # unnorm_key = self.config.datasets.vla_data.data_mix
            # raw_actions = self.unnormalize_actions(normalized_actions, self.norm_stats[unnorm_key]) # TODO 这里应该是一个函数， 但是现在是放在模型里面， 不应该，这个是和具体函数绑定的   
            # TODO unnormalize_actions 会因为数据集不一样而不一样， 这个动态不应该是 framework应该控制的？ 但是 framework 有必须要知道自己的 是什么action？
        return decoded_sequences, normalized_actions # B, action_chunk, action_dim
    
    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将归一化的动作转换为原始动作空间。
        
        :param normalized_actions: 归一化的动作数组，形状为 [B, T, D]。
        :param action_norm_stats: 包含动作归一化统计信息的字典，必须包含以下键：
            - "q01": 动作的第 1 百分位值。
            - "q99": 动作的第 99 百分位值。
            - "mask": 可选，布尔数组，用于标记哪些动作需要反归一化。
        :return: 反归一化后的动作数组，形状与输入 `normalized_actions` 相同。
        """
        # TODO 这个应该是 要存成一个processor 就是 LLM 中的 tokenizer
        # 获取统计信息
        # Un-normalize Actions        
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["max"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.zeros_like(normalized_actions)
        actions[:, :7] = np.where(
            mask,
            0.5 * (normalized_actions[:, :7] + 1) * (action_high[:7] - action_low[:7]) + action_low[:7],
            normalized_actions[:, :7],
        )
        actions[:, 6] = normalized_actions[:, 6]

        return actions
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
        model_config.trainer.pretrained_checkpoint = None # 为了加快加载速度，避免重复加载， TODO 其实不应该在initial的位置设置 load_pretrained_backbones
        qwenQFormerACT = build_model_framework(model_config) 
        # set for action un-norm
        qwenQFormerACT.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu") #["model"]
        logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(qwenQFormerACT.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        # ✅ 1. 加载匹配的权重
        for key in checkpoint_keys:
            if key in model_keys:
                try:
                    qwenQFormerACT.state_dict()[key].copy_(model_state_dict[key])
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to copy weight for key '{key}': {e}")
            else:
                logger.warning(f"⚠️ Checkpoint has unknown key '{key}' (not in model). Ignoring.")

        # ✅ 2. 反向检查：模型中有但 checkpoint 中缺失的
        missing_keys = model_keys - checkpoint_keys # TODO 这里之后要考虑 nontrainable params --> 我觉得没必要省存储空间
        for key in sorted(missing_keys):
                logger.warning(f"⚠️ Model expects key '{key}' but it's missing in checkpoint.")
        miss_keys = list(missing_keys) + list(checkpoint_keys - model_keys)

        # **确保模型在 GPU 上**
        qwenQFormerACT = qwenQFormerACT.to("cuda") # TODO 其实不应该是这里管理to GPU的
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

    def get_action_stats(self, unnorm_key=None):# 这个要来了任何和 lerobot 格式的对齐
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

# TODO 写一个build model 函数

def build_model_framework(config: dict = {}) -> QwenQFormerDiT:
    # TODO  实现和 config 对应的 load 逻辑

    model = QwenQFormerDiT(config=config)

    return model


def read_mode_config(pretrained_checkpoint):
    if os.path.isfile(pretrained_checkpoint):
        logger.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

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
    else:
        logger.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
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
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client() 
    samples = {}

    config_yaml = "llavavla/conf/qwenvla_cotrain.yaml"
    cfg = OmegaConf.load(config_yaml)

    dino = get_dino_model()
    dino.body.get_image_transform()
    model_framework = build_model_framework(cfg)
    print(model_framework)
    # model_framework(samples)
    pass

    # git remote add gitee https://gitee.pjlab.org.cn/L2/MultimodalVLA/llavavla.git
    # git push -u gitee master