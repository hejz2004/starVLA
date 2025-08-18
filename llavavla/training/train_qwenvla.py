"""
train.py
"""
# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from llavavla.training.trainer_utils.metrics import normalize_dotlist_args # TODO 封装成为一个特殊的 arg 类别  --> 参数使用 yaml + .sh 控制， 单例模式
from llavavla.model.framework import build_framework
from llavavla.training.trainer_utils.metrics import TrainerUtils



deepspeed_plugin = DeepSpeedPlugin()# 这个插件是否能使用到 config 的参数呢？ 其实这里应该是可以飞显示用的， 感觉有版本问题 #zero_stage=2, gradient_accumulation_steps=1 ：v2: hf_ds_config="scripts/run_scripts/ds_config.yaml"
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state) # TODO 之后要移动到trainer 内部， --> 直接搬LLaVA trainer

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger
logger = get_logger(__name__)

def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    return fast_tokenizer



def setup_directories(cfg) -> Path:
    """创建输出目录并保存配置"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        # 创建输出目录和检查点目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        
        # 保存配置
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, \
                open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
        
    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """构建模型框架"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)
    
    return model

# 这里的变化需要📦封装 Dataloader
from llavavla.dataloader import build_dataloader

def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """准备训练数据"""
    # VLA 数据加载器
    logger.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader( # 这个写在dataload.py 内部
        cfg=cfg)
    
    # 拒绝自动分发 # TODO 应该写到 accelerator config
    accelerator.dataloader_config.dispatch_batches =  False
    dist.barrier()
    # vla_train_dataloader.dataset_statistics = statistics_data # eval 的时候使用 norm actions
    return vla_train_dataloader

def setup_optimizer_and_scheduler(
    model, cfg
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """设置优化器和学习率调度器"""
    # 初始化优化器
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    
    
    # 打印优化器组信息
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    
    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # 最小学习率
    )
    
    # TODO mv to trainer
    # # 准备所有组件
    # (model, optimizer, vla_train_dataloader, vlm_train_dataloader) = accelerator.prepare(
    #     model, optimizer, vla_train_dataloader, vlm_train_dataloader
    # )
    
    return optimizer, lr_scheduler

class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator): # TODO @JinhuiYE 是否考虑和 VLAM 合并
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        # 训练状态跟踪
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, 'seed') else rank + 3047
        set_seed(seed)

        # 加载预训练权重
        if (hasattr(self.config.trainer, 'pretrained_checkpoint') and self.config.trainer.pretrained_checkpoint):
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = self.config.trainer.reload_modules if hasattr(self.config.trainer, 'reload_modules') else None
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
        
        # 冻结参数
        freeze_modules = ( # 我觉得全局就应该只有一个config， 使用没必要相对路径
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules) # TODO 思考一下self.config 是全局传参数， 还是相对传参数？

        #  打印模型的可训练参数： --> TODO 他应该是要最后 总结check的， 考虑集权管理
        self.print_trainable_parameters(self.model)

        # 初始化分布式训练组件
        # self.accelerator.gradient_accumulation_steps = self.config.trainer.gradient_accumulation_steps
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )


        self._init_wandb()
        self._init_checkpointing()
    

    def _calculate_total_batch_size(self):
        """计算全局批量大小"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
    
    def _init_wandb(self):
        """初始化Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
    
    def _init_checkpointing(self):
        """初始化检查点目录"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # 恢复训练状态
        # 要判断是否有self.config.trainer.pretrained_checkpoint
        if pretrained_checkpoint and is_resume: # TODO 这里还没能够保存state, 思考是否必要 (state 的存储太大了， 需要实现keep last/best 的逻辑， 包括ckpt)
            self._load_checkpoint(self.config.resume_from_checkpoint)
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # TODO: 恢复训练步数和其他状态
    
    def _save_checkpoint(self):
        """保存当前训练状态"""

        if accelerator.is_main_process:
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # 保存模型状态
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            
            # 保存训练元数据
            summary_data = {
                "steps": self.completed_steps,
                # TODO: 添加其他需要保存的训练状态
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """记录训练指标"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0: # 有些参数应该是需要intial 给 class 的了
            if self.accelerator.is_main_process:
                # 计算梯度范数
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                metrics["grad_norm"] = total_norm ** 0.5
                
                # 添加学习率
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                
                # 添加epoch信息
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                
                # 记录到W&B
                wandb.log(metrics, step=self.completed_steps)
                # 调试输出
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")
    
    def _create_data_iterators(self):
        """创建数据迭代器"""
        # TODO 考虑如何兼容不同的模式
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)
    
    def _get_next_batch(self):
        """获取下一批数据（自动处理数据循环）"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:            # 需要改变trainer 的seed --> 其实不要固定seed 就不会有这些问题 # TODO 未来要看怎么样自动处理这些事情。
            # 先判断是否有这个 self.vla_epoch_count
            if not hasattr(self, 'vla_epoch_count'):
                self.vla_epoch_count = 0
            # TODO 需要检验是否 生效
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        
        # try:
        #     batch_vlm = next(self.vlm_iter) # TODO 首尾循环应该是dataset 自己的功能， 这里是考虑到很多人的dataset 是没有这个功能的
        # except StopIteration: 
        #     if not hasattr(self, 'vlm_epoch_count'):
        #         self.vlm_epoch_count = 0
        #     self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(
        #         self.vlm_train_dataloader, self.vlm_epoch_count
        #     )
        #     batch_vlm = next(self.vlm_iter)

        return batch_vla #, batch_vlm
    
    def train(self):
        """执行训练循环"""
        # 打印训练配置
        self._log_training_config()
        
        # 准备数据迭代器
        self._create_data_iterators()
        
        # 创建进度条
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        
        # 主训练循环
        while self.completed_steps < self.config.trainer.max_train_steps:
            # 获取数据批次
            batch_vla = self._get_next_batch()
            
            # 执行训练步骤
            step_metrics = self._train_step(batch_vla)
            
            # 更新进度
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            # 评估模型
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # 记录指标
            self._log_metrics(step_metrics)
            
            

            # 保存检查点
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

                
            
            # 检查终止条件
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
        
        # 训练结束处理
        self._finalize_training()
    
        # 执行评估步骤
    def eval_action_model(self, step_metrics:dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """
        
        if self.accelerator.is_main_process:
            
            examples = self._get_next_batch()
            
            score = 0.0 # 想办法看看证明变成batch 推理
            num_samples = len(examples)

            # @Jinhui TBD TODO 
            images = [example["image"] for example in examples]  #  TODO check 是什么
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples] #label

            # Predict actions using the model
            predicted_solutions, normalized_actions = self.model.predict_action( # TODO 这里有 模型方法 依赖关系, 如果你要保持trainer的独立性，这里应该怎么设计？
                images=images,
                instructions=instructions,
                use_ddim=True,
                num_ddim_steps=20)
            
            # 提前转换 actions 为 numpy.ndarray
            actions = np.array(actions)  # 将 actions 转换为 numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score
        pass
        dist.barrier()  # 确保所有进程同步 TODO 看看是否需要让其他进程等
        return step_metrics


    def _log_training_config(self):
        """记录训练配置"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f" Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

        # TODO 这里应该打印全部 训练中关键的信息： model size, freeze， lr group and so on.
    
    def _train_step(self, batch_vla, batch_vlm=None):
        """执行单个训练步骤"""
        # TODO: 实现梯度累积 @Yioutpi
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            # VLA任务前向传播
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action_loss, action_cot_loss = self.model.forward(batch_vla)
                total_loss = action_loss + action_cot_loss * self.config.trainer.loss_scale.vlm
            
            # VLA反向传播
            self.accelerator.backward(total_loss)
            
            # # VLM任务前向传播
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            #     vlm_output = self.model.qwen_vl_interface(**batch_vlm)
            #     vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
            
            # VLM反向传播
            # self.accelerator.backward(vlm_loss)
            
            # 梯度裁剪
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.gradient_clipping
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.lr_scheduler.step()
        
        return {
            "action_dit_loss": action_loss.item(),
            "action_cot_loss": action_cot_loss.item(),
            # "vlm_loss": vlm_loss.item(),
        }
    
    def _finalize_training(self):
        """训练结束处理"""
        # 保存最终模型
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        
        # 关闭W&B
        if self.accelerator.is_main_process:
            wandb.finish()
        
        self.accelerator.wait_for_everyone()

from llavavla.training.trainer_utils.metrics import build_param_lr_groups
def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # 创建输出目录并保存配置
    output_dir = setup_directories(cfg=cfg)
    # 构建模型
    vla = build_framework(cfg)
    # 准备数据
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    
    # 设置优化器和调度器
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    
    # 创建训练器
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator
    )
    
    # 执行训练前的准备
    trainer.prepare_training()
    # 执行训练
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="llavavla/conf/qwenact.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
