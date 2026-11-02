"""Run with torchrun on two PPUs to check one LR tick per optimizer update."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

from starVLA.training.train_starvla import VLATrainer, accelerator


class TinyActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 1)

    def forward(self, batch):
        inputs, targets = batch
        return {"action_loss": (self.projection(inputs) - targets).square().mean()}


def main():
    assert accelerator.num_processes == 2, accelerator.num_processes
    assert not accelerator.step_scheduler_with_optimizer

    model = TinyActionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=2,
        num_training_steps=6,
        scheduler_specific_kwargs={"min_lr": 1e-6},
    )
    data = DataLoader(TensorDataset(torch.randn(16, 4), torch.randn(16, 1)), batch_size=2)
    model, optimizer, scheduler, data = accelerator.prepare(model, optimizer, scheduler, data)

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator

    for step, (inputs, targets) in enumerate(data, start=1):
        inputs = inputs.to(dtype=next(model.parameters()).dtype)
        targets = targets.to(dtype=inputs.dtype)
        trainer._train_step((inputs, targets))
        assert model.global_steps == step, (accelerator.process_index, step, model.global_steps)
        assert scheduler.scheduler.last_epoch == step, (
            accelerator.process_index, step, scheduler.scheduler.last_epoch
        )
        if step == 3:
            break

    accelerator.wait_for_everyone()
    accelerator.print(f"PASS: {accelerator.num_processes} PPUs, 3 optimizer updates, 3 scheduler steps")


if __name__ == "__main__":
    main()
