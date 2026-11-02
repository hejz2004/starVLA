"""Run the normal trainer with per-rank, per-step memory telemetry."""


def main():
    # Keep imports inside __main__: spawned data workers must not initialize DDP.
    import argparse
    import json
    import time
    from pathlib import Path

    import torch
    from omegaconf import OmegaConf
    import starVLA.training.train_starvla as training
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    cfg = training.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    output = Path(cfg.run_root_dir) / cfg.run_id
    telemetry = output / 'telemetry'
    telemetry.mkdir(parents=True, exist_ok=True)
    rank = training.accelerator.process_index
    device = training.accelerator.local_process_index
    path = telemetry / f'rank_{rank}.jsonl'
    if path.exists():
        raise FileExistsError(f'Refuse to replace telemetry: {path}')
    started = time.perf_counter()

    def memory():
        free, total = torch.cuda.mem_get_info(device)
        return dict(device=device, total_gib=total/2**30,
            device_used_gib=max(0,(total-free)/2**30),
            allocated_gib=torch.cuda.memory_allocated(device)/2**30,
            reserved_gib=torch.cuda.memory_reserved(device)/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)

    def record(payload):
        payload.update(rank=rank, elapsed_seconds=time.perf_counter()-started)
        with path.open('a') as stream:
            stream.write(json.dumps(payload)+'\n')

    original = training.VLATrainer._train_step

    def measured_step(self, batch_vla, batch_vlm=None):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        try:
            metrics = original(self, batch_vla, batch_vlm)
            torch.cuda.synchronize(device)
        except Exception as error:
            record(dict(event='step_error',completed_steps=self.completed_steps,
                batch_size=len(batch_vla),error_type=type(error).__name__,error=str(error),**memory()))
            raise
        record(dict(event='step',step=self.completed_steps+int(self.accelerator.sync_gradients),
            batch_size=len(batch_vla),sync_gradients=self.accelerator.sync_gradients,
            step_seconds=time.perf_counter()-step_started,**metrics,**memory()))
        return metrics

    training.VLATrainer._train_step = measured_step
    record(dict(event='start',world_size=training.accelerator.num_processes,
        repeat=cfg.framework.action_model.repeated_diffusion_steps,
        per_device_batch_size=cfg.datasets.vla_data.per_device_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        max_train_steps=cfg.trainer.max_train_steps,**memory()))
    try:
        training.main(cfg)
    except Exception as error:
        record(dict(event='run_error',error_type=type(error).__name__,error=str(error),**memory()))
        raise
    else:
        record(dict(event='complete',**memory()))


if __name__ == '__main__':
    main()
