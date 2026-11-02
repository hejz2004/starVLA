"""Normal trainer + observational checks for the VLM-init/LayerNorm experiment."""


def main():
    import json
    import runpy
    from pathlib import Path
    import torch
    from torch import nn
    import starVLA.training.train_starvla as training

    original_build = training.build_framework

    def audited_build(cfg):
        model = original_build(cfg)
        rank = training.accelerator.process_index
        folder = Path(cfg.run_root_dir) / cfg.run_id / 'telemetry'
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / f'layernorm_rank_{rank}.jsonl'

        def record(payload):
            payload['rank'] = rank
            with log.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')
            if rank == 0:
                print('[layernorm-audit] ' + json.dumps(payload), flush=True)

        head = model.action_model
        assert isinstance(head.decoder_input_norm, nn.LayerNorm)
        assert not head.decoder_input_norm.elementwise_affine
        blocks = [b for b in head.model.transformer_blocks if b.cross_attention_dim is not None]
        assert blocks
        for block in blocks:
            assert isinstance(block.cross_condition_norm, nn.LayerNorm)
            assert not block.cross_condition_norm.elementwise_affine
        assert not cfg.framework.action_model.use_state
        assert cfg.framework.action_model.repeated_diffusion_steps == 2
        record(dict(event='model_built', decoder_norm='layer_norm', cross_norm='layer_norm',
            cross_blocks=len(blocks), norm_trainable_parameters=0, use_state=False, repeat=2,
            num_parameters=sum(p.numel() for p in model.parameters())))

        def loaded(module, incompatible):
            record(dict(event='checkpoint_loaded', missing_keys=list(incompatible.missing_keys),
                unexpected_keys=list(incompatible.unexpected_keys)))
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError('Full UMI checkpoint does not exactly match model parameter keys')
        model.register_load_state_dict_post_hook(loaded)

        if rank == 0:
            def vector_stats(value):
                x = value.detach()[0, 0].float()
                return dict(absmax=x.abs().max().item(), rms=x.square().mean().sqrt().item(),
                    mean=x.mean().item(), std=x.std(unbiased=False).item())

            def norm_probe(name, module):
                handle = None
                def hook(mod, inputs, output):
                    record(dict(event='first_norm_forward', name=name,
                        before=vector_stats(inputs[0]), after=vector_stats(output)))
                    handle.remove()
                handle = module.register_forward_hook(hook)
            norm_probe('first_cross_condition', blocks[0].cross_condition_norm)
            norm_probe('action_decoder_input', head.decoder_input_norm)

            first = {'done': False}
            def input_probe(mod, args, kwargs):
                if first['done']:
                    return
                actions = args[1]
                mask = kwargs.get('action_mask')
                values = actions.detach().float().square()
                if mask is not None:
                    values = values[mask.bool()]
                first['expected_zero_loss'] = 1.0 + values.mean().item()
                first['head_batch'] = actions.shape[0]
                first['masked_loss'] = mask is not None
            head.register_forward_pre_hook(input_probe, with_kwargs=True)

            def output_probe(mod, args, output):
                if first['done']:
                    return
                first['done'] = True
                record(dict(event='first_head_loss', loss=output.detach().item(),
                    expected_zero_prediction_loss_over_gaussian_noise=first['expected_zero_loss'],
                    head_batch=first['head_batch'], masked_loss=first['masked_loss']))
            head.register_forward_hook(output_probe)
        return model

    training.build_framework = audited_build
    runpy.run_path(str(Path(__file__).with_name('run_umi_smoke_memory.py')), run_name='__main__')


if __name__ == '__main__':
    main()
