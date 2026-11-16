"""Freeze a native-architecture DROID control using an existing MV2 recipe.

This prepares artifacts only. Submission and supervisor activation are separate,
so the exact configuration and platform dry runs can be inspected first.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import yaml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one template occurrence: {old[:100]!r}")
    return text.replace(old, new, 1)


def snapshot(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))


def pretrain_entry(source, reference_initialization):
    text = source.read_text()
    text = replace_once(text, "Train DA3-style QwenPI with audited pretrained weights and per-view registers.",
                        "Train the native QwenPI DROID control with the matched MV2 recipe.")
    text = replace_once(text, "'starVLA.model.modules.vlm.qwen3_vl_cross_view.modeling_qwen3_vl'",
                        "'transformers.models.qwen3_vl.modeling_qwen3_vl'")
    begin = text.index("        assert tuple(vision.global_attention_indexes)")
    end = text.index("        probes = {}", begin)
    text = text[:begin] + """        assert config.framework.qwenvl.cross_view_attention is None
        assert config.framework.qwenvl.load_weights
        assert not hasattr(vision, 'register_tokens')
        assert not any(any(key in name for key in ('register_tokens', 'interaction_stages'))
                       for name, _ in model.named_parameters())
        assert all(p.requires_grad for p in model.parameters())
""" + text[end:]
    text = replace_once(text, "architecture='da3_late_alternating_register4', cross_view_enabled=True,",
                        "architecture='native', cross_view_enabled=False,")
    text = replace_once(text, "global_vision_layers=list(vision.global_attention_indexes), registers_per_view=4,",
                        "global_vision_layers=[], registers_per_view=0,")
    text = replace_once(text, "learned_register_parameters=4096,", "learned_register_parameters=0,")
    marker = "        publish(output / f'initialization-rank{rank}.json',"
    text = replace_once(text, marker, f"""        reference = json.loads(Path({str(reference_initialization)!r}).read_text())
        assert digest.hexdigest() == reference['action_head_sample_sha256'], 'Action initialization differs from MV2'
{marker}""")
    text = replace_once(text, "        register = model.qwen_vl_interface.model.model.visual.register_tokens.weight\n        assert id(register) in expected\n", "")
    text = replace_once(text, "all_trainable_parameters_covered=True, registers_in_optimizer=True,",
                        "all_trainable_parameters_covered=True, registers_in_optimizer=False, architecture='native',")
    text = replace_once(text, "        assert 'qwen_vl_interface.model.model.visual.register_tokens.weight' in state",
                        "        assert not any('register_tokens' in name for name in state)")
    text = replace_once(text, "publish(output / 'TRAINING_COMPLETE.json', dict(status='complete', steps=",
                        "publish(output / 'TRAINING_COMPLETE.json', dict(status='complete', architecture='native', registers_per_view=0, steps=")
    return text


def posttrain_entry(source):
    text = source.read_text()
    text = replace_once(text, "        assert vision.num_register_tokens==4 and tuple(vision.global_attention_indexes)==(16,18,20,22)",
                        """        vlm=model.qwen_vl_interface.model
        assert type(vlm).__module__=='transformers.models.qwen3_vl.modeling_qwen3_vl'
        assert config.framework.qwenvl.cross_view_attention is None
        assert not hasattr(vision,'register_tokens')
        assert not any(any(key in name for key in ('register_tokens','interaction_stages'))
                       for name,_ in model.named_parameters())
        assert all(p.requires_grad for p in model.parameters())""")
    text = replace_once(text, "        assert int(ready['source_step'])==int(cfg.transfer.source_step)",
                        "        assert int(ready['source_step'])==int(cfg.transfer.source_step)\n        assert ready['architecture']=='native' and ready['register_shape'] is None")
    text = replace_once(text, "            'qwen_vl_interface.model.model.visual.register_tokens.weight',\n", "")
    text = replace_once(text, "full_tensor_equality_checked=rank==0,includes_vlm_action_head_registers=True,",
                        "full_tensor_equality_checked=rank==0,includes_vlm_action_head=True,architecture='native',registers_per_view=0,")
    text = replace_once(text, "        assert all(torch.isfinite(t).all().item() for t in state.values() if torch.is_floating_point(t))",
                        "        assert all(torch.isfinite(t).all().item() for t in state.values() if torch.is_floating_point(t))\n        assert not any('register_tokens' in name for name in state)")
    text = replace_once(text, "dict(status='complete',steps=7500,source_droid_step=",
                        "dict(status='complete',architecture='native',registers_per_view=0,steps=int(cfg.trainer.max_train_steps),source_droid_step=")
    text = replace_once(text, "    output=Path(cfg.run_root_dir)/cfg.run_id;output.mkdir(parents=True,exist_ok=True)",
                        """    output=Path(cfg.run_root_dir)/cfg.run_id;output.mkdir(parents=True,exist_ok=True)
    micro_batch=int(cfg.datasets.vla_data.per_device_batch_size)
    global_batch=4*micro_batch*int(cfg.trainer.gradient_accumulation_steps)""")
    text = replace_once(text, "self._calculate_total_batch_size()==512", "self._calculate_total_batch_size()==global_batch")
    text = text.replace("global_batch=512", "global_batch=global_batch")
    text = replace_once(text, "updates=16,micro_batch=128", "updates=16,micro_batch=micro_batch")
    return text


def rewrite_command(source, old_root, new_root, name, description):
    command = json.loads(source.read_text())
    command = [value.replace(str(old_root), str(new_root)) for value in command]
    command[5] = name
    command[command.index("--description") + 1] = description
    assert "--node" not in command and "--node-scheduling" not in command
    return command


def prepare(reference_pretrain, reference_transfer, root, stamp, posttrain_steps=15000, posttrain_batch=64):
    if posttrain_steps < 1 or posttrain_batch < 1:
        raise ValueError("Posttraining steps and batch size must be positive")
    global_batch = 4 * posttrain_batch
    pretrain = root / f"qwenpi-native-droid-8gpu-bs128-epoch1-{stamp}"
    transfer = root / f"native-droid-libero-transfer-{stamp}"
    if pretrain.exists() or transfer.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    for directory in (pretrain, transfer):
        directory.mkdir(parents=True)
        (directory / "platform").mkdir()
        (directory / "launcher").mkdir()
    for directory in ("configs", "nodes/node0", "outputs"):
        (pretrain / directory).mkdir(parents=True)
    snapshot(reference_pretrain / "code", pretrain / "code")
    (transfer / "training-code").symlink_to(pretrain / "code", target_is_directory=True)
    snapshot(reference_transfer / "evaluation-code", transfer / "evaluation-code")
    config = yaml.safe_load((reference_pretrain / "configs/train-bs128.yaml").read_text())
    config["run_id"] = pretrain.name + "-bs128"
    config["run_root_dir"] = str(pretrain / "outputs")
    config["framework"]["qwenvl"]["cross_view_attention"] = None
    config["experiment"]["variant"] = "native-vlminit-droid-control"
    config_path = pretrain / "configs/train-bs128.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    reference_output = Path(json.loads((reference_pretrain / "ACTIVE_ATTEMPT.json").read_text())["output_dir"])
    (pretrain / "launcher/train_entry.py").write_text(pretrain_entry(
        reference_pretrain / "launcher/train_entry.py", reference_output / "initialization-rank0.json"))
    job = replace_once((reference_pretrain / "launcher/job.py").read_text(), "for batch in (128,64,32):", "for batch in (128,):")
    job = replace_once(job, '"""Eight-GPU DROID epoch: try microbatch 128, fall back through 64 and 32 only on CUDA OOM."""',
                       '"""Native DROID control: eight GPUs, fixed batch128 to match MV2 exactly."""')
    (pretrain / "launcher/job.py").write_text(job)
    shutil.copy2(reference_pretrain / "launcher/platform_status.py", pretrain / "launcher/platform_status.py")
    (pretrain / "launcher/platform-command.txt").write_text((reference_pretrain / "launcher/platform-command.txt").read_text().replace(str(reference_pretrain), str(pretrain)))
    command = rewrite_command(reference_pretrain / "platform/submit-command.json", reference_pretrain, pretrain,
        f"hjz-droid-native-8gpu-bs128-{stamp.lower()}",
        "Native QwenPI DROID control; no cross-view/registers; matched MV2 code,data,seed43 action init;8GPU BS128 global1024;6044 steps;research automatic placement")
    write_json(pretrain / "platform/submit-command.json", command)
    # The supervisor may start before the scheduler allocates GPUs.
    write_json(pretrain / "ACTIVE_ATTEMPT.json", dict(batch_size=128, world_size=8, global_batch=1024,
        config=str(config_path), output_dir=str(pretrain / "outputs" / config["run_id"]),
        log=str(pretrain / "nodes/node0/train-bs128.log"), status="awaiting_platform_start"))
    reference_plan = json.loads((reference_pretrain / "platform/training-plan.json").read_text())
    write_json(pretrain / "platform/training-plan.json", {"128": reference_plan["128"]})
    write_json(pretrain / "platform/reference-distributed-data-check.json", dict(
        reference_run=str(reference_pretrain),
        result=json.loads((reference_pretrain / "platform/distributed-data-check.json").read_text())))
    write_json(pretrain / "platform/source.json", dict(reference_pretraining=str(reference_pretrain),
        architecture="native", only_architecture_change="cross_view_attention=null; official Qwen3VL model; no register tokens",
        action_head_seed=config["experiment"]["action_head_seed"], expected_action_head_sample_sha256=json.loads(
            (reference_output / "initialization-rank0.json").read_text())["action_head_sample_sha256"],
        training_code="exact copy of reference pretraining snapshot", batch_policy="fixed128; preserve global1024 comparison"))

    (transfer / "launcher/posttrain_entry.py").write_text(posttrain_entry(reference_transfer / "launcher/posttrain_entry.py"))
    shutil.copy2(reference_transfer / "launcher/posttrain_job.py", transfer / "launcher/posttrain_job.py")
    shutil.copy2(Path(__file__).with_name("transfer_pipeline.py"), transfer / "launcher/orchestrator.py")
    plan = copy.deepcopy(json.loads((reference_transfer / "plan.json").read_text()))
    plan.update(root=str(transfer), architecture="native", pretraining_root=str(pretrain),
                pretraining_output=str(pretrain / "outputs" / config["run_id"]), pretraining_job=None,
                matched_reference_plan=str(reference_transfer / "plan.json"))
    plan["posttrain"].update(steps=posttrain_steps, per_device_batch=posttrain_batch,
                             global_batch=global_batch, save_interval=posttrain_steps,
                             warmup_steps=round(posttrain_steps * 0.05))
    equivalence = []
    for item in plan["items"]:
        label, step = item["label"], item["source_step"]
        old_train, old_eval = Path(item["training"]["root"]), Path(item["evaluation"]["root"])
        branch = transfer / "branches" / label
        train = branch / f"libero-ft-native-{label}-{stamp}"
        evaluation = branch / "libero-plus"
        for d in ("configs", "launcher", "platform", "nodes/node0", "outputs"):
            (train / d).mkdir(parents=True)
        for d in ("configs", "launcher", "platform", "model/final_model", "results"):
            (evaluation / d).mkdir(parents=True)
        (train / "code").symlink_to(pretrain / "code", target_is_directory=True)
        (evaluation / "code").symlink_to(transfer / "evaluation-code", target_is_directory=True)
        source_weights = transfer / "sources" / label / "pytorch_model.pt"
        cfg = yaml.safe_load((old_train / "configs/train.yaml").read_text())
        reference_cfg = copy.deepcopy(cfg)
        cfg["run_id"] = train.name
        cfg["run_root_dir"] = str(train / "outputs")
        cfg["framework"]["qwenvl"]["cross_view_attention"] = None
        cfg["trainer"]["pretrained_checkpoint"] = str(source_weights)
        cfg["datasets"]["vla_data"]["per_device_batch_size"] = posttrain_batch
        cfg["trainer"].update(max_train_steps=posttrain_steps, num_warmup_steps=round(posttrain_steps * 0.05),
                              save_interval=posttrain_steps)
        cfg["transfer"]["source_weights"] = str(source_weights)
        cfg["transfer"]["architecture"] = "native"
        cfg["transfer"]["source_pretraining_root"] = str(pretrain)
        cfg["experiment"].update(variant="native-droid-libero-control", architecture="native",
            expected_global_vision_layers=[], reference_global_batch=global_batch,
            sample_budget=posttrain_steps * global_batch,
            description=f"{posttrain_steps} LIBERO updates,4GPU BS{posttrain_batch}; fresh optimizer; full native DROID initialization")
        (train / "configs/train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        matched_datasets = copy.deepcopy(cfg["datasets"])
        matched_datasets["vla_data"]["per_device_batch_size"] = reference_cfg["datasets"]["vla_data"]["per_device_batch_size"]
        assert matched_datasets == reference_cfg["datasets"], label
        matched_trainer = copy.deepcopy(cfg["trainer"])
        for key in ("pretrained_checkpoint", "max_train_steps", "num_warmup_steps", "save_interval"):
            matched_trainer[key] = reference_cfg["trainer"][key]
        assert matched_trainer == reference_cfg["trainer"], label
        matched_framework = copy.deepcopy(cfg["framework"])
        matched_framework["qwenvl"]["cross_view_attention"] = reference_cfg["framework"]["qwenvl"]["cross_view_attention"]
        assert matched_framework == reference_cfg["framework"], label
        equivalence.append(dict(label=label, datasets_identical_except_batch=True,
            trainer_identical_except_source_steps_warmup_save=True, framework_identical_except_cross_view=True,
            posttrain_steps=posttrain_steps, per_device_batch=posttrain_batch, global_batch=global_batch,
            sample_budget=posttrain_steps * global_batch))
        shutil.copy2(transfer / "launcher/posttrain_entry.py", train / "launcher/train_entry.py")
        shutil.copy2(transfer / "launcher/posttrain_job.py", train / "launcher/job.py")
        shutil.copy2(old_train / "launcher/platform_status.py", train / "launcher/platform_status.py")
        (train / "launcher/platform-command.txt").write_text((old_train / "launcher/platform-command.txt").read_text().replace(str(old_train), str(train)))
        command = rewrite_command(old_train / "platform/submit-command.json", old_train, train,
            f"hjz-native-{label.replace('_','-')}-libft-{stamp.lower()}",
            f"Native DROID step{step} full-model init;no cross-view/registers;LIBERO{posttrain_steps} 4GPU BS{posttrain_batch} global{global_batch};fresh optimizer;Plus10030 seed2026")
        write_json(train / "platform/submit-command.json", command)
        spec = json.loads((old_eval / "configs/evaluation.json").read_text())
        spec.update(model=f"native-{label}-libero-ft{posttrain_steps}", checkpoint=str(evaluation / "model/final_model/pytorch_model.pt"),
                    created_utc=stamp, training_root=str(train), initialization=f"Native DROID step{step}; no cross-view/registers; full VLM+action inheritance;fresh optimizer;matched LIBERO stats",
                    architecture="native", droid_pretraining_step=step, training_steps=posttrain_steps,
                    training_samples=posttrain_steps * global_batch, per_device_batch_size=posttrain_batch,
                    global_batch_size=global_batch)
        spec.pop("checkpoint_bytes", None)
        write_json(evaluation / "configs/evaluation.json", spec)
        for name in ("job.py", "run_benchmarks.py", "platform_status.py"):
            shutil.copy2(old_eval / "launcher" / name, evaluation / "launcher" / name)
        (evaluation / "launcher/platform-command.txt").write_text((old_eval / "launcher/platform-command.txt").read_text().replace(str(old_eval), str(evaluation)))
        command = rewrite_command(old_eval / "platform/submit-command.json", old_eval, evaluation,
            f"hjz-native-{label.replace('_','-')}-plus-{stamp.lower()}",
            f"Native DROID step{step} LIBERO-ft{posttrain_steps};full Plus10030 seed2026;4GPU20workers each;matched protocol;automatic placement")
        write_json(evaluation / "platform/submit-command.json", command)
        item.update(architecture="native", source_weights=str(source_weights), posttrain_steps=posttrain_steps,
                    training=dict(root=str(train), output=str(train / "outputs" / train.name)),
                    evaluation=dict(root=str(evaluation)))
    write_json(transfer / "plan.json", plan)
    write_json(transfer / "platform/recipe-equivalence.json", dict(reference=str(reference_transfer), branches=equivalence))
    reference_config = yaml.safe_load((reference_pretrain / "configs/train-bs128.yaml").read_text())
    for key in ("datasets", "trainer"):
        assert reference_config[key] == config[key], key
    framework = copy.deepcopy(config["framework"])
    framework["qwenvl"]["cross_view_attention"] = reference_config["framework"]["qwenvl"]["cross_view_attention"]
    assert framework == reference_config["framework"]
    write_json(pretrain / "platform/recipe-equivalence.json", dict(reference=str(reference_pretrain),
        datasets_identical=True, trainer_identical=True, framework_identical_except_cross_view=True,
        code_files_identical=True, anchors=6188564, optimizer_steps=6044, global_batch=1024,
        posttraining_plan=str(transfer / "plan.json")))
    write_json(pretrain / "LIBERO_FOLLOWUPS.json", dict(root=str(transfer), plan=str(transfer / "plan.json"),
        status=str(transfer / "STATUS.json"), milestones=[2000,4000,6044], posttrain_steps=posttrain_steps,
        posttrain_world_size=4, posttrain_per_device_batch=posttrain_batch, posttrain_global_batch=global_batch,
        evaluation="full LIBERO-Plus10030 seed2026", architecture="native"))
    for p in (pretrain / "launcher").glob("*.py"):
        compile(p.read_text(), str(p), "exec")
    for p in (transfer / "launcher").glob("*.py"):
        compile(p.read_text(), str(p), "exec")
    return pretrain, transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pretrain", type=Path, required=True)
    parser.add_argument("--reference-transfer", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--posttrain-steps", type=int, default=15000)
    parser.add_argument("--posttrain-batch", type=int, default=64)
    args = parser.parse_args()
    pretrain, transfer = prepare(args.reference_pretrain, args.reference_transfer, args.root, args.stamp,
                                 args.posttrain_steps, args.posttrain_batch)
    print(json.dumps(dict(pretraining_root=str(pretrain), pipeline_root=str(transfer))))


if __name__ == "__main__":
    main()
