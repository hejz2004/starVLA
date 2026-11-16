"""Prepare fresh MV2-register4 LIBERO runs from pinned DROID milestones."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.droid.tools.prepare_native_control import replace_once, rewrite_command, snapshot, write_json


def configurable_entry(source):
    text = source.read_text()
    text = replace_once(text, "    output=Path(cfg.run_root_dir)/cfg.run_id;output.mkdir(parents=True,exist_ok=True)",
                        """    output=Path(cfg.run_root_dir)/cfg.run_id;output.mkdir(parents=True,exist_ok=True)
    micro_batch=int(cfg.datasets.vla_data.per_device_batch_size)
    global_batch=4*micro_batch*int(cfg.trainer.gradient_accumulation_steps)""")
    text = replace_once(text, "        vision=model.qwen_vl_interface.model.model.visual",
                        """        vision=model.qwen_vl_interface.model.model.visual
        assert type(model.qwen_vl_interface.model).__module__=='starVLA.model.modules.vlm.qwen3_vl_cross_view.modeling_qwen3_vl'
        assert config.framework.qwenvl.cross_view_attention.enabled
        assert all(p.requires_grad for p in model.parameters())""")
    text = replace_once(text, "        assert int(ready['source_step'])==int(cfg.transfer.source_step)",
                        """        assert int(ready['source_step'])==int(cfg.transfer.source_step)
        assert ready.get('architecture','mv2-register4')=='mv2-register4'
        assert ready['register_shape']==[4,1024]""")
    text = replace_once(text, "self._calculate_total_batch_size()==512", "self._calculate_total_batch_size()==global_batch")
    text = text.replace("global_batch=512", "global_batch=global_batch")
    text = replace_once(text, "updates=16,micro_batch=128", "updates=16,micro_batch=micro_batch")
    text = replace_once(text, "dict(status='complete',steps=7500,source_droid_step=",
                        "dict(status='complete',architecture='mv2-register4',registers_per_view=4,steps=int(cfg.trainer.max_train_steps),source_droid_step=")
    ast.parse(text)
    return text


def prepare(reference, native_plan_path, destination, stamp):
    if destination.exists():
        raise FileExistsError(destination)
    reference_plan = json.loads((reference / "plan.json").read_text())
    native_plan = json.loads(native_plan_path.read_text())
    destination.mkdir(parents=True)
    for name in ("platform", "launcher"):
        (destination / name).mkdir()
    snapshot(reference / "training-code", destination / "training-code")
    snapshot(reference / "evaluation-code", destination / "evaluation-code")
    entry = configurable_entry(reference / "launcher/posttrain_entry.py")
    (destination / "launcher/posttrain_entry.py").write_text(entry)
    job = (reference / "launcher/posttrain_job.py").read_text()
    job = replace_once(job, '"""Eight-GPU DROID epoch: try microbatch 128, fall back through 64 and 32 only on CUDA OOM."""',
                       '"""Four-GPU LIBERO posttraining with the batch and duration from config."""')
    (destination / "launcher/posttrain_job.py").write_text(job)
    shutil.copy2(Path(__file__).with_name("transfer_pipeline.py"), destination / "launcher/orchestrator.py")
    plan = copy.deepcopy(reference_plan)
    plan.update(root=str(destination), architecture="mv2-register4", matched_native_plan=str(native_plan_path),
                previous_recipe_plan=str(reference / "plan.json"))
    plan["posttrain"].update(steps=15000, world_size=4, per_device_batch=64, global_batch=256,
                             warmup_steps=750, save_interval=15000)
    plan["items"] = sorted(plan["items"], key=lambda item: (item["label"] != "droid_final", item["source_step"]))
    equivalence, sources = [], []
    for item in plan["items"]:
        label, step = item["label"], item["source_step"]
        item.pop("cancelled", None)
        item.pop("cancellation", None)
        old_train, old_eval = Path(item["training"]["root"]), Path(item["evaluation"]["root"])
        source = Path(item["source_weights"])
        ready = json.loads(source.with_name("READY.json").read_text())
        assert source.stat().st_size == ready["bytes"] and ready["source_step"] == step
        assert ready["register_shape"] == [4, 1024]
        sources.append(dict(label=label, source_step=step, weights=str(source), bytes=ready["bytes"],
                            sha256=ready["sha256"], immutable_pinned_source=True))
        train = destination / "branches" / label / f"libero-ft-mv2-reg4-{label}-bs64-s15000-{stamp}"
        evaluation = destination / "branches" / label / "libero-plus"
        for name in ("configs", "launcher", "platform", "nodes/node0", "outputs"):
            (train / name).mkdir(parents=True)
        for name in ("configs", "launcher", "platform", "model/final_model", "results"):
            (evaluation / name).mkdir(parents=True)
        (train / "code").symlink_to(destination / "training-code", target_is_directory=True)
        (evaluation / "code").symlink_to(destination / "evaluation-code", target_is_directory=True)
        cfg = yaml.safe_load((old_train / "configs/train.yaml").read_text())
        cfg.update(run_id=train.name, run_root_dir=str(train / "outputs"))
        cfg["datasets"]["vla_data"]["per_device_batch_size"] = 64
        cfg["trainer"].update(max_train_steps=15000, num_warmup_steps=750, save_interval=15000,
                              pretrained_checkpoint=str(source), is_resume=False)
        cfg["experiment"].update(variant="mv2-reg4-droid-libero-4x64-15k", world_size=4,
            sample_budget=3840000, reference_global_batch=256,
            description="Fresh LIBERO15000 4GPU BS64 global256;full DROID MV2-register4 inheritance")
        cfg["transfer"].update(source_weights=str(source), architecture="mv2-register4")
        native_item = next(x for x in native_plan["items"] if x["label"] == label)
        native_cfg = yaml.safe_load((Path(native_item["training"]["root"]) / "configs/train.yaml").read_text())
        assert cfg["datasets"] == native_cfg["datasets"], label
        trainer = copy.deepcopy(cfg["trainer"])
        trainer["pretrained_checkpoint"] = native_cfg["trainer"]["pretrained_checkpoint"]
        assert trainer == native_cfg["trainer"], label
        framework = copy.deepcopy(cfg["framework"])
        framework["qwenvl"]["cross_view_attention"] = None
        assert framework == native_cfg["framework"], label
        assert cfg["seed"] == native_cfg["seed"] == 42
        equivalence.append(dict(label=label, datasets_identical=True, trainer_identical_except_initial_weights=True,
            framework_identical_except_cross_view=True, seed_identical=True, steps=15000, global_batch=256))
        (train / "configs/train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (train / "launcher/train_entry.py").write_text(entry)
        (train / "launcher/job.py").write_text(job)
        shutil.copy2(old_train / "launcher/platform_status.py", train / "launcher/platform_status.py")
        (train / "launcher/platform-command.txt").write_text((old_train / "launcher/platform-command.txt").read_text().replace(str(old_train), str(train)))
        short_label = label.removeprefix("droid_").replace("_", "-")
        command = rewrite_command(old_train / "platform/submit-command.json", old_train, train,
            f"hjz-mv2-r4-{short_label}-libft15k-{stamp.lower()}",
            f"MV2-register4 DROID step{step};fresh LIBERO15000 4GPU BS64 global256;750warmup;Plus10030 seed2026;matched native recipe")
        write_json(train / "platform/submit-command.json", command)
        spec = json.loads((old_eval / "configs/evaluation.json").read_text())
        spec.update(model=f"mv2-reg4-{label}-libero-ft15000", checkpoint=str(evaluation / "model/final_model/pytorch_model.pt"),
            created_utc=stamp, training_root=str(train), training_steps=15000, training_samples=3840000,
            per_device_batch_size=64, global_batch_size=256, architecture="mv2-register4",
            initialization=f"DROID step{step};full MV2 VLM+action+4 registers;fresh optimizer;matched LIBERO stats")
        spec.pop("checkpoint_bytes", None)
        write_json(evaluation / "configs/evaluation.json", spec)
        for name in ("job.py", "run_benchmarks.py", "platform_status.py"):
            shutil.copy2(old_eval / "launcher" / name, evaluation / "launcher" / name)
        (evaluation / "launcher/platform-command.txt").write_text((old_eval / "launcher/platform-command.txt").read_text().replace(str(old_eval), str(evaluation)))
        command = rewrite_command(old_eval / "platform/submit-command.json", old_eval, evaluation,
            f"hjz-mv2-r4-{short_label}-plus15k-{stamp.lower()}",
            f"MV2-register4 DROID step{step} LIBERO-ft15000;full Plus10030 seed2026;4GPU80workers;matched native evaluation")
        write_json(evaluation / "platform/submit-command.json", command)
        item.update(architecture="mv2-register4", posttrain_steps=15000,
                    training=dict(root=str(train), output=str(train / "outputs" / train.name)),
                    evaluation=dict(root=str(evaluation)))
    write_json(destination / "plan.json", plan)
    write_json(destination / "platform/recipe-equivalence.json", dict(matched_native_plan=str(native_plan_path), branches=equivalence))
    write_json(destination / "platform/source-checkpoints.json", sources)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    plan = prepare(args.reference, args.native_plan, args.destination, args.stamp)
    print(json.dumps(dict(root=plan["root"], recipe=plan["posttrain"], submission_order=[x["label"] for x in plan["items"]])))


if __name__ == "__main__":
    main()
