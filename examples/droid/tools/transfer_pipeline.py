"""Pin DROID milestones, submit independent LIBERO training, then full Plus.

The detached supervisor uses completion records, immutable hardlinks, and
idempotent platform receipts. It never reserves GPUs while waiting for weights.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def completed_steps(path):
    path = Path(path)
    if not path.exists():
        return set()
    # Ignore an in-progress append; only newline-terminated records commit a save.
    lines = path.read_text().splitlines(keepends=True)
    return {int(json.loads(line)["steps"]) for line in lines if line.endswith("\n") and line.strip()}


def state_schema_metadata(state, architecture="mv2-register4"):
    """Reject checkpoints from the other architecture before any paid job starts."""
    if architecture not in ("mv2-register4", "native"):
        raise ValueError(f"Unsupported architecture: {architecture}")
    required = ["qwen_vl_interface.model.model.visual.blocks.23.attn.qkv.weight",
                "qwen_vl_interface.model.model.language_model.layers.27.mlp.down_proj.weight"]
    if not all(name in state for name in required):
        raise ValueError("Checkpoint lacks VLM parameters")
    register_name = "qwen_vl_interface.model.model.visual.register_tokens.weight"
    register_names = [name for name in state if "register_tokens" in name]
    if architecture == "native":
        if register_names:
            raise ValueError("Native checkpoint must not contain register tokens")
        register_shape = None
    else:
        if register_names != [register_name] or tuple(state[register_name].shape) != (4, 1024):
            raise ValueError("Expected four register tokens per view")
        register_shape = [4, 1024]
    if not any(name.startswith("action_model.action_decoder.") and value.ndim == 2 and value.shape[0] == 16
               for name, value in state.items()):
        raise ValueError("Expected UMI16 action decoder")
    return dict(architecture=architecture, tensors=len(state), register_shape=register_shape,
                shape_digest=hashlib.sha256(json.dumps({k: list(v.shape) for k, v in state.items()}, sort_keys=True).encode()).hexdigest())


def checkpoint_metadata(path, *, architecture="mv2-register4"):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    path = Path(path)
    before = path.stat()
    if before.st_size < 1_000_000_000:
        raise ValueError(f"Checkpoint too small: {path}")
    with zipfile.ZipFile(path) as archive:
        if not archive.namelist():
            raise ValueError("Empty checkpoint archive")
    with FakeTensorMode():
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    details = state_schema_metadata(state, architecture)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during inspection")
    return dict(bytes=after.st_size, **details)


def pin_weights(source, destination, metadata, *, inspect=None):
    """A hardlink survives the original trainer's keep-last pruning."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".pinning")
    if not destination.exists():
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError as error:
            import errno
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    details = (inspect(destination) if inspect is not None else
               checkpoint_metadata(destination, architecture=metadata.get("architecture", "mv2-register4")))
    sha = digest_file(destination)
    record = dict(metadata, source=str(source), weights=str(destination), sha256=sha, **details)
    publish(destination.parent / "provenance.json", record)
    return record


def run_command(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def submit(stage, *, execute=run_command):
    root = Path(stage["root"])
    receipt = root / "platform/submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    command = json.loads((root / "platform/submit-command.json").read_text())
    if any(flag in command for flag in ("--node", "--node-scheduling")):
        raise ValueError("Physical node restrictions are forbidden")
    name = command[5]
    # Reconcile a timed-out create before retrying; do not duplicate paid jobs.
    query = execute(["kmctl", "--context", "hjz", "--request-timeout", "15s", "get", "training-jobs",
                     "--keyword", name, "-o", "json"], timeout=25)
    query.check_returncode()
    matches = [item for item in (json.loads(query.stdout) or []) if item["Meta"]["Name"] == name]
    if len(matches) > 1:
        raise ValueError(f"Multiple jobs already named {name}")
    if matches:
        value = dict(Id=matches[0]["Meta"]["Id"], recovered_existing=True)
    else:
        group = command[command.index("--node-group") + 1]
        capacity = execute(["kmctl", "--context", "hjz", "get", "nodes", "--node-group", group,
                            "--page-size", "100", "-o", "json"], timeout=30)
        capacity.check_returncode()
        (root / "platform/nodes-before-submit.json").write_text(capacity.stdout)
        publish(root / "platform/submission-attempted.json", dict(name=name, time=time.time()))
        result = execute(command, timeout=45)
        (root / "platform/submission.stdout").write_text(result.stdout)
        (root / "platform/submission.stderr").write_text(result.stderr)
        result.check_returncode()
        value = json.loads(result.stdout)
    publish(receipt, value)
    return value


def source_ready(plan, item):
    pretrain = Path(plan["pretraining_root"])
    active = json.loads((pretrain / "ACTIVE_ATTEMPT.json").read_text())
    if active["output_dir"] != plan["pretraining_output"]:
        raise ValueError("Pretraining attempt changed; refusing to mix milestone sources")
    output = Path(plan["pretraining_output"])
    step = int(item["source_step"])
    if step not in completed_steps(output / "summary.jsonl"):
        return False
    if item.get("final_source", False):
        complete = output / "TRAINING_COMPLETE.json"
        if not complete.exists():
            return False
        value = json.loads(complete.read_text())
        if value["status"] != "complete" or int(value["steps"]) != step or not value["all_saved_tensors_finite"]:
            raise ValueError("Final pretraining completion record mismatch")
    return (output / f"checkpoints/steps_{step}_pytorch_model.pt").is_file()


def pin_source(plan, item):
    destination = Path(item["source_weights"])
    ready = destination.parent / "READY.json"
    if ready.exists():
        value = json.loads(ready.read_text())
        if value.get("architecture", "mv2-register4") != plan.get("architecture", "mv2-register4"):
            raise ValueError("Pinned source architecture differs from plan")
        if not destination.is_file() or destination.stat().st_size != value["bytes"]:
            raise ValueError("Pinned source changed or disappeared")
        return value
    if not destination.exists() and not source_ready(plan, item):
        return None
    output = Path(plan["pretraining_output"])
    source = output / f"checkpoints/steps_{item['source_step']}_pytorch_model.pt"
    value = pin_weights(source, destination, dict(source_step=item["source_step"], source_run=str(output),
                        architecture=plan.get("architecture", "mv2-register4"),
                        action_contract="droid-command-current-eef-xyzw-open-v1", pinned_at=time.time()))
    for name in ("config.full.yaml", "dataset_statistics.json", "dataset_access.json"):
        shutil.copy2(output / name, destination.parent / name)
    publish(ready, value)
    return value


def evaluation_ready(item):
    train = Path(item["training"]["root"])
    output = Path(item["training"]["output"])
    step = int(item["posttrain_steps"])
    complete, exit_path = output / "TRAINING_COMPLETE.json", train / "nodes/node0/exit.json"
    if not complete.exists() or not exit_path.exists():
        return False
    value, exited = json.loads(complete.read_text()), json.loads(exit_path.read_text())
    if value["status"] != "complete" or value["steps"] != step or exited["exit_code"] != 0:
        raise ValueError("LIBERO posttraining did not complete successfully")
    return step in completed_steps(output / "summary.jsonl")


def bundle_evaluation(item):
    root = Path(item["evaluation"]["root"])
    model = root / "model"
    destination = model / "final_model/pytorch_model.pt"
    if (model / "provenance.json").exists():
        value = json.loads((model / "provenance.json").read_text())
        if value.get("architecture", "mv2-register4") != item.get("architecture", "mv2-register4"):
            raise ValueError("Evaluation snapshot architecture differs from plan")
        if not destination.is_file() or destination.stat().st_size != value["bytes"]:
            raise ValueError("Evaluation model snapshot is missing or changed")
        return True
    if not evaluation_ready(item):
        return False
    output = Path(item["training"]["output"])
    source = output / f"checkpoints/steps_{item['posttrain_steps']}_pytorch_model.pt"
    value = pin_weights(source, destination, dict(posttrain_steps=item["posttrain_steps"],
                        architecture=item.get("architecture", "mv2-register4"),
                        droid_source_step=item["source_step"], action_interface="anchored_goal_binary"))
    for name in ("config.yaml", "config.full.yaml"):
        shutil.copy2(output / "config.full.yaml", model / name)
    shutil.copy2(output / "dataset_statistics.json", model / "dataset_statistics.json")
    spec_path = root / "configs/evaluation.json"
    spec = json.loads(spec_path.read_text())
    spec["checkpoint_bytes"] = value["bytes"]
    publish(spec_path, spec)
    # Commit only after configuration + matched LIBERO statistics are ready.
    publish(model / "provenance.json", value)
    return True


def status_of(stage):
    root = Path(stage["root"])
    receipt = root / "platform/submission.json"
    return json.loads(receipt.read_text()) if receipt.exists() else None


def tick(plan):
    records, errors = [], []
    for item in plan["items"]:
        state = dict(label=item["label"], source_step=item["source_step"], phase="waiting_droid_checkpoint")
        if item.get("cancelled", False):
            state.update(phase="cancelled", training=status_of(item["training"]),
                         evaluation=status_of(item["evaluation"]),
                         result=dict(item.get("cancellation", {}), status="cancelled"))
            records.append(state)
            continue
        try:
            source = pin_source(plan, item)
            if source is not None:
                state.update(source=source, training=submit(item["training"]), phase="libero_posttraining_submitted")
                if bundle_evaluation(item):
                    state.update(evaluation=submit(item["evaluation"]), phase="libero_plus_submitted")
                    evaluation_root = Path(item["evaluation"]["root"])
                    if (evaluation_root / "FAILED.json").exists():
                        raise RuntimeError((evaluation_root / "FAILED.json").read_text())
                    complete = evaluation_root / "EVALUATION_COMPLETE.json"
                    if complete.exists():
                        value = json.loads(complete.read_text())
                        if value["status"] != "complete" or value["completed"] != 10030:
                            raise ValueError("Incomplete Plus evaluation")
                        state.update(phase="complete", result=value)
                training_root = Path(item["training"]["root"])
                exit_path = training_root / "nodes/node0/exit.json"
                if exit_path.exists() and json.loads(exit_path.read_text())["exit_code"] != 0:
                    raise RuntimeError(f"Posttraining failed: {training_root}")
        except Exception as error:
            error_value = dict(label=item["label"], type=type(error).__name__, message=str(error)[:1200])
            errors.append(error_value)
            state["error"] = error_value
        records.append(state)
    return dict(time=time.time(), pid=os.getpid(), branches=records, errors=errors,
                complete=all(item["phase"] in ("complete", "cancelled") for item in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan["root"])
    lock = (root / "platform/orchestrator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish(root / "platform/orchestrator-process.json", dict(pid=os.getpid(), started=time.time(), plan=str(plan_path)))
    while True:
        state = tick(plan)
        publish(root / "STATUS.json", state)
        with (root / "platform/history.jsonl").open("a") as handle:
            handle.write(json.dumps(state, ensure_ascii=False) + "\n")
        if state["complete"]:
            results = {item["label"]: item["result"] for item in state["branches"]}
            publish(root / "PIPELINE_COMPLETE.json", dict(status="complete", results=results))
            return
        if args.once:
            print(json.dumps(state, ensure_ascii=False))
            return
        time.sleep(int(plan.get("poll_seconds", 30)))


if __name__ == "__main__":
    main()
