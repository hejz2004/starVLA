"""Future-event orchestration: save boundaries, pinning, and ambiguous creates."""
import json
from pathlib import Path
import subprocess

import pytest

from examples.droid.tools import transfer_pipeline as pipeline


def inspect_small(path):
    return dict(bytes=Path(path).stat().st_size, tensors=1)


def test_completion_requires_newline(tmp_path):
    summary = tmp_path / "summary.jsonl"
    summary.write_text('{"steps":1000}\n{"steps":2000}')
    assert pipeline.completed_steps(summary) == {1000}
    with summary.open("a") as handle:
        handle.write("\n")
    assert pipeline.completed_steps(summary) == {1000, 2000}


def test_hardlink_survives_trainer_pruning(tmp_path):
    source, destination = tmp_path / "source.pt", tmp_path / "snapshot/model.pt"
    source.write_bytes(b"trained weights")
    record = pipeline.pin_weights(source, destination, {"source_step": 2000}, inspect=inspect_small)
    assert source.samefile(destination)
    source.unlink()
    assert destination.read_bytes() == b"trained weights"
    assert pipeline.digest_file(destination) == record["sha256"]


def source_fixture(tmp_path):
    pretrain = tmp_path / "pretrain"
    output = pretrain / "output"
    (output / "checkpoints").mkdir(parents=True)
    pipeline.publish(pretrain / "ACTIVE_ATTEMPT.json", {"output_dir": str(output)})
    for name in ("config.full.yaml", "dataset_statistics.json", "dataset_access.json"):
        (output / name).write_text("{}")
    source = output / "checkpoints/steps_2000_pytorch_model.pt"
    source.write_bytes(b"complete weights")
    plan = {"pretraining_root": str(pretrain), "pretraining_output": str(output)}
    item = {"source_step": 2000, "source_weights": str(tmp_path / "pinned/model.pt")}
    return plan, item, output, source


def inject_small_inspector(monkeypatch):
    original = pipeline.pin_weights
    monkeypatch.setattr(pipeline, "pin_weights", lambda src, dst, meta: original(src, dst, meta, inspect=inspect_small))


def test_pin_waits_for_save_then_keeps_snapshot(tmp_path, monkeypatch):
    plan, item, output, source = source_fixture(tmp_path)
    inject_small_inspector(monkeypatch)
    assert pipeline.pin_source(plan, item) is None
    (output / "summary.jsonl").write_text('{"steps":2000}\n')
    value = pipeline.pin_source(plan, item)
    source.unlink()
    assert pipeline.pin_source(plan, item) == value
    assert Path(item["source_weights"]).with_name("READY.json").is_file()


def test_recover_link_committed_before_ready_after_source_pruned(tmp_path, monkeypatch):
    plan, item, output, source = source_fixture(tmp_path)
    inject_small_inspector(monkeypatch)
    destination = Path(item["source_weights"])
    pipeline.pin_weights(source, destination, {"source_step": 2000})
    source.unlink()
    # No source save marker remains. The already pinned inode is sufficient.
    assert pipeline.pin_source(plan, item)["source_step"] == 2000
    assert destination.with_name("READY.json").exists()


def test_ambiguous_create_reconciles_without_duplicate_job(tmp_path):
    root = tmp_path / "training"
    (root / "platform").mkdir(parents=True)
    command = ["kmctl", "--context", "hjz", "create", "training-job", "unique-job",
               "--node-group", "pro6000-research"]
    pipeline.publish(root / "platform/submit-command.json", command)
    created = []

    def execute(args, **kwargs):
        if args == command:
            created.append(args)
            raise subprocess.TimeoutExpired(args, 45)
        if "training-jobs" in args:
            body = [] if not created else [{"Meta": {"Name": "unique-job", "Id": "tj-existing"}}]
        else:
            body = []
        return subprocess.CompletedProcess(args, 0, json.dumps(body), "")

    with pytest.raises(subprocess.TimeoutExpired):
        pipeline.submit({"root": str(root)}, execute=execute)
    assert pipeline.submit({"root": str(root)}, execute=execute)["Id"] == "tj-existing"
    assert pipeline.submit({"root": str(root)}, execute=lambda *a, **kw: pytest.fail("Already submitted"))["Id"] == "tj-existing"
    assert len(created) == 1


def test_waiting_sources_do_not_reserve_gpus(monkeypatch):
    monkeypatch.setattr(pipeline, "pin_source", lambda plan, item: None)
    monkeypatch.setattr(pipeline, "submit", lambda *args: pytest.fail("Premature GPU submission"))
    state = pipeline.tick({"items": [dict(label=label, source_step=step) for label, step in
                                     (("2k", 2000), ("4k", 4000), ("final", 6044))]})
    assert not state["complete"] and not state["errors"]
    assert all(item["phase"] == "waiting_droid_checkpoint" for item in state["branches"])


def test_evaluation_requires_successful_posttraining_completion(tmp_path):
    train, output = tmp_path / "train", tmp_path / "output"
    (train / "nodes/node0").mkdir(parents=True)
    output.mkdir()
    item = {"training": {"root": str(train), "output": str(output)}, "posttrain_steps": 7500}
    (output / "summary.jsonl").write_text('{"steps":7500}\n')
    assert not pipeline.evaluation_ready(item)
    pipeline.publish(output / "TRAINING_COMPLETE.json", {"status": "complete", "steps": 7500})
    pipeline.publish(train / "nodes/node0/exit.json", {"exit_code": 1})
    with pytest.raises(ValueError, match="successfully"):
        pipeline.evaluation_ready(item)
    pipeline.publish(train / "nodes/node0/exit.json", {"exit_code": 0})
    assert pipeline.evaluation_ready(item)


@pytest.mark.parametrize("has_registers,architecture,accepted", [
    (False, "native", True), (True, "native", False),
    (True, "mv2-register4", True), (False, "mv2-register4", False),
])
def test_checkpoint_architectures_cannot_be_mixed(has_registers, architecture, accepted):
    import torch
    state = {
        "qwen_vl_interface.model.model.visual.blocks.23.attn.qkv.weight": torch.empty(3, 1),
        "qwen_vl_interface.model.model.language_model.layers.27.mlp.down_proj.weight": torch.empty(1, 1),
        "action_model.action_decoder.weight": torch.empty(16, 1),
    }
    if has_registers:
        state["qwen_vl_interface.model.model.visual.register_tokens.weight"] = torch.empty(4, 1024)
    if accepted:
        details = pipeline.state_schema_metadata(state, architecture)
        assert details["architecture"] == architecture
        assert details["register_shape"] == ([4, 1024] if has_registers else None)
    else:
        with pytest.raises(ValueError, match="register tokens"):
            pipeline.state_schema_metadata(state, architecture)


def test_native_source_pin_retains_architecture_and_rejects_other_plan(tmp_path, monkeypatch):
    plan, item, output, source = source_fixture(tmp_path)
    plan["architecture"] = "native"
    inject_small_inspector(monkeypatch)
    (output / "summary.jsonl").write_text('{"steps":2000}\n')
    assert pipeline.pin_source(plan, item)["architecture"] == "native"
    plan["architecture"] = "mv2-register4"
    with pytest.raises(ValueError, match="architecture differs"):
        pipeline.pin_source(plan, item)


def test_cancelled_branch_never_pins_submits_or_evaluates(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "pin_source", lambda *args: pytest.fail("Cancelled source inspected"))
    monkeypatch.setattr(pipeline, "submit", lambda *args: pytest.fail("Cancelled job submitted"))
    monkeypatch.setattr(pipeline, "bundle_evaluation", lambda *args: pytest.fail("Cancelled job evaluated"))
    item = dict(label="final", source_step=6044, cancelled=True,
                training={"root": str(tmp_path / "training")},
                evaluation={"root": str(tmp_path / "evaluation")},
                cancellation={"reason": "User replaced with a new recipe"})
    state = pipeline.tick({"items": [item]})
    assert state["complete"] and not state["errors"]
    assert state["branches"][0]["phase"] == "cancelled"
    assert state["branches"][0]["result"]["reason"] == item["cancellation"]["reason"]
