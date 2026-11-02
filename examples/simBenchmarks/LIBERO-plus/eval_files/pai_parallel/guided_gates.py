#!/usr/bin/env python3
"""Real, separate-process rendering and accelerator checks for guided Plus eval.

No model is loaded here. The CPU command calls the adapted worker's unchanged
environment helper and the official noise functions. The PPU command must run
in the original PPU interpreter, never the independent CPU simulation runtime.
The caller must also require the command's actual exit code to be zero.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def official_noise_dispatch(source):
    """Read the real official ControlEnv.step branch ranges, without executing it."""
    tree = ast.parse(source)
    environment = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ControlEnv")
    step = next(n for n in environment.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    global_functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    found = []
    for branch in ast.walk(step):
        if not isinstance(branch, ast.If):
            continue
        test = branch.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.LtE)
                and isinstance(test.left, ast.Attribute) and test.left.attr == "noise"
                and isinstance(test.left.value, ast.Name) and test.left.value.id == "self"
                and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                and type(test.comparators[0].value) is int):
            continue
        calls = {node.func.id for statement in branch.body for node in ast.walk(statement)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in global_functions}
        require(len(calls) == 1, "Official noise branch no longer has one clear function")
        found.append((test.comparators[0].value, calls.pop(), branch.lineno))
    found.sort()
    require(len(found) == 5 and [bound for bound, _, _ in found] == [10, 20, 30, 40, 50],
            "Official noise dispatch changed; refuse to guess its semantics")
    require(len({name for _, name, _ in found}) == 5, "Official noise functions are not unique")
    dispatch, previous = [], 0
    for upper, function, line in found:
        dispatch.append({"function": function, "noise_min": previous + 1, "noise_max": upper,
                         "severity_offset": previous, "official_source_line": line})
        previous = upper
    return dispatch


def choose_noise_variants(classification, dispatch):
    chosen = []
    for family in dispatch:
        selected = None
        for suite in SUITES:
            for item in sorted(classification[suite], key=lambda row: row["id"]):
                if item["category"] != "Sensor Noise":
                    continue
                match = re.search(r"_noise_(\d+)$", item["name"])
                require(match is not None, "Official Sensor Noise variant has no noise suffix")
                code = int(match.group(1))
                if family["noise_min"] <= code <= family["noise_max"]:
                    selected = {"suite": suite, "task_id": item["id"] - 1,
                                "variant_id": item["id"], "variant": item["name"],
                                "category": item["category"], "noise_code": code,
                                "noise_function": family["function"],
                                "severity": code - family["severity_offset"],
                                "selection": "first actual classified variant in the official family range"}
                    break
            if selected is not None:
                break
        require(selected is not None, f"No real official variant for {family['function']}")
        chosen.append(selected)
    require(len({(c["suite"], c["task_id"]) for c in chosen}) == 5, "Noise test variants overlap")
    return chosen


def import_worker(worker_path, libero_root):
    worker_path = Path(worker_path).resolve()
    require(worker_path == (HERE / "launchers" / "eval_worker.py").resolve(),
            "Render gate must use this frozen adapted worker")
    source = Path(os.environ.get("STARVLA_DIR", ""))
    require(source.is_absolute() and source.is_dir(), "STARVLA_DIR must point to the matching frozen HJZ source")
    for target in reversed((worker_path.parent, source, libero_root)):
        sys.path.insert(0, str(target))
    spec = importlib.util.spec_from_file_location("guided_frozen_eval_worker", worker_path)
    require(spec is not None and spec.loader is not None, "Cannot import the actual adapted worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    client_file = Path(inspect.getfile(module.ModelClient)).resolve()
    require(client_file.is_relative_to(source.resolve()), "Simulation ModelClient came from an unrelated source")
    require(module.LIBERO_ENV_RESOLUTION == 256 and module.LIBERO_DUMMY_ACTION == [0.0] * 6 + [-1.0],
            "Adapted worker render/stabilization constants changed")
    return module, {"worker_path": str(worker_path), "worker_sha256": sha256(worker_path),
                    "model_client_path": str(client_file), "model_client_sha256": sha256(client_file)}


def cpu_render(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for the independent CPU render process")
    require(os.environ.get("MUJOCO_GL") == "osmesa"
            and os.environ.get("PYOPENGL_PLATFORM") == "osmesa", "CPU rendering must use OSMesa")
    libero_root, config_dir = args.libero_root.resolve(), args.config_dir.resolve()
    require(libero_root.is_dir() and config_dir.is_dir(), "Private source/config directories do not exist")
    os.environ["LIBERO_HOME"] = str(libero_root)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    require(sha256(args.classification) == CLASSIFICATION_SHA256, "Pinned reference classification SHA differs")
    live_classification = libero_root / "libero/libero/benchmark/task_classification.json"
    require(sha256(live_classification) == CLASSIFICATION_SHA256, "Live official classification SHA differs")
    import numpy as np
    import torch
    require(not torch.cuda.is_available(), "Simulation process must hide its accelerators")
    worker, source_proof = import_worker(args.worker_path, libero_root)
    official_wrapper = importlib.import_module("libero.libero.envs.env_wrapper")
    wrapper_file = Path(inspect.getfile(official_wrapper)).resolve()
    require(wrapper_file == (libero_root / "libero/libero/envs/env_wrapper.py").resolve(),
            "Official environment imported from another repository")
    wrapper_source = wrapper_file.read_text(encoding="utf-8")
    dispatch = official_noise_dispatch(wrapper_source)
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    # Reuse the independent auditor to validate exact four-suite and seven-category inventory.
    from audit_results import load_classification
    load_classification(args.classification)
    selected = []
    for suite in SUITES:
        item = next(row for row in classification[suite] if row["id"] == 1)
        selected.append({"suite": suite, "task_id": 0, "variant_id": 1,
                         "variant": item["name"], "category": item["category"], "noise_code": 0,
                         "noise_function": None, "selection": "first actual task of each of four suites"})
    selected.extend(choose_noise_variants(classification, dispatch))
    frames_dir = args.output.parent / (args.output.stem + "_frames")
    require(not frames_dir.exists(), "Render gate frame output already exists; refusing replay")
    frames_dir.mkdir(parents=True)
    from PIL import Image
    benchmarks = worker.benchmark.get_benchmark_dict()
    suites = {name: benchmarks[name]() for name in SUITES}
    tracked_codes = {getattr(official_wrapper, item["function"]).__code__: item["function"] for item in dispatch}
    actual_renders = []
    np.random.seed(7)  # A fresh gate process; formal workers start their own independent seed7 processes.
    for case_index, case in enumerate(selected):
        task_suite = suites[case["suite"]]
        task = task_suite.get_task(case["task_id"])
        require(task.name == case["variant"], "Live benchmark task ordering differs from official classification")
        initial_states = task_suite.get_task_init_states(case["task_id"])
        require(len(initial_states) >= 1, "Actual task has no first initial state")
        noise_calls = Counter()
        previous_profile = sys.getprofile()
        def profile(frame, event, arg):
            if event == "call" and frame.f_code in tracked_codes:
                noise_calls[tracked_codes[frame.f_code]] += 1
        env = None
        case_start = time.monotonic()
        try:
            sys.setprofile(profile)
            env, description = worker._get_libero_env(task, worker.LIBERO_ENV_RESOLUTION, 7)
            require(env.noise == case["noise_code"], "The actual official environment did not enable the selected noise")
            env.reset()
            obs = env.set_init_state(initial_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(worker.LIBERO_DUMMY_ACTION)
            if case["noise_function"]:
                require(noise_calls[case["noise_function"]] >= 10,
                        f"Real official noise function did not execute on ten stabilization steps: {case['noise_function']}")
            else:
                require(not noise_calls, "First-suite variant unexpectedly applied a sensor-noise function")
            views = {}
            for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                pixels = np.asarray(obs[camera])
                require(pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8,
                        f"Actual {camera} has wrong shape/dtype: {pixels.shape}/{pixels.dtype}")
                standard_deviation = float(pixels.std())
                require(np.isfinite(pixels).all() and standard_deviation > 0,
                        f"Actual {camera} is non-finite or blank")
                filename = f"case{case_index:02d}_{case['suite']}_task{case['task_id']:05d}_{camera}.png"
                target = frames_dir / filename
                Image.fromarray(pixels).save(target, format="PNG")
                views[camera] = {"shape": list(pixels.shape), "dtype": str(pixels.dtype),
                                 "std": standard_deviation, "min": int(pixels.min()), "max": int(pixels.max()),
                                 "raw_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 "png_path": str(target), "png_sha256": sha256(target)}
            record = dict(case, initial_state_index=0, wait_steps=10, environment_seed=7,
                          task_description=description, task_bddl_file=task.bddl_file,
                          task_init_states_file=task.init_states_file, actual_noise_calls=dict(noise_calls),
                          elapsed_seconds=time.monotonic() - case_start, views=views,
                          sensor_noise_scope="official wrapper applies sensor noise to agentview only")
        finally:
            sys.setprofile(previous_profile)
            if env is not None:
                env.close()  # Any close/render/deserialization error fails the subprocess.
        record["environment_closed"] = True
        actual_renders.append(record)
        print("CPU_RENDER_CASE_PASSED " + json.dumps({"suite": case["suite"], "task_id": case["task_id"],
                                                       "noise_function": case["noise_function"]}), flush=True)
    require(len(actual_renders) == 9, "Real CPU render coverage differs from four suite tasks plus five noise variants")
    return {
        "status": "passed", "command": "cpu-render", "function_completed": True,
        "process_exit_requirement": "Caller must independently require subprocess exit0, including no finalizer SIGSEGV",
        "python": sys.version, "torch_version": torch.__version__, "torch_file": torch.__file__,
        "renderer": "osmesa", "render_resolution": 256, "worker_numpy_seed": 7,
        "numpy_seed_calls": 1, "environment_seed": 7, "initial_state_index": 0, "wait_steps": 10,
        "classification_sha256": CLASSIFICATION_SHA256,
        "official_environment": {"path": str(wrapper_file), "sha256": sha256(wrapper_file),
                                 "source_modified_by_gate": False, "functions_monkeypatched": False},
        "noise_dispatch_read_from_official_source": dispatch,
        "noise_execution_proof": "Python call profiler matches the actual original function code objects",
        "actual_renders": actual_renders, "dual_view_render_count": len(actual_renders) * 2,
        "worker_source": source_proof, "policy_model_loaded": False,
        "scope": "Actual resource/init-state/OSMesa/noise rendering; this is not a policy rollout or success-rate evaluation",
    }


def ppu_compute(args):
    require(1 <= args.expected_devices <= 16, "Expected between 1 and 16 PPU devices; formal bootstrap requires 16")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") != "", "PPU compute process must keep its allocated accelerators visible")
    import torch
    require(torch.cuda.is_available(), "PPU runtime cannot see an accelerator")
    require(torch.cuda.device_count() == args.expected_devices,
            f"Expected {args.expected_devices} actual accelerators, saw {torch.cuda.device_count()}")
    devices = []
    for index in range(args.expected_devices):
        device = torch.device("cuda", index)
        started = time.monotonic()
        with torch.cuda.device(index), torch.no_grad():
            left = torch.ones((32, 32), device=device, dtype=torch.bfloat16)
            right = torch.ones((32, 32), device=device, dtype=torch.bfloat16)
            product = left @ right
            torch.cuda.synchronize(index)
            require(product.device.type == "cuda" and product.device.index == index,
                    "Matmul did not execute on the assigned accelerator")
            require(product.dtype == torch.bfloat16 and tuple(product.shape) == (32, 32),
                    "Actual accelerator matmul returned wrong precision/shape")
            require(bool(torch.isfinite(product).all().item()), "Actual bf16 matrix result is non-finite")
            maximum_error = float((product.float() - 32.0).abs().max().item())
            require(maximum_error == 0.0, "Actual bf16 ones-matmul result differs from the exact CPU reference")
            torch.cuda.synchronize(index)
            devices.append({"device_index": index, "device_name": torch.cuda.get_device_name(index),
                            "dtype": str(product.dtype), "shape": list(product.shape), "finite": True,
                            "max_absolute_error_against_exact_reference": maximum_error,
                            "synchronize_completed": True, "elapsed_seconds": time.monotonic() - started})
            del left, right, product
        print("PPU_BF16_DEVICE_PASSED " + json.dumps(devices[-1]), flush=True)
    return {
        "status": "passed", "command": "ppu-compute", "function_completed": True,
        "process_exit_requirement": "Caller must independently require subprocess exit0",
        "python": sys.version, "torch_version": torch.__version__, "torch_file": torch.__file__,
        "expected_devices": args.expected_devices, "actual_devices": len(devices), "devices": devices,
        "policy_model_loaded": False, "explicit_rng_seed_calls": 0,
        "random_operations_used": False,
        "scope": "Fresh isolated process executes actual bf16 matmul and synchronization on all16 accelerators; no model or policy RNG state is reused",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-render")
    cpu.add_argument("--libero-root", required=True, type=Path)
    cpu.add_argument("--config-dir", required=True, type=Path)
    cpu.add_argument("--classification", required=True, type=Path)
    cpu.add_argument("--worker-path", type=Path, default=HERE / "launchers" / "eval_worker.py")
    cpu.add_argument("--output", required=True, type=Path)
    ppu = commands.add_parser("ppu-compute")
    ppu.add_argument("--expected-devices", type=int, default=16)
    ppu.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute(), "Gate output must be an absolute private run path")
    require(not args.output.exists(), "Gate output already exists; refusing to replace prior evidence")
    try:
        result = cpu_render(args) if args.command == "cpu-render" else ppu_compute(args)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output, result)
        print("GUIDED_GATE_FUNCTION_PASSED " + json.dumps({"command": args.command, "output": str(args.output)}), flush=True)
    except BaseException as error:
        atomic_json(args.output.with_name(args.output.stem + ".FAILED.json"), {
            "status": "failed", "command": args.command, "error_type": type(error).__name__, "error": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
