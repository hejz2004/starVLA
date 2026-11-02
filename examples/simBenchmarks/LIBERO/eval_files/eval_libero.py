import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.simBenchmarks.LIBERO.eval_files.action_goal_binary import controller_reference
from starVLA.dataloader.libero_action_contract import REPRESENTATION as GOAL_BINARY_REP
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import (
    MAX_STEPS, VIDEO_LAYOUT, atomic_json, inference_seed, select_jobs, suite_directory, valid_result,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    if arr.shape != (1,) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid binary open prediction: {arr}")
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _resolve_gripper_command(raw_action: dict) -> np.ndarray:
    """Resolve binary open predictions or already decoded native -1/+1 commands."""
    if "gripper_command" not in raw_action:
        return _binarize_gripper_open(raw_action["open_gripper"])
    command = np.asarray(raw_action["gripper_command"], dtype=np.float32).reshape(-1)
    if command.shape != (1,) or not np.isfinite(command).all() or not np.isin(command, [-1, 1]).all():
        raise ValueError(f"Invalid direct gripper command: {command}")
    return command


def _save_rollout_video(video_path, frames):
    """Atomically save frames already stitched main-left, wrist-right."""
    if not frames:
        raise ValueError("Rollout video needs at least one frame")
    video_path = pathlib.Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = video_path.with_name(video_path.stem + ".tmp.mp4")
    try:
        imageio.mimwrite(temporary, frames, fps=10)
        temporary.replace(video_path)
    finally:
        temporary.unlink(missing_ok=True)
    return video_path


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Root containing <Suite>/videos/

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True
    action_interface: str = "anchored_goal_binary"

    job_name: str = "test"

    # Optional sharding; empty filters preserve the original task/trial selection.
    task_ids: list[int] = dataclasses.field(default_factory=list)
    trial_ids: list[int] = dataclasses.field(default_factory=list)
    shard_index: int = 0
    num_shards: int = 1
    deterministic_inference: bool = False
    results_dir: str = ""
    protocol_digest: str = ""
    resume: bool = False
    save_video: bool = True


def eval_libero(args: Args) -> None:
    import mujoco
    import robosuite
    if mujoco.__version__ != '3.2.3' or robosuite.__version__ != '1.4.0':
        raise ValueError('LIBERO evaluation requires mujoco 3.2.3 and robosuite 1.4.0, matching action-data generation')
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))
    np.random.seed(args.seed)
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be nonnegative")
    if args.resume and (not args.results_dir or not args.protocol_digest):
        raise ValueError("Resume requires results_dir and a validated protocol_digest")
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    jobs = select_jobs(args.task_suite_name, task_suite.n_tasks, args.num_trials_per_task,
                       args.task_ids, args.trial_ids, args.max_tasks,
                       args.shard_index, args.num_shards)
    video_dir = suite_directory(args.video_out_path, args.task_suite_name) / "videos"
    output = suite_directory(args.results_dir, args.task_suite_name) if args.results_dir else None
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    if output:
        (output / "results").mkdir(parents=True, exist_ok=True)
        (output / "traces").mkdir(parents=True, exist_ok=True)
        for job in jobs:
            path = output / "results" / (job["key"] + ".json")
            if path.exists() and not args.resume:
                raise FileExistsError(f"Existing result; use --args.resume: {path}")
        jobs = [j for j in jobs if not (args.resume and valid_result(
            output / "results" / (j["key"] + ".json"), j, args.protocol_digest, args.save_video,
            required_video_layout=VIDEO_LAYOUT))]
    if not jobs:
        logging.info("No pending episodes for shard %d", args.shard_index)
        return
    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key,
                               action_interface=args.action_interface)
    env = None
    total_episodes = total_successes = 0
    try:
        metadata = client_model._server_metadata
        if args.deterministic_inference and not metadata.get("request_seeded", False):
            raise ValueError("Deterministic evaluation needs the parallel launcher's request-seeded server")
        if args.pretrained_path and metadata.get("ckpt_path"):
            if pathlib.Path(args.pretrained_path).resolve() != pathlib.Path(metadata["ckpt_path"]).resolve():
                raise ValueError("Connected server checkpoint differs from pretrained_path")
        if output:
            atomic_json(output / f"worker_metadata_{args.task_suite_name}_{args.shard_index}.json", metadata)
        for task_id in sorted({j["task_id"] for j in jobs}):
            selected = [j for j in jobs if j["task_id"] == task_id]
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if max(j["trial"] for j in selected) >= len(initial_states):
                raise ValueError(f"Task {task_id} has only {len(initial_states)} initial states")
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_successes = 0
            for job in selected:
                episode_idx = job["trial"]
                started = time.monotonic()
                client_model.reset(task_description=task_description)
                if args.deterministic_inference:
                    np.random.seed(args.seed)
                    env.seed(args.seed)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                initial_hash = hashlib.sha256(np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                replay_images, full_actions, traces = [], [], []
                done = False
                first_images_hash = first_chunk_hash = None
                for step in range(MAX_STEPS[args.task_suite_name]):
                    # Original camera observations and preprocessing are retained.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if args.save_video:
                        replay_images.append(np.concatenate((img, wrist_img), axis=1))
                    example = {"image": [img, wrist_img], "lang": str(task_description)}
                    if first_images_hash is None:
                        first_images_hash = hashlib.sha256(img.tobytes() + wrist_img.tobytes()).hexdigest()
                    width = float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
                    seed = inference_seed(args.seed, args.task_suite_name, task_id, episode_idx,
                                          step // client_model.action_chunk_size) if args.deterministic_inference else None
                    native_reference = controller_reference(env) if client_model.action_representation == GOAL_BINARY_REP else {}
                    response = client_model.step(example=example, step=step,
                        eef_pos=obs["robot0_eef_pos"], eef_quat=obs["robot0_eef_quat"],
                        diagnostic_seed=seed, **native_reference)
                    raw = response["raw_action"]
                    translation = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
                    rotation = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
                    if translation.size != 3 or rotation.size != 3:
                        raise ValueError("Invalid translation/rotation dimensions")
                    command = np.concatenate([translation, rotation, _resolve_gripper_command(raw)])
                    if command.shape != (7,) or not np.isfinite(command).all():
                        raise ValueError("Invalid LIBERO action")
                    if first_chunk_hash is None:
                        first_chunk_hash = hashlib.sha256(np.asarray(client_model.raw_actions).tobytes()).hexdigest()
                    predicted_gripper = client_model.raw_actions[step % client_model.action_chunk_size,
                        7 if client_model.action_representation == GOAL_BINARY_REP else 6]
                    traces.append(np.r_[step, width, predicted_gripper, command, obs["robot0_eef_pos"]])
                    full_actions.append(command)
                    obs, _, done, _ = env.step(command.tolist())
                    if done:
                        break
                video = None
                if args.save_video:
                    suffix = "success" if done else "failure"
                    filename = f"rollout_{task_description.replace(' ', '_')}_episode{episode_idx}_{suffix}.mp4"
                    video_path = video_dir / filename
                    video_path = _save_rollout_video(video_path, replay_images)
                    video = os.path.relpath(video_path, output) if output else str(video_path)
                if output:
                    trace_path = output / "traces" / (job["key"] + ".npz")
                    temporary = trace_path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, trace=np.asarray(traces), actions=np.asarray(full_actions))
                    temporary.replace(trace_path)
                    # Completion record is committed only after artifacts are saved.
                    record = dict(job, protocol_digest=args.protocol_digest, success=bool(done),
                        steps=len(full_actions), elapsed_seconds=time.monotonic() - started,
                        gripper_encoding="binary_open_code", action_representation=client_model.action_representation,
                        predicted_gripper_semantics="binary_open_code",
                        shard_index=args.shard_index, port=args.port,
                        video=video, video_layout=VIDEO_LAYOUT if args.save_video else None,
                        video_fps=10 if args.save_video else None,
                        video_frames=len(replay_images) if args.save_video else 0,
                        trace=str(trace_path.relative_to(output)),
                        initial_state_sha256=initial_hash, first_images_sha256=first_images_hash,
                        first_chunk_sha256=first_chunk_hash)
                    atomic_json(output / "results" / (job["key"] + ".json"), record)
                total_episodes += 1
                total_successes += int(done)
                task_successes += int(done)
                logging.info("EPISODE %s success=%s steps=%d", job["key"], done, len(full_actions))
                logging.info("Success: %s", done)
            logging.info("Current task success rate: %s", task_successes / len(selected))
            logging.info("Current total success rate: %s", total_successes / total_episodes)
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()
        client_model.client.close()
    logging.info("Total success rate: %s", total_successes / total_episodes)
    logging.info("Total episodes: %s", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
