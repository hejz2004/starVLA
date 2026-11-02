"""External normalization for the UMI LeRobot dataset (v2).

Uniform per-dimension q01/q99 min-max normalization to [-1, 1] over the
whole state/action vector (translation, quaternion and gripper channels
alike; arm block layout [pos(3), quat(4), grip(1)] per arm). Quaternion
inputs are hemisphere-canonicalized upstream, so per-dim affine
normalization is continuous in the data domain.

Statistics are computed offline by
examples/umi/tools/compute_umi_lerobot_statistics.py (schema
umi-lerobot-statistics-v2: per-dim q01/q99 plus mean/std diagnostics) and
loaded here. The LIBERO UMI16 adapter uses the same q01/q99 format with
a separate dataset-specific statistics file. Fitting at training startup is
deliberately not supported.

The inverse lives in UMINormalizer.denormalize_* so train and inference
share one implementation. The math is exactly the [-1,1] min-max rescale
of FrameworkTools.unnormalize_actions, except its default gripper
binarization (index 6, a rotation dim here — UMI grippers are continuous
at dim % 8 == 7); UMI eval harnesses must not binarize.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "umi-lerobot-statistics-v2"
LIBERO_SCHEMA = "libero-umi16-statistics-v1"
MIN_SPAN = 1e-6


def load_statistics(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") not in (SCHEMA, LIBERO_SCHEMA):
        raise ValueError(f"statistics file {path} must use schema {SCHEMA} or {LIBERO_SCHEMA}")
    for key in ("state", "action"):
        block = payload.get(key)
        if not isinstance(block, dict):
            raise ValueError(f"statistics[{key}] block missing")
        for field in ("q01", "q99", "mean", "std"):
            if len(block.get(field, [])) == 0:
                raise ValueError(f"statistics[{key}].{field} missing/empty")
        if not (len(block["q01"]) == len(block["q99"]) == len(block["mean"]) == len(block["std"])):
            raise ValueError(f"statistics[{key}] per-dim length mismatch")
    return payload


class UMINormalizer:
    """Per-dim q01/q99 min-max normalization to [-1, 1] on the last axis."""

    def __init__(self, statistics_path):
        self.payload = load_statistics(statistics_path)
        self._params = {}
        for key in ("state", "action"):
            block = self.payload[key]
            q01 = np.asarray(block["q01"], dtype=np.float64)
            q99 = np.asarray(block["q99"], dtype=np.float64)
            span = q99 - q01
            if not np.isfinite(q01).all() or not np.isfinite(q99).all():
                raise ValueError(f"statistics[{key}] q01/q99 contain NaN/Inf")
            bad = np.flatnonzero(span < MIN_SPAN)
            if bad.size:
                raise ValueError(f"statistics[{key}] degenerate q01/q99 span on dims {bad.tolist()}")
            self._params[key] = {"q01": q01.astype(np.float32), "span": span.astype(np.float32)}

    def _forward(self, x: np.ndarray, key: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        q01, span = self._params[key]["q01"], self._params[key]["span"]
        if x.shape[-1] != len(q01):
            raise ValueError(f"{key} last dim {x.shape[-1]} != statistics dim {len(q01)}")
        return np.clip(2.0 * (x - q01) / span - 1.0, -1.0, 1.0)

    def _inverse(self, n: np.ndarray, key: str) -> np.ndarray:
        n = np.asarray(n, dtype=np.float32)
        q01, span = self._params[key]["q01"], self._params[key]["span"]
        if n.shape[-1] != len(q01):
            raise ValueError(f"{key} last dim {n.shape[-1]} != statistics dim {len(q01)}")
        return (np.clip(n, -1.0, 1.0) + 1.0) * 0.5 * span + q01

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self._forward(state, "state")

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return self._forward(action, "action")

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return self._inverse(state, "state")

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self._inverse(action, "action")


class UMINormalizedDataset:
    """Wrap UMILerobotDataset and normalize state/action in place."""

    def __init__(self, dataset, normalizer: UMINormalizer):
        if dataset.action_dim != len(normalizer._params["action"]["q01"]):
            raise ValueError(
                f"action_dim {dataset.action_dim} != statistics dim {len(normalizer._params['action']['q01'])}"
            )
        if dataset.state_dim != len(normalizer._params["state"]["q01"]):
            raise ValueError(
                f"state_dim {dataset.state_dim} != statistics dim {len(normalizer._params['state']['q01'])}"
            )
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["state"] = self.normalizer.normalize_state(sample["state"])
        sample["action"] = self.normalizer.normalize_action(sample["action"])
        return sample

    def save_dataset_statistics(self, output_path):
        """Write run-dir statistics in the starVLA norm_stats schema.

        Top-level key is the unnorm_key. All dims share the [-1,1] min-max
        math of FrameworkTools.unnormalize_actions (its mask defaults to
        all-True); only its default gripper binarization (index 6) is
        inapplicable to UMI vectors.
        """
        from pathlib import Path as _P

        output_path = _P(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        provenance = self.dataset.save_dataset_statistics(output_path.with_suffix(".provenance.json"))
        stats = {
            key: {
                "q01": [float(v) for v in self.normalizer.payload[key]["q01"]],
                "q99": [float(v) for v in self.normalizer.payload[key]["q99"]],
            }
            for key in ("state", "action")
        }
        payload = {"umi_lerobot": stats | {"meta": provenance}}
        output_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"[umi_lerobot] norm_stats written to {output_path}")
        return payload

    def __getattr__(self, name):
        if name.startswith("__") or name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)
