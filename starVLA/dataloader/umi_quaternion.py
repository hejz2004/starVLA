"""Versioned episode-wise quaternion signs; never rotate or normalize poses."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

POLICY = 'episode_continuous_xyzw_v1'
QUATERNION_DIMS = np.array([3, 4, 5, 6, 11, 12, 13, 14])
POLICY_DETAILS = dict(
    version=POLICY, initial_sign='first nonzero component in w,x,y,z is positive',
    continuation='flip when dot(previous signed quaternion, current raw quaternion) < 0',
    zero_dot='choose canonical sign; either sign is equally near',
    reset='episode start, invalid quaternion, or nonconsecutive frame_index',
    invalid='leave unchanged; existing valid-window policy decides eligibility',
    magnitude='unchanged', ordering='xyzw', unit_projection=False,
)


def continuity_report(values):
    """Check physical xyzw values before normalization; never alter a pose.

    Negative adjacent unit dot products identify an antipodal discontinuity.
    This is not a physical angular-velocity or tracking-quality test.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16 or len(values) < 2:
        raise ValueError("Quaternion continuity requires [N>=2,16] physical values")
    q = values[:, QUATERNION_DIMS].reshape(len(values), 2, 4)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms <= 1e-8):
        raise ValueError("Invalid physical quaternion in continuity check")
    unit = q / norms
    dots = np.sum(unit[1:] * unit[:-1], axis=-1)
    return dict(adjacent_pairs=int(dots.size), negative_pairs=int(np.sum(dots < -1e-7)),
                minimum_unit_dot=float(dots.min()), tolerance=1e-7)


def continuous_signs(quaternions, frames=None):
    """Return int8 +/-1 per row. Work on a full episode, independently per hand.

    The vectorized parity calculation equals sequential hemisphere unwrapping.
    Invalid rows break continuity and are not repaired. A zero dot restarts the
    deterministic canonical gauge, since neither antipode is nearer.
    """
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError('Expected episode quaternions [N,4] in xyzw order')
    n = len(q)
    if not n:
        return np.empty(0, dtype=np.int8)
    valid = np.isfinite(q).all(axis=1) & np.any(q != 0, axis=1)
    # float32 source values cannot overflow FP64 dot products.
    dots = np.sum(q[1:] * q[:-1], axis=1)
    linked = valid[1:] & valid[:-1] & (dots != 0)
    if frames is not None:
        frames = np.asarray(frames)
        if frames.shape != (n,) or frames.dtype.kind not in 'iu':
            raise ValueError('Expected integer frame_index [N]')
        linked &= np.diff(frames) == 1
    starts = np.r_[True, ~linked]
    parity = np.bitwise_xor.accumulate(np.r_[0, linked & (dots < 0)].astype(np.uint8))
    start_index = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    ordered = q[:, [3, 0, 1, 2]]
    first = ordered[np.arange(n), np.argmax(ordered != 0, axis=1)]
    canonical_negative = first < 0
    negative = parity ^ parity[start_index] ^ canonical_negative[start_index].astype(np.uint8)
    return np.where(valid & negative.astype(bool), -1, 1).astype(np.int8)


def transformed_meta(meta, manifest):
    result = deepcopy(meta)
    result['quaternion_transform'] = dict(policy=POLICY, manifest_fingerprint=manifest['fingerprint'])
    return result


class QuaternionSigns:
    """Small per-file bit maps, checked lazily and cached independently per worker."""
    def __init__(self, directory, meta, max_files=8, *, audit_assets=True):
        from .umi_normalization import content_fingerprint, parent_fingerprint
        if type(audit_assets) is not bool:
            raise ValueError("audit_assets must be a boolean")
        self.audit_assets = audit_assets
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        body = {k: v for k, v in m.items() if k != 'fingerprint'}
        if self.audit_assets and (m.get('version') != POLICY or m.get('policy') != POLICY_DETAILS
                or m.get('status') != 'completed' or m.get('fingerprint') != content_fingerprint(body)
                or m.get('parent_data_fingerprint') != parent_fingerprint(meta)
                or m.get('metadata_sha256') != meta['artifacts']['metadata.sqlite3']['sha256']):
            raise ValueError('Quaternion sign map is incomplete, changed, or for different data')
        self.max_files = max_files
        self._pid = None
        self._cache = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'], state['_pid'] = OrderedDict(), None
        return state

    def close(self):
        self._cache.clear()

    def _file(self, relative):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if relative in self._cache:
            value = self._cache.pop(relative)
            self._cache[relative] = value
            return value
        record = self.manifest['files'].get(relative)
        if record is None:
            raise ValueError(f'Missing quaternion signs for {relative}')
        path = (self.directory / record['path']).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Sign shard escapes its preparation directory')
        # Shards contain only two bits per physical row, not numeric/video data.
        data = path.read_bytes()
        if self.audit_assets and (len(data) != record['size_bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']):
            raise ValueError(f'Quaternion sign shard changed: {relative}')
        import io
        bits = np.load(io.BytesIO(data), allow_pickle=False)
        if self.audit_assets and (bits.dtype != np.uint8 or bits.shape != ((record['rows'] + 7) // 8, 2)):
            raise ValueError('Invalid quaternion sign shard shape/dtype')
        while len(self._cache) >= self.max_files:
            self._cache.popitem(last=False)
        self._cache[relative] = (bits, record['rows'])
        return self._cache[relative]

    def apply(self, values, locations):
        """Use physical row addresses from the original immutable access index."""
        result = np.asarray(values, dtype=np.float32).copy()
        if self.audit_assets and (result.ndim != 2 or result.shape[1] != 16):
            raise ValueError('Expected raw world-pose [N,16] values')
        cursor = 0
        for loc in locations:
            bits, nrows = self._file(loc['data_file'])
            a, b = loc['file_row_start'], loc['file_row_end_exclusive']
            if self.audit_assets and (not 0 <= a < b <= nrows or cursor + b - a > len(result)):
                raise ValueError('Quaternion sign row range mismatch')
            rows = np.arange(a, b, dtype=np.int64)
            negative = (bits[rows // 8] >> (rows % 8)[:, None]) & 1
            for hand, start in enumerate((3, 11)):
                signs = 1 - 2 * negative[:, hand].astype(np.int8)
                result[cursor:cursor + b - a, start:start + 4] *= signs[:, None]
            cursor += b - a
        if self.audit_assets and cursor != len(result):
            raise ValueError('Quaternion sign locations do not cover sample')
        return result

    def provenance(self):
        return dict(policy=POLICY, directory=str(self.directory), runtime_asset_audit_enabled=self.audit_assets,
                    manifest_fingerprint=self.manifest['fingerprint'], details=POLICY_DETAILS)
