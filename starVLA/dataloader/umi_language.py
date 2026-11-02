"""Read Luna language with explicit legacy-anchor or strict-window semantics.

Strict mode checks every row and fails on stale/crossing indexes. It never
changes membership or substitutes labels at runtime.
"""
import math
from numbers import Integral, Real


FINE_COLUMNS = (
    "annotation.instruction",
    "annotation.valid",
    "annotation.segment_index",
    "annotation.curated_start_s",
    "annotation.curated_end_s",
)
FINE_LANGUAGE_SOURCE = "luna_fine"
SAME_SEGMENT_PROTOCOL = "luna-segment-v1-anchor-and-future16-same-valid-segment"
SAME_SEGMENT_POLICY = {
    "version": SAME_SEGMENT_PROTOCOL,
    "instruction": "identical nonempty instruction in the anchor and every future action row",
    "interval": "one episode-local curated half-open [start,end) interval",
    "future_segment_crossing": "forbidden, including distinct segments with identical text",
    "invalid_label": "raise; no fallback, skipping, padding, masking, or resampling",
}
FINE_LANGUAGE_POLICY = {
    "version": "luna-anchor-instruction-v1",
    "instruction": "annotation.instruction at the current anchor frame",
    "interval": "episode-local curated half-open [start,end); raw model times are not used",
    "future_segment_crossing": "allowed; existing numeric windows and sampling weights unchanged",
    "invalid_label": "raise; no fallback, skipping, or resampling",
}


def _finite_seconds(value, field):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Fine Luna language requires finite {field}")
    return float(value)


def extract_fine_language(table, *, same_segment=False):
    """Return ``(instruction, trace)`` from row zero of an Arrow window table.

The caller verifies numeric row identities. ``same_segment`` additionally
requires one valid instruction/segment across all 17 rows. The default retains
legacy anchor-only semantics. Source instruction whitespace is preserved.
"""
    required = set(FINE_COLUMNS) | {"timestamp"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"Missing fine Luna language columns: {sorted(missing)}")
    if table.num_rows < 1:
        raise ValueError("Fine Luna language requires a current anchor row")
    anchor = {key: table[key][0].as_py() for key in required}
    if anchor["annotation.valid"] is not True:
        raise ValueError("Fine Luna anchor annotation.valid must be true")
    instruction = anchor["annotation.instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Fine Luna anchor instruction must be a nonempty string")
    segment = anchor["annotation.segment_index"]
    if isinstance(segment, bool) or not isinstance(segment, Integral) or segment < 0:
        raise ValueError("Fine Luna anchor segment_index must be a nonnegative integer")
    start = _finite_seconds(anchor["annotation.curated_start_s"], "annotation.curated_start_s")
    end = _finite_seconds(anchor["annotation.curated_end_s"], "annotation.curated_end_s")
    timestamp = _finite_seconds(anchor["timestamp"], "timestamp")
    if not 0 <= start < end or not start <= timestamp < end:
        raise ValueError("Fine Luna anchor lies outside its curated half-open interval")
    if same_segment:
        if table.num_rows != 17 or "episode_index" not in table.column_names:
            raise ValueError("Same-segment language requires 17 rows with episode identity")
        columns = {key: table[key].to_pylist() for key in required | {"episode_index"}}
        episode = columns["episode_index"][0]
        if isinstance(episode, bool) or not isinstance(episode, Integral) or episode < 0:
            raise ValueError("Same-segment language requires nonnegative integer episode identity")
        for i in range(17):
            current_segment = columns["annotation.segment_index"][i]
            if (columns["episode_index"][i] != episode
                    or columns["annotation.valid"][i] is not True
                    or columns["annotation.instruction"][i] != instruction
                    or isinstance(current_segment, bool)
                    or not isinstance(current_segment, Integral)
                    or current_segment != segment
                    or columns["annotation.curated_start_s"][i] != start
                    or columns["annotation.curated_end_s"][i] != end):
                raise ValueError(f"Same-segment language violation at window row {i}")
            timestamp_i = _finite_seconds(columns["timestamp"][i], "timestamp")
            if not start <= timestamp_i < end:
                raise ValueError(f"Same-segment row {i} lies outside its curated half-open interval")
    return instruction, {
        "language_source": FINE_LANGUAGE_SOURCE,
        "language_policy": SAME_SEGMENT_PROTOCOL if same_segment else FINE_LANGUAGE_POLICY["version"],
        "annotation_segment_index": int(segment),
        "annotation_curated_start_s": start,
        "annotation_curated_end_s": end,
        "future_segment_crossing": "forbidden" if same_segment else "allowed; window membership unchanged",
    }
