"""Decode nearby requests once, keeping the established per-query selector.

Only requests with the same physical video and logical episode interval share
a decoder. There is no persistent image/GOP cache: a group retains the two last
presentation frames and the requested resized outputs. A fresh seek starts
every bounded group, and its decoder is closed before the next seek.
"""
from collections import defaultdict, deque
from itertools import chain
import math

from .umi_video_time import seconds


def validate_batch_span(value):
    span = float(value)
    if not math.isfinite(span) or not 0 < span <= 2:
        raise ValueError("batched_video_max_span_seconds must be in (0, 2]")
    return span


def decode_video_requests(requests, *, open_video, make_window, render_frame,
                          max_span_seconds=2):
    """Return keyed images/timing for ``(key, camera, timestamp)`` requests.

    Callbacks preserve the caller's path checks, VideoTimeWindow construction,
    pixel conversion and timing metadata. Requests may arrive out of order or
    repeat; selection is performed in timestamp order and results keep keys.
    """
    span = seconds(validate_batch_span(max_span_seconds))
    owners = defaultdict(list)
    for key, camera, timestamp in requests:
        owner = (camera["video_path"], seconds(camera["from_timestamp"]),
                 seconds(camera["to_timestamp"]))
        owners[owner].append((key, camera, timestamp))
    results = {}
    for owner_requests in owners.values():
        container, stream = open_video(owner_requests[0][1])
        pending = [(make_window(camera, timestamp, stream), key, camera)
                   for key, camera, timestamp in owner_requests]
        pending.sort(key=lambda request: request[0].target)
        begin = 0
        while begin < len(pending):
            stop = begin + 1
            while (stop < len(pending)
                   and pending[stop][0].target - pending[begin][0].target <= span):
                stop += 1
            group = pending[begin:stop]
            container.seek(min(window.seek_pts for window, _, _ in group),
                           stream=stream, backward=True, any_frame=False)
            decoder = container.decode(video=0)
            previous = deque(maxlen=2)

            def tracked_frames():
                for frame in decoder:
                    # The selector retains the first frame when equal PTS
                    # have equal distance. Preserve that identity in replay,
                    # rather than allowing later duplicates to evict it.
                    if (frame.pts is not None
                            and (not previous or frame.pts != previous[-1].pts)):
                        previous.append(frame)
                    yield frame

            tracked = tracked_frames()
            try:
                for window, key, camera in group:
                    # Do not track replayed frames again. With duplicate
                    # queries, replay may stop at its first frame; appending
                    # that frame would corrupt the presentation order.
                    closest, closest_time = window.select(chain(tuple(previous), tracked))
                    results[key] = render_frame(camera, window, stream, closest, closest_time)
            finally:
                tracked.close()
                close = getattr(decoder, "close", None)
                if close is not None:
                    close()
            begin = stop
    return results
