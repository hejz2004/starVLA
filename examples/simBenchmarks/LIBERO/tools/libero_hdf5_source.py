"""Read pinned official LIBERO HDF5 sources without downloading image payloads."""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CachedHTTPRangeReader(io.BufferedIOBase):
    """Seekable HTTP reader with validated, atomically cached byte ranges.

    HDF5 metadata and low-dimensional arrays are tiny compared with RGB images.
    Pin the URL to a repository revision and supply the size from that revision's
    tree. Never accept a server response for a different range or whole file.
    """

    def __init__(self, url: str, size: int, cache: Path, block_size: int = 262144):
        super().__init__()
        self.source_url = url
        self.resolved_url = url
        self.size = size
        self.position = 0
        self.block_size = block_size
        self.cache = cache / hashlib.sha256(url.encode()).hexdigest()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.network_bytes = 0
        self.network_requests = 0
        self.started = time.monotonic()
        self._memory = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unknown seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def _block(self, index):
        if index in self._memory:
            return self._memory[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        expected = end - start + 1
        path = self.cache / f"{index:08d}.bin"
        if path.exists():
            data = path.read_bytes()
            if len(data) != expected:
                raise RuntimeError(f"Incomplete cached source block: {path}")
        else:
            for attempt in range(3):
                with self.session.get(
                    self.resolved_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 90),
                    stream=True,
                ) as response:
                    if response.status_code in (401, 403) and self.resolved_url != self.source_url:
                        self.resolved_url = self.source_url
                        continue
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status_code != 206 or match is None:
                        raise RuntimeError(f"Source range request failed: HTTP {response.status_code}, block {index}")
                    if tuple(map(int, match.groups())) != (start, end, self.size):
                        raise RuntimeError(f"Source returned a different byte range for block {index}")
                    data = response.raw.read(expected + 1)
                    if len(data) != expected:
                        raise RuntimeError(f"Truncated source byte range for block {index}")
                    self.resolved_url = response.url
                    self.network_bytes += len(data)
                    self.network_requests += 1
                    temporary = path.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    break
            else:
                raise RuntimeError(f"Unable to authenticate public source request for block {index}")
        self._memory[index] = data
        return data

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        if size is None or size < 0:
            if remaining > 64 * 1024 * 1024:
                raise ValueError("Refusing an unbounded whole-HDF5 read")
            size = remaining
        size = min(size, remaining)
        chunks = []
        while size:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            length = min(size, len(block) - offset)
            chunks.append(block[offset:offset + length])
            self.position += length
            size -= length
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        self.session.close()
        super().close()
