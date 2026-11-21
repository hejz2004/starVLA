"""One-batch lookahead without modifying the trainer's consumed-batch cursor."""
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors(child)


class CudaPrefetchIterator:
    def __init__(self, source, device):
        self.source = iter(source)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("cuda_prefetch requires a CUDA device")
        self.stream = torch.cuda.Stream(device=self.device)
        self.pending = None
        self.host_inflight = []
        self.exhausted = False

    def __iter__(self):
        return self

    def _enqueue(self):
        from accelerate.utils import send_to_device
        try:
            host = next(self.source)
        except StopIteration:
            self.exhausted = True
            return
        if not all(x.device.type == "cpu" and x.is_pinned() for x in tensors(host)):
            raise ValueError("cuda_prefetch requires pinned CPU batches (disable loader device placement)")
        self.host_inflight = [(h, e) for h, e in self.host_inflight if not e.query()]
        with torch.cuda.stream(self.stream):
            batch = send_to_device(host, self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.host_inflight.append((host, ready))
        self.pending = batch, ready

    def __next__(self):
        if self.pending is None and not self.exhausted:
            self._enqueue()
        if self.pending is None:
            raise StopIteration
        batch, ready = self.pending
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready)
        for tensor in tensors(batch):
            tensor.record_stream(current)
        self.pending = None
        if not self.exhausted:
            self._enqueue()
        return batch
