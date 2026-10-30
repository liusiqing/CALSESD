from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch


class InferenceTiming:
    """CPU wall time or CUDA current-stream interval timing.

    Does not change model mode, gradients, actions, or RNG.
    CUDA measurements assume measured work runs on the recorded stream,
    or that any auxiliary streams are properly joined.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.pending = []
        self.cpu_samples = defaultdict(list)

    @contextmanager
    def measure(self, name, enabled=True):
        if not enabled:
            yield
            return

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record(stream)
                try:
                    yield
                finally:
                    end.record(stream)
                    self.pending.append((name, start, end))
        else:
            start = perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (perf_counter() - start) * 1000.0
                self.cpu_samples[name].append(elapsed_ms)

    def flush(self):
        """Finish measurements, summarize, then clear per-rollout samples."""
        samples = defaultdict(list)

        for name, values in self.cpu_samples.items():
            samples[name].extend(values)

        if self.pending:
            torch.cuda.synchronize(self.device)

            for name, start, end in self.pending:
                samples[name].append(start.elapsed_time(end))

        result = {}
        for name, values in samples.items():
            values = np.asarray(values, dtype=np.float64)

            result[name] = {
                "calls": int(values.size),
                "total_ms": float(values.sum()),
                "mean_ms": float(values.mean()),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(values.max()),
            }

        self.pending.clear()
        self.cpu_samples.clear()

        return {
            "timing_kind": (
                "cuda_stream_interval"
                if self.device.type == "cuda"
                else "cpu_wall"
            ),
            "stages": result,
        }