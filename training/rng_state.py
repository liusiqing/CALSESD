"""保留全局训练随机数生成器"""
from contextlib import contextmanager
import random
import numpy as np
import torch

@contextmanager
def preserve_rng():
    py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
