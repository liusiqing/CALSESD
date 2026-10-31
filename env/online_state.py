"""Attach online controller context WITHOUT replacing the physical environment.

This is an adapter, not a rewritten BatchCVRPEnv/get_batch_state implementation.
Keep original x/node_aux/context/edge features and transitions unchanged.
Critic must later explicitly consume graph-level online_context[B,6].
"""

from copy import copy
import torch


def attach_online_context(state, trigger, steps):
    if not torch.is_tensor(getattr(state, "x", None)) or not torch.is_tensor(getattr(state, "batch", None)):
        raise TypeError("Expected a BatchCVRPEnv PyG state with x and batch")
    if state.batch.ndim != 1 or state.batch.numel() != state.x.size(0):
        raise ValueError("invalid node/batch layout")
    B = trigger.config.batch_size
    if not state.batch.numel() or int(state.batch.min()) != 0 or int(state.batch.max()) + 1 != B:
        raise ValueError("state graph count does not match trigger batch size")
    # PyG Data's shallow copy duplicates its attribute store but reuses existing
    # tensors. Adding the new attribute avoids cloning large edge tensors.
    result = copy(state)
    result.online_context = torch.tensor(trigger.context(steps), device=state.x.device, dtype=torch.float32)
    if result.online_context.shape != (B, len(trigger.context_fields)):
        raise ValueError("online context shape mismatch")
    return result


__all__ = ["attach_online_context"]
