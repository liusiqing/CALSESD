"""Batch CVRP rollouts with per-row episode resets on a shared instance/traffic clock.

Full reset creates a new shared instance; reset_done resets only selected rows,
keeping node geometry, expected demands, current traffic, and other rows intact.
Requires a CVRPGridEnvForLatent-compatible environment (including Euclidean).
"""
from __future__ import annotations

import os
from typing import Optional
from numbers import Integral
import numpy as np
import torch
from torch_geometric.data import Data


class BatchCVRPEnv:
    _SNAPSHOT_TENSORS = (
        'current_nodes', 'vehicle_loads', 'total_costs', 'dones', 'terminated', 'truncated',
        'steps', 'visited_masks', 'revealed_masks', 'actual_demands_tensor', 'expected_demands_tensor',
    )
    def __init__(self, original_env, batch_size: int, valid_reward: Optional[float] = None,
                 step_penalty: Optional[float] = None, max_steps: Optional[int] = None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.env = original_env
        original_env.build_state_on_reset = False   # 批量环境自己构造状态
        self.batch_size = int(batch_size)
        self.device = original_env.device
        self.num_nodes = original_env.num_nodes
        self.max_steps = max_steps if max_steps is not None else original_env.max_steps
        self.failure_penalty = 100.0
        self.route_interval = 10
        self.route_enabled = False
        self.route_at_reset = False
        self._online_trigger = None
        self.norm_capacity = float(original_env.vehicle_capacity)
        self.norm_dist = max(float(original_env.grid_size) * float(os.environ.get("DVRP_NORM_DIST_K", 1.5)), 1e-6)
        self.valid_reward = 0.0 if valid_reward is None else float(valid_reward)
        self.step_penalty = 0.0 if step_penalty is None else float(step_penalty)
        self._instance_id = 0
        self._build_edges()

    def _traffic(self):
        """交通矩阵每次更新都是新对象,按对象身份缓存 GPU 张量。"""
        matrix = self.env.distance_matrix
        if getattr(self, '_traffic_src', None) is not matrix:
            self._traffic_src = matrix
            self._traffic_cache = torch.as_tensor(matrix, device=self.device, dtype=torch.float32)
        return self._traffic_cache

    def _build_edges(self):
        self.base_edge_index = self.env.edge_index.to(self.device).clone()
        e = self.base_edge_index.size(1)
        shifts = torch.arange(self.batch_size, device=self.device).repeat_interleave(e) * self.num_nodes
        self.batch_edge_index = self.base_edge_index.repeat(1, self.batch_size) + shifts
        self.batch_vec = torch.arange(self.batch_size, device=self.device).repeat_interleave(self.num_nodes)

    @property
    def online_trigger(self):
        return self._online_trigger

    @online_trigger.setter
    def online_trigger(self, trigger):
        # Protect direct assignment as well as the explicit binding API.
        self.set_online_trigger(trigger)

    def set_online_trigger(self, trigger):
        """Change binding only before the first action or after the FULL batch ends.

        Pending search always blocks changes. With some rows still running,
        binding/replacement/detachment is forbidden. Reassigning the identical
        object is a no-op (unless its search is pending). After a real binding
        change, use the appropriate full reset before starting the next batch.
        """
        current = self._online_trigger
        if current is not None and current.pending:
            raise RuntimeError('finish the pending search before changing trigger binding')
        if trigger is current:
            return
        if hasattr(self, 'steps') and bool((self.steps > 0).any()):
            full_batch_done = (hasattr(self, 'dones')
                and self.dones.shape == (self.batch_size,) and bool(self.dones.all()))
            if not full_batch_done:
                raise ValueError('cannot bind, replace or detach the trigger while any trajectory is still running')
        if trigger is not None:
            for name in ('reset_episode', 'context', 'decide', 'begin_attempt', 'finish_attempt'):
                if not callable(getattr(trigger, name, None)):
                    raise TypeError(f'trigger missing {name}')
            if trigger.config.batch_size != self.batch_size or trigger.pending:
                raise ValueError('trigger batch size/pending state mismatch')
        # Assign backing storage only after EVERY check has passed.
        self._online_trigger = trigger


    def online_decision(self):
        if self.online_trigger is None:
            raise RuntimeError('bind an online trigger first')
        remaining = (~self.visited_masks[:, self.customer_mask]).sum(1)
        return self.online_trigger.decide(steps=self.steps, active=~self.dones,
            remaining_customers=remaining, legal_action_counts=self.action_mask().sum(1))

    def begin_online_search(self, decision):
        if self.online_trigger is None:
            raise RuntimeError('bind an online trigger first')
        token = self.online_trigger.begin_attempt(decision)
        return token, self.get_batch_state()

    def finish_online_search(self, token, *, switched=None, elapsed_seconds=None, timed_out=False):
        if self.online_trigger is None:
            raise RuntimeError('bind an online trigger first')
        report = self.online_trigger.finish_attempt(token, switched=switched,
            elapsed_seconds=elapsed_seconds, timed_out=timed_out)
        return self.get_batch_state(), report

    def reset(self, num_customers=None, *, online_completed_updates=None, online_training=None):
        if self.online_trigger is not None:
            if self.online_trigger.pending:
                raise RuntimeError('finish pending search before reset')
            if (isinstance(online_completed_updates, bool) or not isinstance(online_completed_updates, Integral)
                    or online_completed_updates < 0 or not isinstance(online_training, bool)):
                raise ValueError('bound online reset requires completed-updates integer and online_training bool')
        elif online_completed_updates is not None or online_training is not None:
            raise RuntimeError('online reset options require a bound online trigger')
        self.env.reset(num_customers)
        self._instance_id += 1
        self.num_nodes = self.env.num_nodes
        self.norm_capacity = float(self.env.vehicle_capacity)
        self.norm_dist = max(float(self.env.grid_size) * float(os.environ.get("DVRP_NORM_DIST_K", 1.5)), 1e-6)
        self._build_edges()
        self.customer_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        self.customer_mask[torch.as_tensor(self.env.customer_nodes, device=self.device, dtype=torch.long)] = True
        self.current_nodes = torch.full((self.batch_size,), self.env.depot, dtype=torch.long, device=self.device)
        self.vehicle_loads = torch.full((self.batch_size,), float(self.env.vehicle_capacity), device=self.device)
        self.total_costs = torch.zeros(self.batch_size, device=self.device)
        self.dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros_like(self.dones)
        self.truncated = torch.zeros_like(self.dones)
        self.steps = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.visited_masks = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.bool, device=self.device)
        self.actual_demands_tensor = torch.zeros(self.batch_size, self.num_nodes, device=self.device)
        self.expected_demands_tensor = torch.zeros(self.batch_size, self.num_nodes, device=self.device)
        self.revealed_masks = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.bool, device=self.device)
        rows = torch.arange(self.batch_size, device=self.device)
        self._initialize_demands(rows)
        self._update_edge_attr()
        if self.online_trigger is not None:
            self.online_trigger.reset_episode(completed_updates=online_completed_updates, training=online_training)
        return self.get_batch_state()
    def snapshot(self):
        """Copy all mutable batch state plus the shared traffic/RNG state of this instance."""
        if not hasattr(self, 'steps'):
        raise RuntimeError('call reset() before snapshot()')
        data = {name: getattr(self, name).clone() for name in self._SNAPSHOT_TENSORS}
        data['env'] = self.env.save_state()
        data['instance_id'] = self._instance_id
        return data

    def restore(self, snapshot):
        """Rewind to a snapshot taken on the SAME instance. Does not touch the online trigger."""
        if snapshot['instance_id'] != self._instance_id:
            raise ValueError('snapshot belongs to a different instance; reset() invalidates snapshots')
        for name in self._SNAPSHOT_TENSORS:
            getattr(self, name).copy_(snapshot[name])
        self.env.load_state(snapshot['env'])
        self._update_edge_attr()
    def _initialize_demands(self, rows):
        if rows.numel() == 0:
            return
        n = rows.numel()
        nodes = torch.as_tensor(self.env.customer_nodes, device=self.device, dtype=torch.long)
        expected = np.array([float(self.env.expected_demands[v]) for v in self.env.customer_nodes])
        if getattr(self.env, "demand_mode", None) == "uchoa":
            sampled = np.repeat(expected[:, None], n, axis=1)
        else:
            sampled = np.clip(np.rint(self.env.np_rng.normal(
                expected[:, None], self.env.demand_noise_std, size=(len(expected), n))),
                1, self.env.vehicle_capacity)
        index = (rows[:, None], nodes[None, :])
        self.expected_demands_tensor[index] = torch.as_tensor(
            expected, dtype=torch.float32, device=self.device)[None, :].expand(n, -1)
        self.actual_demands_tensor[index] = torch.as_tensor(
            sampled.T, dtype=torch.float32, device=self.device)

    def reset_done(self, indices=None):
        """Reset ONLY rows currently done; return a new batched graph observation.

        indices=None selects all done rows. If indices are given, every selected
        row must already be done. Expected demands and shared traffic remain.
        """
        if self.online_trigger is not None:
            raise NotImplementedError('online phase supports full-batch episode resets only')
        if indices is None:
            rows = torch.where(self.dones)[0]
        elif isinstance(indices, torch.Tensor) and indices.dtype == torch.bool:
            if indices.shape != (self.batch_size,):
                raise ValueError("boolean indices must have shape [batch_size]")
            rows = torch.where(indices.to(self.device))[0]
        else:
            rows = torch.as_tensor(indices, device=self.device, dtype=torch.long).reshape(-1)
        if rows.numel() == 0:
            return self.get_batch_state()
        if torch.any(rows < 0) or torch.any(rows >= self.batch_size):
            raise ValueError("row index out of range")
        rows = torch.unique(rows)
        if not bool(self.dones[rows].all()):
            raise ValueError("reset_done may only reset done rows")
        self.current_nodes[rows] = self.env.depot
        self.vehicle_loads[rows] = float(self.env.vehicle_capacity)
        self.total_costs[rows] = 0.0
        self.steps[rows] = 0
        self.visited_masks[rows] = False
        self.revealed_masks[rows] = False
        self.actual_demands_tensor[rows] = 0.0
        self.expected_demands_tensor[rows] = 0.0
        self._initialize_demands(rows)
        self.dones[rows] = False
        self.terminated[rows] = False
        self.truncated[rows] = False
        return self.get_batch_state()

    def _update_edge_attr(self):
        src, dst = self.base_edge_index
        traffic = self._traffic()
        self.batch_edge_attr = (traffic[src, dst].unsqueeze(-1) / self.norm_dist).repeat(self.batch_size, 1)

    def get_batch_state(self):
        B,N=self.batch_size,self.num_nodes
        demand=torch.where(self.revealed_masks,self.actual_demands_tensor,self.expected_demands_tensor)/self.norm_capacity
        expected=self.expected_demands_tensor/self.norm_capacity
        current=(torch.arange(N,device=self.device)[None,:]==self.current_nodes[:,None]).float()
        load=(self.vehicle_loads/self.norm_capacity)[:,None].expand(-1,N)
        traffic=self._traffic()
        distance=traffic[self.current_nodes]/self.norm_dist
        x=torch.stack((demand,expected,self.visited_masks.float(),current,load,distance),-1).reshape(-1,6)
        customer=self.customer_mask[None,:].expand(B,-1)
        depot=(torch.arange(N,device=self.device)==self.env.depot)[None,:].expand(B,-1)
        aux=torch.stack((customer,depot,self.revealed_masks),-1).float().reshape(-1,3)
        pos=torch.as_tensor(self.env.coords,dtype=torch.float32,device=self.device)/max(1.,float(self.env.grid_size))
        interval=max(1,self.route_interval);budget=max(1,self.max_steps or 1)
        phase=float(self.env.step_count%self.env.traffic_update_interval)/self.env.traffic_update_interval
        context=torch.stack((torch.full((B,),phase,device=self.device),
            (1-self.steps.float()/budget).clamp(0,1),(self.steps%interval).float()/interval,
            torch.full((B,),float(self.route_enabled),device=self.device),
            torch.full((B,),interval/budget,device=self.device),
            torch.full((B,),float(self.route_at_reset),device=self.device)),-1)
        state=Data(x=x,edge_index=self.batch_edge_index,edge_attr=self.batch_edge_attr,
                   batch=self.batch_vec,pos=pos.repeat(B,1),node_aux=aux,context=context)
        if self.online_trigger is not None:
            from .online_state import attach_online_context
            state=attach_online_context(state,self.online_trigger,self.steps)
            state.active_mask=(~self.dones).clone()
            state.online_search_pending=bool(self.online_trigger.pending)
            state.episode_steps=self.steps.clone()
        return state

    def action_mask(self):
        mask=self.customer_mask[None,:].expand(self.batch_size,-1).clone()&~self.visited_masks
        mask&=~(self.revealed_masks&(self.actual_demands_tensor>self.vehicle_loads[:,None]))
        mask[:,self.env.depot]=self.current_nodes!=self.env.depot
        completed=self.visited_masks[:,self.customer_mask].all(1)
        mask[completed,self.env.depot]=True
        mask[self.dones]=False;mask[self.dones,self.env.depot]=True
        if not bool(mask.any(1).all()):raise RuntimeError('No legal action; check demand feasibility')
        return mask

    def step(self, actions, action_mask=None, return_info: bool = False):
        if self.online_trigger is not None and self.online_trigger.pending:
            raise RuntimeError('do not advance the live environment during a pending online search')
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).reshape(-1)
        if actions.numel() != self.batch_size or bool(((actions < 0) | (actions >= self.num_nodes)).any()):
            raise ValueError("actions must contain one valid node index per row")
        active = ~self.dones
        mask = self.action_mask() if action_mask is None else action_mask
        if mask.shape != (self.batch_size, self.num_nodes):
            raise ValueError("action_mask shape mismatch")
        legal=mask.gather(1,actions[:,None]).squeeze(1)
        if bool((active&~legal).any()):raise ValueError("Illegal action under action_mask")
        if bool(active.any()):
            self.env._advance_traffic_if_due()
            self._update_edge_attr()
        matrix = self._traffic()
        costs = matrix[self.current_nodes, actions]
        self.total_costs[active] += costs[active]
        self.current_nodes[active] = actions[active]
        self.steps[active] += 1
        candidate = active & self.customer_mask[actions] & ~self.visited_masks.gather(1, actions[:, None]).squeeze(1)
        required = self.actual_demands_tensor.gather(1, actions[:, None]).squeeze(1)
        served = candidate & (self.vehicle_loads >= required)
        row = torch.where(candidate)[0]
        self.revealed_masks[row, actions[row]] = True
        row = torch.where(served)[0]
        self.vehicle_loads[row] -= required[row]
        self.visited_masks[row, actions[row]] = True
        at_depot = active & (actions == self.env.depot)
        self.vehicle_loads[at_depot] = float(self.env.vehicle_capacity)
        customers = torch.where(self.customer_mask)[0]
        served_all = self.visited_masks[:, customers].all(dim=1)
        self.terminated |= active & served_all & at_depot
        if self.max_steps is not None:
            self.truncated |= active & ~self.terminated & (self.steps >= self.max_steps)
        self.dones |= self.terminated | self.truncated
        rewards = -costs / self.norm_dist - self.step_penalty + self.valid_reward * served.float()
        failed=active&self.truncated&~self.terminated
        remaining=(~self.visited_masks[:,self.customer_mask]).sum(1)
        rewards[failed]-=self.failure_penalty*(remaining[failed].float()+1.)
        rewards[~active] = 0.0
        state = self.get_batch_state()
        if return_info:
            return state, rewards, self.dones.clone(), dict(
                terminated=self.terminated.clone(), truncated=self.truncated.clone())
        return state, rewards, self.dones.clone()
