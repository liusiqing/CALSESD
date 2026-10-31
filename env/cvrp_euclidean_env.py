"""Euclidean dynamic CVRP environments compatible with cvrp_env.py."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

try:
    from .cvrp_env import CVRPGridEnv
except ImportError:
    from cvrp_env import CVRPGridEnv


def pick_norm_grid_size(num_customers: int) -> int:
    if num_customers <= 75:
        return 8
    if num_customers <= 150:
        return 12
    return 15


class CVRPEuclideanEnv(CVRPGridEnv):
    def __init__(self, coords, demands, capacity: int, depot_idx: int = 0,
                 norm_grid_size: Optional[int] = None, seed: Optional[int] = None,
                 device: str | torch.device | None = None, demand_noise_std: float = 1.0,
                 traffic_transition_beta: float = 0.07, traffic_noise_std: float = 1.0,
                 traffic_trend: float = 5.0, use_knn_graph: bool = False,
                 traffic_trend_mode: str = "additive", traffic_trend_ratio: float = 0.3,
                 k_neighbors: Optional[int] = None, demand_mode: str = "uchoa",
                 synth_demand_lo: int = 1, synth_demand_hi: int = 5,
                 synth_capacity: int = 15, max_steps: Optional[int] = None):
        raw_coords = np.asarray(coords, dtype=np.float64)
        raw_demands = np.asarray(demands, dtype=np.float64)
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
            raise ValueError("coords must have shape [num_nodes, 2].")
        if len(raw_coords) != len(raw_demands) or len(raw_coords) < 2:
            raise ValueError("coords and demands must have equal length of at least 2.")
        if not 0 <= depot_idx < len(raw_coords):
            raise ValueError("depot_idx is out of range.")
        if demand_mode not in {"uchoa", "synthetic"}:
            raise ValueError("demand_mode must be 'uchoa' or 'synthetic'.")

        order = [depot_idx] + [i for i in range(len(raw_coords)) if i != depot_idx]
        raw_coords = raw_coords[order]
        raw_demands = raw_demands[order]
        scale = int(norm_grid_size or pick_norm_grid_size(len(raw_coords) - 1))
        translated = raw_coords - raw_coords.min(axis=0, keepdims=True)
        extent = max(float(translated.max()), 1.0)
        self.coords = translated / extent * float(scale)
        self._instance_demands = {i: max(0, int(round(raw_demands[i]))) for i in range(1, len(raw_coords))}
        self.demand_mode = demand_mode
        self.synth_demand_lo = int(synth_demand_lo)
        self.synth_demand_hi = int(synth_demand_hi)
        self._euclidean_capacity = int(capacity) if demand_mode == "uchoa" else int(synth_capacity)

        super().__init__(
            grid_size=scale, vehicle_capacity=self._euclidean_capacity, num_customers=len(raw_coords) - 1,
            seed=seed, device=device, demand_noise_std=(0.0 if demand_mode == "uchoa" else demand_noise_std),
            traffic_transition_beta=traffic_transition_beta, traffic_noise_std=traffic_noise_std,
            traffic_trend=traffic_trend, use_knn_graph=use_knn_graph,
            traffic_trend_mode=traffic_trend_mode, traffic_trend_ratio=traffic_trend_ratio,
            k_neighbors=(k_neighbors if k_neighbors is not None else min(40, max(1, (len(raw_coords) - 1) // 5))),
            max_steps=max_steps,
        )

    def _set_grid_geometry(self) -> None:
        self.num_nodes = len(self.coords)
        self.depot = 0
        self.nodes = list(range(self.num_nodes))
        self.node_to_coord = {i: (float(self.coords[i, 0]), float(self.coords[i, 1])) for i in range(self.num_nodes)}
        differences = self.coords[:, None, :] - self.coords[None, :, :]
        self.baseline_distance_matrix = np.sqrt((differences ** 2).sum(axis=-1)).astype(np.float64)
        np.fill_diagonal(self.baseline_distance_matrix, 0.0)

    def _create_knn_edge_index(self) -> torch.Tensor:
        coords = torch.as_tensor(self.coords, dtype=torch.float32, device=self.device)
        dists = torch.cdist(coords, coords, p=2)
        k = min(max(1, self.k_neighbors), self.num_nodes - 1)
        dists.fill_diagonal_(float("inf"))
        indices = dists.topk(k, dim=1, largest=False).indices
        src = torch.arange(self.num_nodes, device=self.device).view(-1, 1).expand_as(indices)
        return torch.stack((src.reshape(-1), indices.reshape(-1)), dim=0).long().contiguous()

    def reset(self, num_customers=None):
        if num_customers is not None and int(num_customers) != self.num_nodes - 1:
            raise ValueError("CVRPEuclideanEnv has a fixed instance; num_customers cannot be changed")
        self.nodes = list(range(self.num_nodes))
        self.customer_nodes = list(range(1, self.num_nodes))
        self.num_customers = len(self.customer_nodes)
        if self.demand_mode == "uchoa" and any(v > self.vehicle_capacity for v in self._instance_demands.values()):
            raise ValueError("Demand exceeds capacity; split delivery is not supported")
        if self.demand_mode == "uchoa":
            self.expected_demands = self._instance_demands.copy()
        else:
            self.expected_demands = {
                node: self.rng.randint(self.synth_demand_lo, self.synth_demand_hi)
                for node in self.customer_nodes
            }
        self.actual_demands = {node: None for node in self.customer_nodes}
        self.visited = set()
        self.vehicle_load = self.vehicle_capacity
        self.current_node = self.depot
        self.total_cost = 0.0
        self.step_count = 0
        self.time_step = 0
        self.current_traffic_matrix = self.baseline_distance_matrix.copy()
        self.distance_matrix = self._generate_traffic_matrix()
        return self._get_state() if self.build_state_on_reset else None

    def _reveal_demand(self, node: int) -> int:
        if self.actual_demands[node] is None:
            if self.demand_mode == "uchoa":
                self.actual_demands[node] = int(self.expected_demands[node])
            else:
                self.actual_demands[node] = super()._reveal_demand(node)
        return int(self.actual_demands[node])


class CVRPEuclideanEnvForLatent(CVRPEuclideanEnv):
    def __init__(self, coords, demands, capacity: int, depot_idx: int = 0,
                 norm_grid_size: Optional[int] = None, seed: Optional[int] = None,
                 device: str | torch.device | None = None, demand_noise_std: float = 1.0,
                 traffic_noise_std: float = 1.0, traffic_trend: float = 5.0,
                 traffic_beta: float = 0.07, demand_mode: str = "uchoa",
                 synth_capacity: int = 15, max_steps: Optional[int] = None,
                 traffic_trend_mode: str = "additive", traffic_trend_ratio: float = 0.3):
        super().__init__(
            coords=coords, demands=demands, capacity=capacity, depot_idx=depot_idx,
            norm_grid_size=norm_grid_size, seed=seed, device=device,
            demand_noise_std=demand_noise_std, traffic_transition_beta=traffic_beta,
            traffic_noise_std=traffic_noise_std, traffic_trend=traffic_trend,
            traffic_trend_mode=traffic_trend_mode, traffic_trend_ratio=traffic_trend_ratio,
            use_knn_graph=True, k_neighbors=min(40, max(1, (len(coords) - 1) // 5)),
            demand_mode=demand_mode, synth_capacity=synth_capacity, max_steps=max_steps,
        )
        self.traffic_update_interval = 5
