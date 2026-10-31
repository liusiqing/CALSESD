"""Dynamic grid-based CVRP environment with reproducible traffic and demand revelation."""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from .traffic import generate_traffic_matrix

class CVRPGridEnv:
    build_state_on_reset = True 
    def __init__(
        self,
        grid_size: int = 15,
        vehicle_capacity: int = 15,
        num_customers: int = 200,
        seed: Optional[int] = None,
        device: str | torch.device | None = None,
        demand_noise_std: float = 1.0,
        traffic_transition_beta: float = 0.07,
        traffic_noise_std: float = 1.0,
        traffic_trend: float = 5.0,
        traffic_trend_mode: str = "additive",
        traffic_trend_ratio: float = 0.3,
        use_knn_graph: bool = False,
        k_neighbors: int = 200,
        traffic_update_interval: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive.")
        if not 0.0 <= traffic_transition_beta <= 1.0:
            raise ValueError("traffic_transition_beta must be in [0, 1].")
        if demand_noise_std < 0 or traffic_noise_std < 0:
            raise ValueError("Noise standard deviations must be non-negative.")
        if traffic_trend_mode not in ("additive", "relative") or traffic_trend_ratio < 0:
            raise ValueError("invalid traffic_trend_mode/traffic_trend_ratio.")

        self.grid_size = int(grid_size)
        self.num_nodes = self.grid_size * self.grid_size
        self.vehicle_capacity = int(vehicle_capacity)
        self.num_customers = min(int(num_customers), self.num_nodes - 1)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.demand_noise_std = float(demand_noise_std)
        self.traffic_transition_beta = float(traffic_transition_beta)
        self.traffic_noise_std = float(traffic_noise_std)
        self.traffic_trend = float(traffic_trend)
        self.traffic_trend_mode = traffic_trend_mode
        self.traffic_trend_ratio = float(traffic_trend_ratio)
        self.use_knn_graph = bool(use_knn_graph)
        self.k_neighbors = int(k_neighbors)
        self.traffic_update_interval = int(
            traffic_update_interval if traffic_update_interval is not None else (5 if use_knn_graph else 1)
        )
        if self.traffic_update_interval < 1:
            raise ValueError("traffic_update_interval must be at least 1.")
        self.max_steps = max_steps

        self._torch_generator = torch.Generator(device="cpu")
        self._seed_rngs()
        self.depot = 0
        self._set_grid_geometry()
        self.edge_index = self._create_knn_edge_index() if self.use_knn_graph else self._create_full_edge_index()
        self.reset()

    def _seed_rngs(self) -> None:
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)
        if self.seed is None:
            self._torch_generator.seed()
        else:
            self._torch_generator.manual_seed(int(self.seed))

    def _set_grid_geometry(self) -> None:
        coords = np.array(
            [(node % self.grid_size, node // self.grid_size) for node in range(self.num_nodes)], dtype=np.float64
        )
        center_node = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
        if center_node != self.depot:
            coords[[self.depot, center_node]] = coords[[center_node, self.depot]]
        self.coords = coords
        self.nodes = list(range(self.num_nodes))
        self.node_to_coord = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(self.num_nodes)}
        diffs = coords[:, None, :] - coords[None, :, :]
        self.baseline_distance_matrix = np.abs(diffs).sum(axis=-1).astype(np.float64)
        np.fill_diagonal(self.baseline_distance_matrix, 0.0)

    def _create_full_edge_index(self) -> torch.Tensor:
        src, dst = np.where(~np.eye(self.num_nodes, dtype=bool))
        return torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=self.device)

    def _create_knn_edge_index(self) -> torch.Tensor:
        coords = torch.as_tensor(self.coords, dtype=torch.float32, device=self.device)
        dists = torch.cdist(coords, coords, p=1)
        k = min(max(1, self.k_neighbors), self.num_nodes - 1)
        dists.fill_diagonal_(float("inf"))
        indices = dists.topk(k, dim=1, largest=False).indices
        src = torch.arange(self.num_nodes, device=self.device).view(-1, 1).expand_as(indices)
        return torch.stack((src.reshape(-1), indices.reshape(-1)), dim=0).long().contiguous()

    def reset(self, num_customers: Optional[int] = None) -> Data:
        if num_customers is not None:
            self.num_customers = min(int(num_customers), self.num_nodes - 1)
        self.nodes = list(range(self.num_nodes))
        self.customer_nodes = self.rng.sample(list(range(1, self.num_nodes)), self.num_customers)
        self.expected_demands = {node: self.rng.randint(1, 5) for node in self.customer_nodes}
        self.actual_demands = {node: None for node in self.customer_nodes}
        self.visited: set[int] = set()
        self.vehicle_load = self.vehicle_capacity
        self.current_node = self.depot
        self.total_cost = 0.0
        self.step_count = 0
        self.time_step = 0
        self.current_traffic_matrix = self.baseline_distance_matrix.copy()
        self.distance_matrix = self._generate_traffic_matrix()
        return self._get_state() if self.build_state_on_reset else None

    def _generate_traffic_matrix(self) -> np.ndarray:
        self.current_traffic_matrix = generate_traffic_matrix(
            self.current_traffic_matrix, self.baseline_distance_matrix,
            beta=self.traffic_transition_beta, noise_std=self.traffic_noise_std,
            trend=self.traffic_trend, generator=self._torch_generator,
            mode=self.traffic_trend_mode, ratio=self.traffic_trend_ratio)
        self.time_step += 1
        return self.current_traffic_matrix

    def _advance_traffic_if_due(self) -> None:
        self.step_count += 1
        if self.step_count % self.traffic_update_interval == 0:
            self.distance_matrix = self._generate_traffic_matrix()

    def _reveal_demand(self, node: int) -> int:
        value = self.actual_demands[node]
        if value is None:
            raw = self.np_rng.normal(self.expected_demands[node], self.demand_noise_std)
            value = int(np.clip(np.rint(raw), 1, self.vehicle_capacity))
            self.actual_demands[node] = value
        return int(value)

    def _get_state(self) -> Data:
        features = []
        for node in range(self.num_nodes):
            expected = float(self.expected_demands.get(node, 0))
            actual = self.actual_demands.get(node)
            observed = float(actual) if actual is not None else expected
            features.append([
                observed,
                expected,
                float(node in self.visited),
                float(node == self.current_node),
                float(self.vehicle_load),
                float(self.distance_matrix[self.current_node, node]),
            ])
        src, dst = self.edge_index
        edge_attr = torch.as_tensor(
            self.distance_matrix[src.detach().cpu().numpy(), dst.detach().cpu().numpy()],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        return Data(
            x=torch.as_tensor(features, dtype=torch.float32, device=self.device),
            edge_index=self.edge_index,
            edge_attr=edge_attr,
        )

    def step(self, action: int):
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer node index.")
        action = int(action)
        if not 0 <= action < self.num_nodes:
            raise ValueError(f"action must be in [0, {self.num_nodes - 1}].")

        self._advance_traffic_if_due()
        cost = float(self.distance_matrix[self.current_node, action])
        self.total_cost += cost
        self.current_node = action

        if action in self.customer_nodes and action not in self.visited:
            demand = self._reveal_demand(action)
            if self.vehicle_load >= demand:
                self.vehicle_load -= demand
                self.visited.add(action)
        if action == self.depot:
            self.vehicle_load = self.vehicle_capacity

        completed = len(self.visited) == len(self.customer_nodes) and self.current_node == self.depot
        truncated = self.max_steps is not None and self.step_count >= self.max_steps
        return self._get_state(), -cost, bool(completed or truncated)

    def save_state(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes[:], "customer_nodes": self.customer_nodes[:],
            "expected_demands": self.expected_demands.copy(), "actual_demands": self.actual_demands.copy(),
            "visited": self.visited.copy(), "vehicle_load": self.vehicle_load,
            "current_node": self.current_node, "total_cost": self.total_cost,
            "step_count": self.step_count, "time_step": self.time_step,
            "current_traffic_matrix": self.current_traffic_matrix.copy(),
            "distance_matrix": self.distance_matrix.copy(), "rng_state": self.rng.getstate(),
            "np_rng_state": self.np_rng.get_state(), "torch_rng_state": self._torch_generator.get_state().clone(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.nodes = state["nodes"][:]
        self.customer_nodes = state["customer_nodes"][:]
        self.expected_demands = state["expected_demands"].copy()
        self.actual_demands = state["actual_demands"].copy()
        self.visited = state["visited"].copy()
        self.vehicle_load = state["vehicle_load"]
        self.current_node = state["current_node"]
        self.total_cost = state["total_cost"]
        self.step_count = state["step_count"]
        self.time_step = state["time_step"]
        self.current_traffic_matrix = state["current_traffic_matrix"].copy()
        self.distance_matrix = state.get("distance_matrix", self.current_traffic_matrix).copy()
        self.rng.setstate(state["rng_state"])
        self.np_rng.set_state(state["np_rng_state"])
        if "torch_rng_state" in state:
            self._torch_generator.set_state(state["torch_rng_state"])


class CVRPGridEnvForLatent(CVRPGridEnv):
    def __init__(self, grid_size: int = 12, vehicle_capacity: int = 15, num_customers: int = 100,
                 seed: Optional[int] = None, device: str | torch.device | None = None,
                 demand_noise_std: float = 1.0, traffic_noise_std: float = 1.0,
                 traffic_trend: float = 5.0, traffic_beta: float = 0.07, max_steps: Optional[int] = None,
                 traffic_trend_mode: str = "additive", traffic_trend_ratio: float = 0.3):
        super().__init__(
            grid_size=grid_size, vehicle_capacity=vehicle_capacity, num_customers=num_customers,
            seed=seed, device=device, demand_noise_std=demand_noise_std,
            traffic_transition_beta=traffic_beta, traffic_noise_std=traffic_noise_std,
            traffic_trend=traffic_trend, use_knn_graph=True,
            traffic_trend_mode=traffic_trend_mode, traffic_trend_ratio=traffic_trend_ratio,
            k_neighbors=min(40, max(1, num_customers // 5)), traffic_update_interval=5,
            max_steps=max_steps,
        )
