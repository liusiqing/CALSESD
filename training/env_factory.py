"""Grid training/deployment task factory; traffic engine is unchanged."""

from CALSESD.env import BatchCVRPEnv, CVRPGridEnvForLatent


def make_env(cfg, seed, device):
    base = CVRPGridEnvForLatent(
        grid_size=cfg.grid_size,
        vehicle_capacity=cfg.vehicle_capacity,
        num_customers=cfg.num_customers,
        seed=int(seed),
        device=device,
        demand_noise_std=cfg.demand_noise_std,
        traffic_noise_std=cfg.traffic_noise_std,
        traffic_trend=cfg.traffic_trend,
        traffic_beta=cfg.traffic_beta,
        max_steps=cfg.max_episode_steps,
        traffic_trend_mode=cfg.traffic_trend_mode,
        traffic_trend_ratio=cfg.traffic_trend_ratio,
    )
    env = BatchCVRPEnv(base, cfg.batch_size, valid_reward=0.0, step_penalty=0.0, max_steps=cfg.max_episode_steps)
    env.failure_penalty = cfg.failure_penalty
    env.reset()
    return env


__all__ = ["make_env"]
