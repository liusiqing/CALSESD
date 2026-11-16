"""当前状态上的 Critic 引导共享隐变量进化，不执行环境 rollout。"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
from time import perf_counter
import torch


@dataclass
class HybridSearchConfig:
    candidates: int = 64
    generations: int = 3
    latent_bound: float = 1.0
    elite_fraction: float = 0.25
    immigrant_fraction: float = 0.125
    crossover_alpha_min: float = 0.2
    crossover_alpha_max: float = 0.8
    gradient_steps: int = 1
    gradient_lr: float = 0.05
    max_gradient_norm: float = 1.0
    mutation_sigma_base: float = 0.15
    mutation_sigma_min: float = 0.03
    mutation_sigma_max: float = 0.4
    adaptive_gradient_trust: bool = True
    adaptive_mutation: bool = True
    uncertainty_kappa: float = 1.0
    scale_decay: float = 0.95
    scale_floor: float = 1e-3
    scale_max_version_lag: int = 50
    candidate_chunk_size: int = 32
    min_gain: float = 0.1
    min_gain_ratio: float = 0.0
    lcb_lambda: float = 0.0
    normalize_gradient: bool = False
    budget_seconds: float = 1.0
    seed: int = 42
    refinement_mode: str = "value_gradient"
    # 保留旧配置入口；精简版不再计算额外的精修前后审计。
    record_refinement_diagnostics: bool = True
    evaluation_backend: str = "critic"
    acceptance: str = "critic"
    search_scope: str = "episode"

    def validate(self):
        pass


class SearchTimeout(TimeoutError):
    pass


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise SearchTimeout("online search soft budget exhausted")


@contextmanager
def _evaluation(module):
    modes = [(m, m.training) for m in module.modules()]
    try:
        for m, _ in modes:
            m.training = False
        yield
    finally:
        for m, training in modes:
            m.training = training


def search_stream_seed(seed, call_index, stream, generation=-1, step=-1):
    key = f"latent-search-stream-v2:{int(seed)}:{int(call_index)}:{stream}:{int(generation)}:{int(step)}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % (2**63 - 1)


def uncertainty_controls(std, scale, *, kappa=1.0, sigma_base=0.15,
                         sigma_min=0.03, sigma_max=0.4,
                         adaptive_gradient=True, adaptive_mutation=True):
    """分歧越大，梯度信任越低，自适应变异越强。"""
    risk = std / scale
    strength = kappa * risk
    trust = 1 / (1 + strength) if adaptive_gradient else torch.ones_like(risk)
    sigma = (sigma_min + (sigma_max - sigma_min) * strength / (1 + strength)
             if adaptive_mutation and kappa > 0 else torch.full_like(risk, sigma_base))
    return risk, trust, sigma


class HybridLatentSearch:
    def __init__(self, config=None):
        self.config = config or HybridSearchConfig()
        self.reset()

    def reset(self):
        self.calls = 0
        self.last_diagnostics = {}
        self.reset_uncertainty_scale()
        self._deadline = None

    def reset_uncertainty_scale(self):
        self.ema_std2 = None
        self.scale_version = None

    def state_dict(self):
        return dict(format_version=3, rng_scheme="keyed-operators-v1",
                    value_semantics="online_actor_controller_v1", config=asdict(self.config),
                    calls=self.calls, ema_std2=self.ema_std2, scale_version=self.scale_version)

    def load_state_dict(self, state):
        self.calls = int(state["calls"])
        scale = state.get("ema_std2")
        self.ema_std2 = None if scale is None else float(scale)
        self.scale_version = state.get("scale_version")
        self.last_diagnostics = {}
        self._deadline = None

    def _rng(self, device, stream, call, generation=-1, step=-1):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device)
        return generator.manual_seed(search_stream_seed(
            self.config.seed, call, stream, generation, step))

    def update_uncertainty_scale(self, std, model_version=None):
        """维护分歧平方的 EMA，提供跨搜索调用的归一化尺度。"""
        c = self.config
        if model_version is not None and self.scale_version is not None:
            lag = model_version - self.scale_version
            if lag < 0 or lag > c.scale_max_version_lag:
                self.reset_uncertainty_scale()
        second = float(torch.as_tensor(std).detach().float().square().mean())
        self.ema_std2 = (second if self.ema_std2 is None else
                         c.scale_decay * self.ema_std2 + (1 - c.scale_decay) * second)
        self.scale_version = None if model_version is None else int(model_version)
        return max(math.sqrt(self.ema_std2), c.scale_floor)

    def _heads(self, critic, features, population):
        parts = []
        for chunk in population.split(self.config.candidate_chunk_size):
            _check_deadline(self._deadline)
            parts.append(critic.population_values(features, chunk))
            self.last_diagnostics["critic_forward_calls"] += 1
            _check_deadline(self._deadline)
        return torch.cat(parts, 2)

    def _score(self, heads):
        mean = heads.mean(0)
        lam = self.config.lcb_lambda
        return mean - lam * heads.std(0, unbiased=False) if lam > 0 else mean

    def _controls(self, std, scale):
        c = self.config
        return uncertainty_controls(
            std, scale, kappa=c.uncertainty_kappa,
            sigma_base=c.mutation_sigma_base, sigma_min=c.mutation_sigma_min,
            sigma_max=c.mutation_sigma_max, adaptive_gradient=c.adaptive_gradient_trust,
            adaptive_mutation=c.adaptive_mutation)

    def _initial_population(self, z, due, archive, generator):
        c = self.config
        bank = z[due].detach()
        if archive is not None:
            bank = torch.cat((bank, torch.as_tensor(archive, device=z.device, dtype=z.dtype)), 0)
        bank = torch.unique(bank, dim=0)
        keep = min(bank.size(0), max(1, c.candidates // 3))
        if bank.size(0) > keep:
            bank = bank[torch.randperm(bank.size(0), device=z.device, generator=generator)[:keep]]
        missing = c.candidates - bank.size(0)
        if missing:
            random = (2 * torch.rand(missing, z.size(1), device=z.device,
                                     dtype=z.dtype, generator=generator) - 1) * c.latent_bound
            bank = torch.cat((bank, random), 0)
        return bank.detach().clone()

    def _refine_children(self, critic, features, children, scale, call, generation):
        """交叉后沿集成均值梯度精修；支持等长随机方向消融。"""
        c = self.config
        x = children.detach()
        norms = []
        steps = c.gradient_steps if c.gradient_lr and c.refinement_mode != "none" else 0
        for step in range(steps):
            _check_deadline(self._deadline)
            with torch.enable_grad():
                leaf = x.detach().requires_grad_(True)
                heads = self._heads(critic, features, leaf)
                objective = heads.mean(0).mean(0).sum()
                gradient, = torch.autograd.grad(objective, leaf)
            norm = gradient.norm(dim=-1, keepdim=True)
            _, trust, _ = self._controls(heads.detach().std(0, unbiased=False).mean(0), scale)
            direction = (gradient / norm.clamp_min(1e-12) if c.normalize_gradient else
                         gradient * (c.max_gradient_norm / norm.clamp_min(1e-12)).clamp(max=1))
            delta = c.gradient_lr * trust[:, None] * direction
            if c.refinement_mode != "value_gradient":
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                    generator=self._rng(x.device, "random_direction", call, generation, step))
                delta = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12) * delta.norm(dim=-1, keepdim=True)
            else:
                self.last_diagnostics["gradient_applied_steps"] += 1
            x = (leaf.detach() + delta).clamp(-c.latent_bound, c.latent_bound).detach()
            norms.append(float(norm.mean()))
            self.last_diagnostics["gradient_evaluations"] += 1
            self.last_diagnostics["refinement_applied_steps"] += 1
            _check_deadline(self._deadline)
        return x, sum(norms) / len(norms) if norms else 0.0

    @torch.no_grad()
    def choose(self, critic, state, z, due=None, *, initial_population=None,
               model_version=None, deadline=None):
        """共享种群进化、逐行接受；软超时则全部保留原 z。"""
        c = self.config
        started = perf_counter()
        B, D = z.shape
        original = z.detach().clone()
        due = (torch.ones(B, dtype=torch.bool, device=z.device) if due is None else
               torch.as_tensor(due, dtype=torch.bool, device=z.device).clone())
        switched = torch.zeros_like(due)
        d = self.last_diagnostics = dict(
            search_scope="episode", evaluation_backend="critic", acceptance="critic",
            refinement_mode=c.refinement_mode, due_rows=int(due.sum()), switched_rows=0,
            proposed_rows=0, timed_out=False, calls=self.calls, model_version=model_version,
            predicted_gain_mean=0.0, accepted_predicted_gain_mean=0.0, selection_gain_mean=0.0,
            critic_forward_calls=0, population_evaluations=0, gradient_evaluations=0,
            gradient_applied_steps=0, refinement_applied_steps=0,
            generations_completed=0, generation_stats=[])
        if not bool(due.any()):
            d["elapsed_seconds"] = perf_counter() - started
            return original, switched
        limits = [limit for limit in (deadline, started + c.budget_seconds if c.budget_seconds else None)
                  if limit is not None]
        self._deadline = min(limits) if limits else None
        call = self.calls
        self.calls += 1
        d["calls"] = self.calls
        with _evaluation(critic):
            try:
                _check_deadline(self._deadline)
                features = critic.encode_state(state).detach()[due]
                incumbent = original[due]
                base_heads = critic.evaluate_features(features, incumbent)
                d["critic_forward_calls"] += 1
                _check_deadline(self._deadline)
                std = base_heads.std(0, unbiased=False)
                need_scale = (c.adaptive_gradient_trust or c.adaptive_mutation or
                              c.min_gain_ratio > 0 or c.record_refinement_diagnostics or
                              (c.refinement_mode != "none" and c.gradient_steps > 0 and c.gradient_lr > 0))
                scale = self.update_uncertainty_scale(std, model_version) if need_scale else None
                d.update(ensemble_heads=int(critic.n_critics), normalization_scale=scale,
                         mean_incumbent_disagreement=float(std.mean()))
                population = self._initial_population(
                    original, due, initial_population, self._rng(z.device, "initial", call))
                elites = max(1, math.ceil(c.candidates * c.elite_fraction))
                immigrants = min(c.candidates - elites, round(c.candidates * c.immigrant_fraction))
                offspring = c.candidates - elites - immigrants
                d.update(elite_count=elites, offspring_per_generation=offspring,
                         immigrants_per_generation=immigrants)

                # 精英保留 -> 线性交叉 -> 梯度精修 -> 自适应变异 -> 随机移民。
                for generation in range(c.generations):
                    _check_deadline(self._deadline)
                    fitness = self._score(self._heads(critic, features, population)).mean(0)
                    d["population_evaluations"] += 1
                    elite = population[fitness.topk(elites).indices]
                    parts = [elite]
                    stats = dict(generation=generation + 1, best_selection_fitness=float(fitness.max()))
                    if offspring:
                        parents = torch.randint(elites, (2, offspring), device=z.device,
                                                generator=self._rng(z.device, "parents", call, generation))
                        alpha = torch.empty(offspring, 1, device=z.device, dtype=z.dtype).uniform_(
                            c.crossover_alpha_min, c.crossover_alpha_max,
                            generator=self._rng(z.device, "crossover", call, generation))
                        children = alpha * elite[parents[0]] + (1 - alpha) * elite[parents[1]]
                        children, norm = self._refine_children(
                            critic, features, children, scale, call, generation)
                        if c.adaptive_mutation:
                            uncertainty = self._heads(critic, features, children).std(0, unbiased=False).mean(0)
                            _, _, sigma = self._controls(uncertainty, scale)
                        else:
                            sigma = torch.full((offspring,), c.mutation_sigma_base,
                                               device=z.device, dtype=z.dtype)
                        noise = torch.randn(children.shape, device=z.device, dtype=z.dtype,
                                            generator=self._rng(z.device, "mutation", call, generation))
                        parts.append((children + sigma[:, None] * noise).clamp(
                            -c.latent_bound, c.latent_bound).detach())
                        stats.update(mean_gradient_norm=norm, mutation_sigma={"mean": float(sigma.mean())})
                    if immigrants:
                        parts.append((2 * torch.rand(immigrants, D, device=z.device, dtype=z.dtype,
                                                     generator=self._rng(z.device, "immigrants", call, generation))
                                      - 1) * c.latent_bound)
                    population = torch.cat(parts, 0).detach()
                    d["generation_stats"].append(stats)
                    d["generations_completed"] += 1

                # 种群共享，但各环境独立选择候选并与原 z 比较。
                scores = self._score(self._heads(critic, features, population))
                d["population_evaluations"] += 1
                best, indices = scores.max(1)
                proposal = population[indices]
                gain = best - self._score(base_heads)
                threshold = c.min_gain + c.min_gain_ratio * (scale if scale is not None else 0)
                accepted = (gain > threshold) & (proposal != incumbent).any(1)
                updated = original.clone()
                updated[due] = torch.where(accepted[:, None], proposal, incumbent)
                switched[due] = accepted
                _check_deadline(self._deadline)
                d.update(switched_rows=int(switched.sum()), proposed_rows=int(accepted.sum()),
                         predicted_gain_mean=float(gain.mean()), selection_gain_mean=float(gain.mean()),
                         accepted_predicted_gain_mean=float(gain[accepted].mean()) if bool(accepted.any()) else 0.0,
                         acceptance_threshold=float(threshold), elapsed_seconds=perf_counter() - started)
                return updated.detach().clone(), switched.detach().clone()
            except TimeoutError:
                d.update(timed_out=True, switched_rows=0, accepted_predicted_gain_mean=0.0,
                         elapsed_seconds=perf_counter() - started)
                return original, torch.zeros_like(due)
            finally:
                self._deadline = None


__all__ = ["HybridSearchConfig", "HybridLatentSearch", "SearchTimeout",
           "uncertainty_controls", "search_stream_seed"]
