"""Episode-internal current-state Critic-guided shared latent evolution.

Only current state is evaluated; no representatives, Actor rollout, callback
backend or winner confirmation. Call after env.begin_online_search(), before
the real action. On soft timeout keep every original latent.

Actor/Critic parameters are never optimized here. Critic features are encoded
once, then only due-row features are used for candidate heads and z gradients.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import math
from numbers import Integral, Real
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
    record_refinement_diagnostics: bool = True
    evaluation_backend: str = "critic"
    acceptance: str = "critic"
    search_scope: str = "episode"

    def validate(self):
        for key in ("candidates", "generations", "scale_max_version_lag", "candidate_chunk_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"invalid {key}")
        for key in ("seed", "gradient_steps"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"invalid {key}")
        for key in (
            "adaptive_gradient_trust",
            "adaptive_mutation",
            "record_refinement_diagnostics",
            "normalize_gradient",
        ):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"invalid {key}")
        keys = (
            "latent_bound",
            "elite_fraction",
            "immigrant_fraction",
            "crossover_alpha_min",
            "crossover_alpha_max",
            "gradient_lr",
            "max_gradient_norm",
            "mutation_sigma_base",
            "mutation_sigma_min",
            "mutation_sigma_max",
            "uncertainty_kappa",
            "scale_decay",
            "scale_floor",
            "min_gain",
            "budget_seconds",
            "min_gain_ratio",
            "lcb_lambda",
        )
        for key in keys:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"invalid {key}")
        if min(self.latent_bound, self.max_gradient_norm, self.scale_floor) <= 0:
            raise ValueError("invalid bounds/scales")
        if not 0 < self.elite_fraction <= 1 or not 0 <= self.immigrant_fraction < 1:
            raise ValueError("invalid fractions")
        if not 0 <= self.crossover_alpha_min <= self.crossover_alpha_max <= 1:
            raise ValueError("invalid crossover")
        if not 0 < self.mutation_sigma_min <= self.mutation_sigma_base <= self.mutation_sigma_max:
            raise ValueError("invalid mutation scales")
        if not 0 <= self.scale_decay < 1:
            raise ValueError("invalid scale decay")
        if (
            min(
                self.gradient_lr,
                self.uncertainty_kappa,
                self.min_gain,
                self.budget_seconds,
                self.min_gain_ratio,
                self.lcb_lambda,
            )
            < 0
        ):
            raise ValueError("negative search scale")
        if self.refinement_mode not in ("none", "value_gradient", "random_matched"):
            raise ValueError("invalid refinement_mode")
        if (self.search_scope, self.evaluation_backend, self.acceptance) != ("episode", "critic", "critic"):
            raise ValueError("this phase supports only episode/current-state Critic scoring and acceptance")


class SearchTimeout(TimeoutError):
    pass


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise SearchTimeout("online search soft budget exhausted")


@contextmanager
def _evaluation(module):
    saved = [(m, m.training) for m in module.modules()]
    try:
        for m, _ in saved:
            m.training = False
        yield
    finally:
        for m, flag in saved:
            m.training = flag


def _summary(x):
    x = x.detach().float()
    return dict(min=float(x.min()), mean=float(x.mean()), max=float(x.max()), std=float(x.std(unbiased=False)))


def search_stream_seed(seed, call_index, stream, generation=-1, step=-1):
    payload = f"latent-search-stream-v2:{int(seed)}:{int(call_index)}:{stream}:{int(generation)}:{int(step)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def uncertainty_controls(
    std,
    scale,
    *,
    kappa=1.0,
    sigma_base=0.15,
    sigma_min=0.03,
    sigma_max=0.4,
    adaptive_gradient=True,
    adaptive_mutation=True,
):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid scale")
    risk = std / scale
    strength = kappa * risk
    trust = 1.0 / (1.0 + strength) if adaptive_gradient else risk * 0.0 + 1.0
    sigma = (
        sigma_min + (sigma_max - sigma_min) * strength / (1.0 + strength)
        if adaptive_mutation and kappa > 0
        else risk * 0.0 + sigma_base
    )
    return risk, trust, sigma


class HybridLatentSearch:
    def __init__(self, config=None):
        self.config = config or HybridSearchConfig()
        self.config.validate()
        self.calls = 0
        self.last_diagnostics = {}
        self.ema_std2 = None
        self.scale_version = None
        self._deadline = None

    def reset(self):
        self.calls = 0
        self.last_diagnostics = {}
        self.reset_uncertainty_scale()
        self._deadline = None

    def reset_uncertainty_scale(self):
        self.ema_std2 = None
        self.scale_version = None

    def _rng(self, device, stream, call, generation=-1, step=-1):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=device)
        g.manual_seed(search_stream_seed(self.config.seed, call, stream, generation, step))
        return g

    def state_dict(self):
        return dict(
            format_version=3,
            rng_scheme="keyed-operators-v1",
            value_semantics="online_actor_controller_v1",
            config=asdict(self.config),
            calls=self.calls,
            ema_std2=self.ema_std2,
            scale_version=self.scale_version,
        )

    def load_state_dict(self, state):
        if (
            state.get("format_version") != 3
            or state.get("rng_scheme") != "keyed-operators-v1"
            or state.get("value_semantics") != "online_actor_controller_v1"
            or state.get("config") != asdict(self.config)
        ):
            raise ValueError("requires online search state v3 with identical configuration")
        calls = state["calls"]
        version = state.get("scale_version")
        scale = state.get("ema_std2")
        if isinstance(calls, bool) or not isinstance(calls, Integral) or calls < 0:
            raise ValueError("invalid calls")
        if version is not None and (isinstance(version, bool) or not isinstance(version, Integral) or version < 0):
            raise ValueError("invalid version")
        if scale is not None and (not math.isfinite(float(scale)) or float(scale) < 0):
            raise ValueError("invalid EMA")
        self.calls = int(calls)
        self.ema_std2 = None if scale is None else float(scale)
        self.scale_version = version
        self.last_diagnostics = {}
        self._deadline = None

    def update_uncertainty_scale(self, std, model_version=None):
        x = torch.as_tensor(std).detach().float()
        if x.numel() == 0 or not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
            raise ValueError("invalid disagreement")
        if model_version is not None:
            if isinstance(model_version, bool) or not isinstance(model_version, Integral) or model_version < 0:
                raise ValueError("invalid model_version")
            if self.scale_version is not None and (
                model_version < self.scale_version
                or model_version - self.scale_version > self.config.scale_max_version_lag
            ):
                self.reset_uncertainty_scale()
        second = float(x.square().mean())
        if not math.isfinite(second):
            raise ValueError("disagreement overflow")
        self.ema_std2 = (
            second
            if self.ema_std2 is None
            else self.config.scale_decay * self.ema_std2 + (1 - self.config.scale_decay) * second
        )
        self.scale_version = None if model_version is None else int(model_version)
        return max(math.sqrt(self.ema_std2), self.config.scale_floor)

    def _heads(self, critic, features, population):
        parts = []
        M = int(critic.n_critics)
        S = features.size(0)
        for candidate in population.split(self.config.candidate_chunk_size):
            _check_deadline(self._deadline)
            if callable(getattr(critic, "population_values", None)):
                out = critic.population_values(features, candidate)
            else:
                K = candidate.size(0)
                f = features[:, None, :].expand(S, K, -1).reshape(S * K, -1)
                z = candidate[None, :, :].expand(S, K, -1).reshape(S * K, -1)
                out = critic.evaluate_features(f, z).reshape(M, S, K)
            self.last_diagnostics["critic_forward_calls"] += 1
            if out.shape != (M, S, candidate.size(0)) or not bool(torch.isfinite(out).all()):
                raise ValueError("invalid population values")
            parts.append(out)
            _check_deadline(self._deadline)
        return torch.cat(parts, 2)

    def _individual_heads(self, critic, features, z):
        _check_deadline(self._deadline)
        out = critic.evaluate_features(features, z)
        self.last_diagnostics["critic_forward_calls"] += 1
        if out.shape != (int(critic.n_critics), features.size(0)) or not bool(torch.isfinite(out).all()):
            raise ValueError("invalid individual values")
        _check_deadline(self._deadline)
        return out

    def _conservative(self, heads):
        mean = heads.mean(0)
        lam = self.config.lcb_lambda
        return mean - lam * heads.std(0, unbiased=False) if lam > 0 else mean

    def _score(self, critic, features, population):
        self.last_diagnostics["population_evaluations"] += 1
        return self._conservative(self._heads(critic, features, population))

    def _initial_population(self, z, due, archive, generator):
        c = self.config
        bank = z[due].detach()
        if archive is not None:
            archive = torch.as_tensor(archive, device=z.device, dtype=z.dtype)
            if archive.ndim != 2 or archive.shape[-1] != z.size(1) or not archive.numel():
                raise ValueError("initial_population must be nonempty [K,D]")
            if not bool(torch.isfinite(archive).all()) or bool((archive.abs() > c.latent_bound).any()):
                raise ValueError("invalid archive values")
            bank = torch.cat((bank, archive), 0)
        bank = torch.unique(bank, dim=0)
        keep = min(bank.size(0), max(1, c.candidates // 3))
        if bank.size(0) > keep:
            bank = bank[torch.randperm(bank.size(0), device=z.device, generator=generator)[:keep]]
        missing = c.candidates - bank.size(0)
        if missing:
            bank = torch.cat(
                (
                    bank,
                    (2 * torch.rand(missing, z.size(1), device=z.device, dtype=z.dtype, generator=generator) - 1)
                    * c.latent_bound,
                ),
                0,
            )
        return bank.detach().clone()

    def _controls(self, std, scale):
        c = self.config
        return uncertainty_controls(
            std,
            scale,
            kappa=c.uncertainty_kappa,
            sigma_base=c.mutation_sigma_base,
            sigma_min=c.mutation_sigma_min,
            sigma_max=c.mutation_sigma_max,
            adaptive_gradient=c.adaptive_gradient_trust,
            adaptive_mutation=c.adaptive_mutation,
        )

    def _refine_children(self, critic, features, children, joint_due, scale, call, generation, deadline):
        c = self.config
        start = perf_counter()
        origin = children.detach().clone()
        x = origin.clone()
        stats = dict(
            refinement_mode=c.refinement_mode,
            mean_gradient_norm=0.0,
            refinement_step_norm_mean=0.0,
            random_reference_step_norm_mean=0.0,
            refinement_actual_displacement_mean=0.0,
            refinement_boundary_fraction=0.0,
            gradient_direction_cosine_mean=None,
            refinement_predicted_gain_mean=None,
        )
        norms = []
        lengths = []
        references = []
        cosines = []
        before = None
        if c.record_refinement_diagnostics:
            before = self._heads(critic, features, x).mean(0)[joint_due].mean(0)
        steps = c.gradient_steps if c.gradient_lr and c.refinement_mode != "none" else 0
        for step in range(steps):
            _check_deadline(deadline)
            with torch.enable_grad():
                leaf = x.detach().requires_grad_(True)
                heads = self._heads(critic, features, leaf)
                objective = heads.mean(0)[joint_due].mean(0).sum()
                if not objective.requires_grad:
                    raise RuntimeError("Critic detached latent gradient")
                (gradient,) = torch.autograd.grad(objective, leaf, allow_unused=True)
            self.last_diagnostics["gradient_evaluations"] += 1
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise ValueError("invalid gradient")
            norm = gradient.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(norm).all()):
                raise ValueError("gradient overflow")
            _, trust, _ = self._controls(heads.detach().std(0, unbiased=False)[joint_due].mean(0), scale)
            clipped = (
                gradient / norm.clamp_min(1e-12)
                if c.normalize_gradient
                else gradient * (c.max_gradient_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
            )
            reference = c.gradient_lr * trust[:, None] * clipped
            if c.refinement_mode == "value_gradient":
                delta = reference
                self.last_diagnostics["gradient_applied_steps"] += 1
            else:
                noise = torch.randn(
                    x.shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=self._rng(x.device, "random_direction", call, generation, step),
                )
                delta = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12) * reference.norm(dim=-1, keepdim=True)
            x = (leaf.detach() + delta).clamp(-c.latent_bound, c.latent_bound).detach()
            norms.append(float(norm.mean()))
            lengths.append(float(delta.norm(dim=-1).mean()))
            references.append(float(reference.norm(dim=-1).mean()))
            valid = (norm.squeeze(-1) > 1e-12) & (delta.norm(dim=-1) > 1e-12)
            if bool(valid.any()):
                cosines.append(
                    float(torch.nn.functional.cosine_similarity(delta[valid], gradient[valid], dim=-1).mean())
                )
            self.last_diagnostics["refinement_applied_steps"] += 1
            _check_deadline(deadline)
        stats.update(
            mean_gradient_norm=sum(norms) / len(norms) if norms else 0.0,
            refinement_step_norm_mean=sum(lengths) / len(lengths) if lengths else 0.0,
            random_reference_step_norm_mean=sum(references) / len(references) if references else 0.0,
            refinement_actual_displacement_mean=float((x - origin).norm(dim=-1).mean()),
            refinement_boundary_fraction=float((x.abs() >= c.latent_bound - 1e-6).float().mean()),
            gradient_direction_cosine_mean=sum(cosines) / len(cosines) if cosines else None,
        )
        if before is not None:
            stats["refinement_predicted_gain_mean"] = float(
                (self._heads(critic, features, x).mean(0)[joint_due].mean(0) - before).mean()
            )
        self.last_diagnostics["refinement_seconds"] += perf_counter() - start
        return x, stats

    @torch.no_grad()
    def choose(self, critic, state, z, due=None, *, initial_population=None, model_version=None, deadline=None):
        c = self.config
        c.validate()
        started = perf_counter()
        if z.ndim != 2 or not z.is_floating_point() or min(z.shape) < 1:
            raise ValueError("z must be floating [B,D]")
        if not bool(torch.isfinite(z).all()) or bool((z.abs() > c.latent_bound).any()):
            raise ValueError("invalid z")
        B, D = z.shape
        due = (
            torch.ones(B, dtype=torch.bool, device=z.device)
            if due is None
            else torch.as_tensor(due, dtype=torch.bool, device=z.device).clone()
        )
        if due.shape != (B,):
            raise ValueError("due must be [B]")
        original = z.detach().clone()
        switched = torch.zeros_like(due)
        self.last_diagnostics = dict(
            search_scope="episode",
            evaluation_backend="critic",
            acceptance="critic",
            refinement_mode=c.refinement_mode,
            due_rows=int(due.sum()),
            switched_rows=0,
            proposed_rows=0,
            timed_out=False,
            calls=self.calls,
            model_version=model_version,
            predicted_gain_mean=0.0,
            accepted_predicted_gain_mean=0.0,
            selection_gain_mean=0.0,
            critic_forward_calls=0,
            population_evaluations=0,
            gradient_evaluations=0,
            gradient_applied_steps=0,
            refinement_applied_steps=0,
            refinement_seconds=0.0,
            generations_completed=0,
            generation_stats=[],
            rollout_episodes=0,
            rollout_env_steps=0,
            rollout_batch_steps=0,
            rollout_counts_complete=True,
        )
        if not bool(due.any()):
            self.last_diagnostics["elapsed_seconds"] = perf_counter() - started
            return original, switched
        if (
            getattr(critic, "value_semantics", None) != "online_actor_controller_v1"
            or getattr(critic, "online_context_dim", None) != 6
        ):
            raise ValueError("install the online-context Critic first")
        M = getattr(critic, "n_critics", 0)
        if isinstance(M, bool) or not isinstance(M, Integral) or M < 2:
            raise ValueError("requires ensemble M>=2")
        online = getattr(state, "online_context", None)
        active = getattr(state, "active_mask", None)
        if not torch.is_tensor(online) or online.shape != (B, 6) or not bool(torch.isfinite(online).all()):
            raise ValueError("state needs online_context[B,6]")
        if not torch.is_tensor(active) or active.shape != (B,) or active.dtype != torch.bool:
            raise ValueError("state needs active_mask[B]")
        if active.device != z.device or online.device != z.device:
            raise ValueError("state/z device mismatch")
        if bool((due & ~active).any()) or bool((online[due, 5] < 0.5).any()):
            raise ValueError("due includes inactive or disabled rows")
        if getattr(state, "online_search_pending", False) is not True:
            raise ValueError("call begin_online_search and refresh current state before choose")
        if model_version is not None and (
            isinstance(model_version, bool) or not isinstance(model_version, Integral) or model_version < 0
        ):
            raise ValueError("invalid model_version")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("invalid deadline")
        limits = [v for v in (deadline, started + c.budget_seconds if c.budget_seconds else None) if v is not None]
        deadline = min(limits) if limits else None
        if c.refinement_mode != "none" and c.gradient_steps and c.gradient_lr and torch.is_inference_mode_enabled():
            raise RuntimeError("use no_grad, not inference_mode, around latent gradients")
        call = self.calls
        self.calls += 1
        self.last_diagnostics["calls"] = self.calls
        self._deadline = deadline
        with _evaluation(critic):
            try:
                _check_deadline(deadline)
                encoding = perf_counter()
                features = critic.encode_state(state).detach()
                if (
                    features.ndim != 2
                    or features.size(0) != B
                    or features.device != z.device
                    or not bool(torch.isfinite(features).all())
                ):
                    raise ValueError("invalid current state encoding")
                features = features[due]
                self.last_diagnostics["state_encoding_seconds"] = perf_counter() - encoding
                incumbent = original[due]
                base_heads = self._individual_heads(critic, features, incumbent)
                baseline = base_heads.mean(0)
                need_std = (
                    c.adaptive_gradient_trust
                    or c.adaptive_mutation
                    or c.record_refinement_diagnostics
                    or (c.refinement_mode != "none" and c.gradient_steps > 0 and c.gradient_lr > 0)
                    or c.min_gain_ratio > 0
                )
                scale = None
                if need_std:
                    std = base_heads.std(0, unbiased=False)
                    scale = self.update_uncertainty_scale(std, model_version)
                row_mask = torch.ones(features.size(0), dtype=torch.bool, device=z.device)
                population = self._initial_population(
                    original, due, initial_population, self._rng(z.device, "initial", call)
                )
                elites = max(1, math.ceil(c.candidates * c.elite_fraction))
                free = c.candidates - elites
                immigrants = min(free, round(c.candidates * c.immigrant_fraction))
                offspring = free - immigrants
                d = self.last_diagnostics
                d.update(
                    ensemble_heads=M,
                    normalization_scale=scale,
                    mean_incumbent_disagreement=float(std.mean()) if need_std else None,
                    elite_count=elites,
                    offspring_per_generation=offspring,
                    immigrants_per_generation=immigrants,
                )
                for generation in range(c.generations):
                    _check_deadline(deadline)
                    values = self._score(critic, features, population)
                    fitness = values.mean(0)
                    elite = population[fitness.topk(elites).indices]
                    parts = [elite]
                    stats = dict(
                        generation=generation + 1,
                        best_selection_fitness=float(fitness.max()),
                        best_predicted_fitness=float(fitness.max()),
                    )
                    if offspring:
                        parent = torch.randint(
                            elites,
                            (2, offspring),
                            device=z.device,
                            generator=self._rng(z.device, "parents", call, generation),
                        )
                        alpha = torch.empty(offspring, 1, device=z.device, dtype=z.dtype).uniform_(
                            c.crossover_alpha_min,
                            c.crossover_alpha_max,
                            generator=self._rng(z.device, "crossover", call, generation),
                        )
                        children = alpha * elite[parent[0]] + (1 - alpha) * elite[parent[1]]
                        children, ref = self._refine_children(
                            critic, features, children, row_mask, scale, call, generation, deadline
                        )
                        stats.update(ref)
                        if need_std:
                            u = self._heads(critic, features, children).std(0, unbiased=False).mean(0)
                            risk, trust, sigma = self._controls(u, scale)
                            stats.update(risk=_summary(risk), gradient_trust=_summary(trust))
                        else:
                            sigma = torch.full((offspring,), c.mutation_sigma_base, device=z.device, dtype=z.dtype)
                        noise = torch.randn(
                            children.shape,
                            device=z.device,
                            dtype=z.dtype,
                            generator=self._rng(z.device, "mutation", call, generation),
                        )
                        mutated = (children + sigma[:, None] * noise).clamp(-c.latent_bound, c.latent_bound)
                        parts.append(mutated.detach())
                        stats.update(
                            mutation_sigma=_summary(sigma),
                            boundary_fraction=float((mutated.abs() >= c.latent_bound - 1e-6).float().mean()),
                        )
                    if immigrants:
                        parts.append(
                            (
                                2
                                * torch.rand(
                                    immigrants,
                                    D,
                                    device=z.device,
                                    dtype=z.dtype,
                                    generator=self._rng(z.device, "immigrants", call, generation),
                                )
                                - 1
                            )
                            * c.latent_bound
                        )
                    population = torch.cat(parts, 0).detach()
                    d["generation_stats"].append(stats)
                    d["generations_completed"] += 1
                final = self._score(critic, features, population)
                best, index = final.max(1)
                lam = c.lcb_lambda
                base_score = baseline - lam * base_heads.std(0, unbiased=False) if lam > 0 else baseline
                proposal = population[index]
                gain = best - base_score
                threshold = c.min_gain + c.min_gain_ratio * (scale if scale is not None else 0.0)
                accepted = (gain > threshold) & (proposal != incumbent).any(1)
                selected = torch.where(accepted[:, None], proposal, incumbent)
                updated = original.clone()
                updated[due] = selected
                switched[due] = accepted
                d.update(
                    proposed_rows=int(accepted.sum()),
                    predicted_gain_mean=float(gain.mean()),
                    selection_gain_mean=float(gain.mean()),
                    population_diversity=float(population.float().std(0, unbiased=False).mean()),
                    acceptance_threshold=float(threshold),
                )
                _check_deadline(deadline)
                d.update(
                    switched_rows=int(switched.sum()),
                    accepted_predicted_gain_mean=float(gain[accepted].mean()) if bool(accepted.any()) else 0.0,
                    elapsed_seconds=perf_counter() - started,
                )
                return updated.detach().clone(), switched.detach().clone()
            except TimeoutError:
                self.last_diagnostics.update(
                    timed_out=True,
                    switched_rows=0,
                    accepted_predicted_gain_mean=0.0,
                    elapsed_seconds=perf_counter() - started,
                )
                return original, torch.zeros_like(due)
            finally:
                self._deadline = None


__all__ = ["HybridSearchConfig", "HybridLatentSearch", "SearchTimeout", "uncertainty_controls", "search_stream_seed"]
