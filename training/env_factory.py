"""Grid training/deployment task factory; traffic engine is unchanged."""
from CALSESD_internal.env import CVRPGridEnvForLatent,BatchCVRPEnv
import importlib
import math
from numbers import Integral, Real

def make_env(cfg,seed,device):
    base=CVRPGridEnvForLatent(grid_size=cfg.grid_size,vehicle_capacity=cfg.vehicle_capacity,
        num_customers=cfg.num_customers,seed=int(seed),device=device,demand_noise_std=cfg.demand_noise_std,
        traffic_noise_std=cfg.traffic_noise_std,traffic_trend=cfg.traffic_trend,
        traffic_beta=cfg.traffic_beta,max_steps=cfg.max_episode_steps)
    env=BatchCVRPEnv(base,cfg.batch_size,valid_reward=0.,step_penalty=0.,max_steps=cfg.max_episode_steps)
    env.failure_penalty=cfg.failure_penalty
    env.route_interval=1;env.route_at_reset=False
    env.route_enabled=False
    env.reset()
    return env

def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _resolve_hook(spec, name):
    if spec is None or spec == '':
        return None
    if callable(spec):
        return spec
    if not isinstance(spec, str) or ':' not in spec:
        raise ValueError(f'{name} must be callable or module:function')
    module, _, attribute = spec.strip().partition(':')
    if not module or not attribute:
        raise ValueError(f'invalid {name}')
    obj = importlib.import_module(module)
    for part in attribute.split('.'):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f'{name} is not callable')
    return obj


def _hook_identifier(hook):
    if hook is None:
        return None
    return f"{getattr(hook, '__module__', type(hook).__module__)}:{getattr(hook, '__qualname__', type(hook).__qualname__)}"


class RepresentativePool:
    """Reproducible representative tasks; build() keeps its original contract."""

    def __init__(self, cfg):
        self.cfg = cfg
        start = _integer('representative_seed', cfg.representative_seed)
        count = _integer('representative_instances', cfg.representative_instances, 1)
        self.seeds = [start+j for j in range(count)]
        self.calls = 0

    def _make_envs(self, device):
        return [make_env(self.cfg, seed, device) for seed in self.seeds]

    def build(self, device):
        """Return states, NOT environments; used by existing Critic search."""
        from .rng_state import preserve_rng
        with preserve_rng():
            states = [env.get_batch_state() for env in self._make_envs(device)]
        self.calls += 1
        return states

    def build_envs(self, device):
        """Return independent representative envs, never the live training env.

        Uses the same make_env construction/reset behavior as build().
        The full-rollout evaluator owns subsequent cloning of these tasks.
        """
        from .rng_state import preserve_rng
        with preserve_rng():
            envs = self._make_envs(device)
        self.calls += 1
        return envs

    def state_dict(self):
        return dict(seeds=list(self.seeds), calls=self.calls)

    def load_state_dict(self, state):
        calls = _integer('representative calls', state['calls'])
        if list(state['seeds']) != self.seeds:
            raise ValueError('Representative pool configuration/state mismatch')
        self.calls = calls


class RolloutTaskProvider:
    """Staged training bridge for exp2/rollout.

    __call__(device=..., call_index=..., model_version=...) -> SearchTaskBundle.
    Uses current Actor reference without freezing/updating any model weights.
    A new evaluator/geometry cache is built for every search trigger.
    """

    _TASK_FIELDS = (
        'seed', 'grid_size', 'num_customers', 'vehicle_capacity', 'batch_size',
        'max_episode_steps', 'demand_noise_std', 'traffic_noise_std',
        'traffic_trend', 'traffic_beta', 'failure_penalty', 'gamma',
        'latent_dim', 'latent_bound', 'representative_seed', 'representative_instances',
    )

    def __init__(self, cfg, policy, *, clone_env_fn=None,
                 scenario_prepare_fn=None, evaluator_cls=None):
        self.cfg = cfg
        self.policy = policy
        self.pool = RepresentativePool(cfg)
        self.calls = 0
        self.last_call_index = None
        self.last_model_version = None
        self.scenario_count = _integer('rollout_scenarios', getattr(cfg, 'rollout_scenarios', 1), 1)
        self.decoder = getattr(cfg, 'rollout_decoder', 'sample')
        self.scenario_mode = getattr(cfg, 'rollout_scenario_mode', 'snapshot_replay_reference')
        self.seed = _integer('rollout_seed', getattr(cfg, 'rollout_seed', cfg.seed+2000))
        self.max_steps = _integer('max_episode_steps', cfg.max_episode_steps, 1)
        if isinstance(cfg.gamma, bool) or not isinstance(cfg.gamma, Real) or not math.isfinite(cfg.gamma) or not 0 < cfg.gamma <= 1:
            raise ValueError('gamma must be finite and in (0,1]')
        self.gamma = float(cfg.gamma)
        if self.decoder not in ('sample', 'greedy'):
            raise ValueError('rollout_decoder must be sample/greedy')
        if self.scenario_mode not in ('snapshot_replay_reference', 'resampled_future'):
            raise ValueError('invalid rollout_scenario_mode')
        clone_spec = clone_env_fn if clone_env_fn is not None else getattr(cfg, 'rollout_clone_env_fn', '')
        prepare_spec = scenario_prepare_fn if scenario_prepare_fn is not None else getattr(cfg, 'rollout_scenario_preparer', '')
        self.clone_env_fn = _resolve_hook(clone_spec, 'rollout_clone_env_fn')
        self.scenario_prepare_fn = _resolve_hook(prepare_spec, 'rollout_scenario_preparer')
        if self.scenario_mode == 'resampled_future' and self.scenario_prepare_fn is None:
            raise ValueError('resampled_future requires an observation-preserving future sampler')
        if self.scenario_count > 1 and self.scenario_prepare_fn is None:
            raise ValueError('multiple future scenarios require a future-sampling hook, not global reseeding alone')
        if evaluator_cls is None:
            try:
                from CALSESD.experiments.rollout_evaluator import RolloutPopulationEvaluator
            except ImportError as exc:
                raise RuntimeError('Cannot import the experiment-2 rollout evaluator; install its module and dependencies first') from exc
            evaluator_cls = RolloutPopulationEvaluator
        if not callable(evaluator_cls):
            raise TypeError('evaluator_cls must be callable')
        self.evaluator_cls = evaluator_cls

    def _fingerprint(self):
        return dict(task_config={name: getattr(self.cfg, name, None) for name in self._TASK_FIELDS},
                    seeds=list(self.pool.seeds), gamma=self.gamma, max_steps=self.max_steps,
                    scenario_count=self.scenario_count, decoder=self.decoder,
                    scenario_mode=self.scenario_mode, seed=self.seed,
                    clone_hook=_hook_identifier(self.clone_env_fn),
                    scenario_hook=_hook_identifier(self.scenario_prepare_fn))

    def __call__(self, *, device, call_index, model_version=None):
        from .rng_state import preserve_rng
        from .ablation_config import SearchTaskBundle, validate_task_bundle
        call_index = _integer('call_index', call_index)
        if model_version is not None:
            model_version = _integer('model_version', model_version)
        old_pool_calls = self.pool.calls
        try:
            with preserve_rng():
                envs = self.pool.build_envs(device)
                evaluator = self.evaluator_cls(
                    self.policy, envs, gamma=self.gamma, max_steps=self.max_steps,
                    scenario_count=self.scenario_count, decoder=self.decoder,
                    seed=self.seed, scenario_mode=self.scenario_mode,
                    scenario_prepare_fn=self.scenario_prepare_fn,
                    clone_env_fn=self.clone_env_fn)
                bundle = SearchTaskBundle(
                    states=tuple(evaluator.states), population_evaluator=evaluator,
                    metadata=dict(representative_seeds=list(self.pool.seeds),
                        task_source='independent_representative_instances',
                        evaluation_backend='rollout', decoder=self.decoder,
                        scenario_mode=self.scenario_mode, scenarios=self.scenario_count,
                        rollout_seed=self.seed, gamma=self.gamma,
                        score_objective='discounted_environment_return',
                        call_index=call_index, model_version=model_version,
                        provider_calls=self.calls+1,
                        snapshot_preparation_seconds=getattr(evaluator, 'preparation_seconds', None)))
                bundle = validate_task_bundle(bundle, self.cfg, require_evaluator=True)
        except Exception:
            # A failed build must not look like a successful provider step.
            self.pool.calls = old_pool_calls
            raise
        self.calls += 1
        self.last_call_index = call_index
        self.last_model_version = model_version
        return bundle

    def state_dict(self):
        """Expose bridge state for the NEXT checkpoint patch.

        No environments, evaluator cache, Actor weights or hook internals are
        serialized here. The checkpoint module must explicitly save this dict.
        """
        return dict(format_version=1, fingerprint=self._fingerprint(), calls=self.calls,
                    pool=self.pool.state_dict(), last_call_index=self.last_call_index,
                    last_model_version=self.last_model_version)

    def load_state_dict(self, state):
        if state.get('format_version') != 1 or state.get('fingerprint') != self._fingerprint():
            raise ValueError('Rollout task-provider configuration/state mismatch')
        calls = _integer('provider calls', state['calls'])
        pool_calls = _integer('provider pool calls', state['pool']['calls'])
        if pool_calls != calls or list(state['pool']['seeds']) != self.pool.seeds:
            raise ValueError('Inconsistent provider/pool state')
        last_call = state.get('last_call_index')
        last_version = state.get('last_model_version')
        if last_call is not None:
            last_call = _integer('last_call_index', last_call)
        if last_version is not None:
            last_version = _integer('last_model_version', last_version)
        self.pool.load_state_dict(state['pool'])
        self.calls = calls
        self.last_call_index = last_call
        self.last_model_version = last_version


def build_rollout_task_provider(*, cfg, policy):
    """Factory used by --rollout-task-provider module:factory.

    This factory binds the current training Actor; it does NOT load a frozen
    checkpoint, call requires_grad_(False), or alter the real training env.
    """
    return RolloutTaskProvider(cfg, policy)


__all__ = ['make_env', 'RepresentativePool', 'RolloutTaskProvider', 'build_rollout_task_provider']
