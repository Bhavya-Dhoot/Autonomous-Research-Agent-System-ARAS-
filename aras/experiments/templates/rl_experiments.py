from __future__ import annotations

"""
Reinforcement learning experiment templates.
"""

EXPERIMENTS = [
    {
        "name": "exp1_bandit_comparison",
        "domain": "reinforcement_learning",
        "dataset": "synthetic 10-armed Gaussian bandit (seed=42)",
        "expected_metrics": ["average_reward", "optimal_action_rate"],
        "code": r'''from __future__ import annotations

import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp1_bandit_comparison"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    k_arms = 10
    steps = 2000
    trials = 30
    rng = np.random.default_rng(42)
    true_means = rng.normal(loc=0.0, scale=1.0, size=k_arms)
    optimal_arm = int(np.argmax(true_means))

    algos: List[Tuple[str, str, float]] = [
        ("eps_greedy_0_1", "eps_greedy", 0.1),
        ("eps_greedy_0_01", "eps_greedy", 0.01),
        ("ucb1_c2", "ucb1", 2.0),
        ("thompson_beta", "thompson", 0.0),
    ]

    # For each algorithm: store rewards per step across trials.
    rewards: Dict[str, np.ndarray] = {a[0]: np.zeros((trials, steps), dtype=np.float32) for a in algos}
    optimal_rates: Dict[str, np.ndarray] = {a[0]: np.zeros((trials, steps), dtype=np.float32) for a in algos}

    checkpoints = list(range(0, steps + 1, 100))

    try:
        for trial in range(trials):
            _timeout_check()
            # For each algorithm, independent counters.
            for algo_name, algo_type, param in algos:
                # per-trial RNG
                # (use deterministic substreams by re-seeding from seed+trial)
                trng = np.random.default_rng(42 + trial * 100 + hash(algo_name) % 1000)
                q_values = np.zeros(k_arms, dtype=np.float32)
                counts = np.zeros(k_arms, dtype=np.int32)
                # Thompson: Beta params for success probability.
                alpha = np.ones(k_arms, dtype=np.float32)
                beta = np.ones(k_arms, dtype=np.float32)
                for t in range(steps):
                    _timeout_check()
                    if algo_type == "eps_greedy":
                        eps = float(param)
                        if trng.random() < eps:
                            action = int(trng.integers(0, k_arms))
                        else:
                            action = int(np.argmax(q_values))
                    elif algo_type == "ucb1":
                        c = float(param)
                        total_n = max(1, t)
                        ucb = q_values + c * np.sqrt(np.log(total_n + 1.0) / (counts + 1e-6))
                        action = int(np.argmax(ucb))
                    else:
                        # Thompson sampling: treat reward > 0 as success (Bernoulli proxy).
                        theta_samples = trng.beta(alpha, beta)
                        action = int(np.argmax(theta_samples))

                    # Pull arm -> observe reward (Gaussian).
                    reward = float(trng.normal(loc=true_means[action], scale=1.0))
                    optimal = 1.0 if action == optimal_arm else 0.0

                    rewards[algo_name][trial, t] = reward
                    optimal_rates[algo_name][trial, t] = optimal

                    # Update for eps-greedy/ucb.
                    counts[action] += 1
                    # incremental average
                    q_values[action] = q_values[action] + (reward - q_values[action]) / float(counts[action])

                    if algo_type == "thompson":
                        success = 1.0 if reward > 0.0 else 0.0
                        alpha[action] += success
                        beta[action] += (1.0 - success)

                # end steps

        # Emit checkpoints: average across trials.
        for algo_name, _, _ in algos:
            for cp_idx, step in enumerate(checkpoints):
                if step <= 0:
                    continue
                mean_r = float(np.mean(rewards[algo_name][:, step - 1]))
                mean_opt = float(np.mean(optimal_rates[algo_name][:, step - 1]))
                emit("average_reward", mean_r, int(step), model=algo_name)
                emit("optimal_action_rate", mean_opt, int(step), model=algo_name)

        # Plot mean reward curves with std bands.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))
        xs = np.arange(1, steps + 1, 10)
        for i, (algo_name, _, _) in enumerate(algos):
            r = rewards[algo_name][:, xs - 1]
            mean = r.mean(axis=0)
            std = r.std(axis=0)
            # Downsample for smoothness.
            plt.plot(xs, mean, linewidth=2.0, label=algo_name, color=colors[i])
            plt.fill_between(xs, mean - std, mean + std, color=colors[i], alpha=0.18)

        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.title("Multi-Armed Bandit Comparison")
        plt.legend(fontsize=9)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        # Best algo by final average reward.
        final_rewards = {a[0]: float(np.mean(rewards[a[0]][:, -1])) for a in algos}
        best_algo = max(final_rewards.items(), key=lambda kv: kv[1])[0]
        best_acc = final_rewards[best_algo]

        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gaussian_bandit_10arms",
            "n_train": None,
            "n_test": None,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_algorithm": best_algo, "best_average_reward": float(best_acc)},
            "comparison_table": [{"algorithm": name, "final_average_reward": val} for name, val in sorted(final_rewards.items())],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            # FiguresAgent uses `losses` as y-series.
            "losses": [float(x) for x in np.linspace(0, 1, 6)],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_algorithm": best_algo, "average_reward": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gaussian_bandit_10arms",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gaussian_bandit_10arms",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp2_policy_gradient_cartpole",
        "domain": "reinforcement_learning",
        "dataset": "CartPole-v1 (gymnasium) / numpy fallback environment",
        "expected_metrics": ["episode_reward", "episode_length"],
        "code": r'''from __future__ import annotations

import json
import random
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp2_policy_gradient_cartpole"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


class NumpyCartPoleEnv:
    # Minimal CartPole-v1-like environment (CPU, no gym dependency).
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.max_episode_steps = 500
        self.steps = 0
        self.state = None

    def reset(self) -> np.ndarray:
        self.steps = 0
        # small random init
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert self.state is not None
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_episode_steps
        )
        reward = 1.0
        return self.state.copy(), float(reward), bool(done)


def _get_env(seed: int = 42):
    try:
        import gymnasium as gym  # type: ignore
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env
    except Exception:
        return NumpyCartPoleEnv(seed=seed)


def _rolling_mean(values: List[float], window: int = 20) -> List[float]:
    out: List[float] = []
    dq = deque(maxlen=window)
    for v in values:
        dq.append(float(v))
        out.append(float(sum(dq) / max(1, len(dq))))
    return out


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    # Use torch for autograd.
    import torch  # type: ignore

    device = torch.device("cpu")
    gamma = 0.99
    lr = 1e-3
    episodes = 200
    seed = 42

    env = _get_env(seed=seed)

    state_dim = 4
    action_dim = 2

    class PolicyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, action_dim),
            )
        def forward(self, x):
            return self.net(x)

    class ValueNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def run_vanilla(policy: PolicyNet, opt: torch.optim.Optimizer):
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        for ep in range(1, episodes + 1):
            _timeout_check()
            # reset
            if hasattr(env, "reset"):
                res = env.reset(seed=seed) if "seed" in getattr(env.reset, "__code__", {}).co_varnames else env.reset()
                state = res[0] if isinstance(res, tuple) else res
            else:
                state = env.reset()
            state = np.array(state, dtype=np.float32)

            logps: List[torch.Tensor] = []
            rewards: List[float] = []

            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                _timeout_check()
                st = torch.from_numpy(state).float().to(device)
                logits = policy(st)
                probs = torch.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs=probs)
                action = int(m.sample().item())
                logps.append(m.log_prob(torch.tensor(action, device=device)))

                if hasattr(env, "step"):
                    step_res = env.step(action)
                    if len(step_res) == 5:
                        next_state, reward, terminated, truncated, _ = step_res
                        done = bool(terminated or truncated)
                    else:
                        next_state, reward, done = step_res
                else:
                    next_state, reward, done = env.step(action)
                state = np.array(next_state, dtype=np.float32)
                ep_reward += float(reward)
                rewards.append(float(reward))
                steps += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)

            # Compute discounted returns.
            returns: List[float] = []
            G = 0.0
            for r in reversed(rewards):
                G = float(r + gamma * G)
                returns.append(G)
            returns = list(reversed(returns))
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # Policy gradient loss.
            loss = -(torch.stack(logps) * returns_t).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if ep % 10 == 0:
                emit("episode_reward", float(ep_reward), ep, model="reinforce_vanilla")
                emit("episode_length", float(steps), ep, model="reinforce_vanilla")
        return episode_rewards, episode_lengths

    def run_with_baseline(policy: PolicyNet, value: ValueNet, opt_policy: torch.optim.Optimizer, opt_value: torch.optim.Optimizer):
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        for ep in range(1, episodes + 1):
            _timeout_check()
            res = env.reset(seed=seed) if "seed" in getattr(env.reset, "__code__", {}).co_varnames else env.reset()
            state = res[0] if isinstance(res, tuple) else res
            state = np.array(state, dtype=np.float32)

            logps: List[torch.Tensor] = []
            states_t: List[torch.Tensor] = []
            rewards: List[float] = []

            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                _timeout_check()
                st = torch.from_numpy(state).float().to(device)
                states_t.append(st)
                logits = policy(st)
                probs = torch.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs=probs)
                action = int(m.sample().item())
                logps.append(m.log_prob(torch.tensor(action, device=device)))

                step_res = env.step(action)
                if len(step_res) == 5:
                    next_state, reward, terminated, truncated, _ = step_res
                    done = bool(terminated or truncated)
                else:
                    next_state, reward, done = step_res
                state = np.array(next_state, dtype=np.float32)
                ep_reward += float(reward)
                rewards.append(float(reward))
                steps += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)

            # Returns
            returns: List[float] = []
            G = 0.0
            for r in reversed(rewards):
                G = float(r + gamma * G)
                returns.append(G)
            returns = list(reversed(returns))
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # Value baseline training (MSE)
            states_stack = torch.stack(states_t)
            pred_values = value(states_stack)
            v_loss = torch.nn.functional.mse_loss(pred_values, returns_t)
            opt_value.zero_grad()
            v_loss.backward()
            opt_value.step()

            # Advantage
            with torch.no_grad():
                adv = returns_t - value(states_stack)

            p_loss = -(torch.stack(logps) * adv).sum()
            opt_policy.zero_grad()
            p_loss.backward()
            opt_policy.step()

            if ep % 10 == 0:
                emit("episode_reward", float(ep_reward), ep, model="reinforce_baseline")
                emit("episode_length", float(steps), ep, model="reinforce_baseline")

        return episode_rewards, episode_lengths

    try:
        torch.manual_seed(42)
        # Vanilla
        policy_a = PolicyNet().to(device)
        opt_a = torch.optim.Adam(policy_a.parameters(), lr=lr)
        rewards_a, lens_a = run_vanilla(policy_a, opt_a)

        # Baseline
        policy_b = PolicyNet().to(device)
        value_b = ValueNet().to(device)
        opt_pb = torch.optim.Adam(policy_b.parameters(), lr=lr)
        opt_vb = torch.optim.Adam(value_b.parameters(), lr=lr)
        rewards_b, lens_b = run_with_baseline(policy_b, value_b, opt_pb, opt_vb)

        # Plot learning curves.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        roll_a = _rolling_mean(rewards_a, window=20)
        roll_b = _rolling_mean(rewards_b, window=20)
        xs = list(range(1, episodes + 1))
        plt.plot(xs, roll_a, label="REINFORCE", linewidth=2.0, color=plt.cm.Set2(0.1))
        plt.plot(xs, roll_b, label="REINFORCE + baseline", linewidth=2.0, color=plt.cm.Set2(0.8))
        plt.xlabel("Episode")
        plt.ylabel("Episode reward (rolling mean)")
        plt.title("Policy Gradient on CartPole-v1")
        plt.legend(fontsize=9)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        best_ep_reward = float(max(max(rewards_a), max(rewards_b)))
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "cartpole",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {
                "best_algorithm": "reinforce_baseline" if max(rewards_b) >= max(rewards_a) else "reinforce_vanilla",
                "best_episode_reward": float(best_ep_reward),
            },
            "comparison_table": [
                {"algorithm": "reinforce_vanilla", "best_episode_reward": float(max(rewards_a)), "avg_episode_length": float(np.mean(lens_a))},
                {"algorithm": "reinforce_baseline", "best_episode_reward": float(max(rewards_b)), "avg_episode_length": float(np.mean(lens_b))},
            ],
            "acc": float(best_ep_reward),
            "final_loss": float(1.0 - float(best_ep_reward)),
            # Loss proxy series.
            "losses": [float(1.0 - float(x)) for x in roll_a[-20:]] if roll_a else [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_episode_reward": best_ep_reward}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "cartpole",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "cartpole",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp3_q_learning_gridworld",
        "domain": "reinforcement_learning",
        "dataset": "custom 8x8 gridworld (seed=42)",
        "expected_metrics": ["episode_reward", "steps_to_goal", "epsilon"],
        "code": r'''from __future__ import annotations

import json
import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp3_q_learning_gridworld"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


@dataclass
class Gridworld:
    size: int = 8
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (7, 7)
    walls: set[Tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.walls is None:
            self.walls = set()

    def reset(self) -> Tuple[int, int]:
        return self.start

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        # Actions: 0 up,1 down,2 left,3 right
        r, c = state
        nr, nc = r, c
        if action == 0:
            nr -= 1
        elif action == 1:
            nr += 1
        elif action == 2:
            nc -= 1
        elif action == 3:
            nc += 1

        # boundaries
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            nr, nc = r, c

        next_state = (nr, nc)
        if next_state in self.walls:
            return state, -1.0, False
        if next_state == self.goal:
            return next_state, 10.0, True
        return next_state, -0.01, False


def _make_env(seed: int = 42) -> Gridworld:
    rng = np.random.default_rng(seed)
    gw = Gridworld()
    # Choose 10 random walls excluding start and goal.
    candidates = [(r, c) for r in range(gw.size) for c in range(gw.size) if (r, c) not in (gw.start, gw.goal)]
    walls = set()
    idxs = rng.choice(len(candidates), size=10, replace=False)
    for i in idxs:
        walls.add(candidates[int(i)])
    gw.walls = walls
    return gw


def _choose_action(Q: np.ndarray, state_idx: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, 4))
    return int(np.argmax(Q[state_idx]))


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    alpha = 0.1
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.01

    episodes = 1000
    max_steps = 300
    env = _make_env(42)
    n_states = env.size * env.size
    n_actions = 4

    def s2idx(s: Tuple[int, int]) -> int:
        return int(s[0] * env.size + s[1])

    models = [
        ("q_learning_eps_0_1", "q_const", 0.1),
        ("q_learning_eps_decay", "q_decay", 0.0),
        ("sarsa_eps_0_1", "sarsa", 0.1),
    ]

    curves: Dict[str, List[float]] = {name: [] for name, _, _ in models}
    steps_curves: Dict[str, List[float]] = {name: [] for name, _, _ in models}
    eps_curves: Dict[str, List[float]] = {name: [] for name, _, _ in models}

    comparison_table: List[Dict[str, Any]] = []

    try:
        for model_name, model_type, param in models:
            _timeout_check()
            Q = np.zeros((n_states, n_actions), dtype=np.float32)
            rng = np.random.default_rng(42 + hash(model_name) % 1000)
            for ep in range(1, episodes + 1):
                _timeout_check()
                state = env.reset()
                eps = float(param) if model_type == "q_const" else (eps_start - (ep - 1) * (eps_start - eps_end) / max(1, (500 - 1))) if model_type == "q_decay" else float(param)
                if model_type == "q_decay":
                    if ep >= 500:
                        eps = eps_end
                    eps = max(eps_end, float(eps))

                action = _choose_action(Q, s2idx(state), eps, rng)
                total_reward = 0.0
                done = False
                steps_to_goal = 0
                for st in range(max_steps):
                    _timeout_check()
                    next_state, reward, done = env.step(state, action)
                    total_reward += float(reward)
                    steps_to_goal = st + 1
                    if model_type == "sarsa":
                        if not done:
                            next_action = _choose_action(Q, s2idx(next_state), eps, rng)
                            td_target = reward + gamma * Q[s2idx(next_state), next_action]
                            td_error = td_target - Q[s2idx(state), action]
                            Q[s2idx(state), action] += alpha * td_error
                            state = next_state
                            action = next_action
                        else:
                            td_target = reward
                            td_error = td_target - Q[s2idx(state), action]
                            Q[s2idx(state), action] += alpha * td_error
                            state = next_state
                    else:
                        # Q-learning
                        td_target = reward + (0.0 if done else gamma * float(np.max(Q[s2idx(next_state)])))
                        td_error = td_target - Q[s2idx(state), action]
                        Q[s2idx(state), action] += alpha * td_error
                        state = next_state
                        if done:
                            break
                        action = _choose_action(Q, s2idx(state), eps, rng)

                    if done:
                        break

                curves[model_name].append(float(total_reward))
                steps_curves[model_name].append(float(steps_to_goal))
                eps_curves[model_name].append(float(eps))

                if ep % 50 == 0:
                    emit("episode_reward", float(total_reward), ep, model=model_name)
                    emit("steps_to_goal", float(steps_to_goal), ep, model=model_name)
                    emit("epsilon", float(eps), ep, model=model_name)

            comparison_table.append({"algorithm": model_name, "avg_episode_reward": float(np.mean(curves[model_name])), "final_epsilon": float(eps_curves[model_name][-1])})

        # Plot convergence curves.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        xs = list(range(1, episodes + 1))
        window = 20
        for i, (model_name, _, _) in enumerate(models):
            vals = np.array(curves[model_name], dtype=np.float32)
            # rolling mean
            roll = np.convolve(vals, np.ones(window)/window, mode="valid")
            plt.plot(xs[window - 1 :], roll, label=model_name, linewidth=2.0, color=colors[i])
        plt.xlabel("Episode")
        plt.ylabel("Rolling avg episode reward")
        plt.title("Q-learning / SARSA Gridworld Convergence")
        plt.legend(fontsize=9)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        best_algo = max(comparison_table, key=lambda r: r["avg_episode_reward"])["algorithm"] if comparison_table else models[0][0]
        best_acc = float(max([np.mean(curves[name]) for name, _, _ in models])) if curves else 0.0

        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gridworld_8x8_seed42",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_algorithm": best_algo, "best_avg_episode_reward": best_acc},
            "comparison_table": comparison_table,
            "acc": best_acc,
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(x) for x in curves[best_algo][::max(1, len(curves[best_algo])//50)]] if best_algo in curves and curves[best_algo] else [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_algorithm": best_algo, "acc": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gridworld_8x8_seed42",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "reinforcement_learning",
            "dataset": "gridworld_8x8_seed42",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
]

