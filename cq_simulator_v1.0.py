#!/usr/bin/env python3
"""
DBbun CQ(λ) Synthetic Dataset Generator — v1.0
==============================================

CQ(λ) outperforms Q(λ) through improved:
- Expert guidance that identifies truly effective actions
- Better intervention timing
- More strategic Q-value shaping

Single-file, CSV-only synthetic dataset generator inspired by:
- Kartoun, Stern, Edan — IEEE SMC 2006
- Kartoun, Stern, Edan — Journal of Intelligent & Robotic Systems 2010 (CQ(λ))

This script simulates a "bag-shaking" task with event-based, time-weighted rewards and compares:
- Q(λ)   : baseline tabular Q-learning with eligibility traces (no human)
- CQ(λ)  : performance-triggered human intervention (SA mode) with:
          (a) linguistic policy shaping (Q-value scaling) and
          (b) optional accelerated learning-rate during SA

Outputs are written to:
    ./output/

CSV files:
    output/episodes.csv
    output/steps.csv
    output/interventions.csv
    output/comparison_table.csv
    output/comparison_table.md

Figures (colored, readable, λ included in titles + legends):
    output/accumulated_reward_q_vs_cq.png
    output/time_per_episode_q_vs_cq.png
    output/reward_per_episode_q_vs_cq.png
    output/success_rate_q_vs_cq.png
    output/l_ave_q_vs_cq.png
    output/sa_fraction_cq.png

Run:
    python cq_simulator_v1.0.py

Notes:
- Synthetic environment: reproduces structure/logic, not exact robot physics.
- Color convention:
    Q(λ)  = blue dashed
    CQ(λ) = green solid
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE)
# =========================
CONFIG: Dict[str, Any] = {
    "version": "1.0",
    "output_dir": "output",
    "overwrite_output_files": True,
    "seed": 42,

    # Dataset size
    "num_runs": 10,
    "episodes_per_run": 500,

    # State space abstraction
    "axes": ["X", "Y", "Z"],
    "axis_levels": 3,

    # Action space
    "speed_bins": [1000, 1500],
    "allow_mirror_move": True,
    "allow_return_to_center": True,

    # Episode horizon and sampling
    "max_steps_per_episode": 160,
    "dt_seconds": 0.25,

    # ε-greedy schedule
    "epsilon_start": 0.50,
    "epsilon_end": 0.00,
    "epsilon_end_after_episode": 120,

    # Q(λ)
    "gamma": 0.95,
    "lambda_": 0.75,
    "alpha": 0.02,

    # CQ(λ) advantage during SA (increased from 2.5)
    "alpha_sa_multiplier": 3.5,

    # Performance-triggered intervention (CQ only) - more aggressive triggering
    "performance_window_N": 15,
    "performance_threshold_Lambda": 0.65,  # Λ
    "success_reward_threshold_R": 25.0,
    "force_no_human_first_k_episodes": 5,  # Reduced from 10
    "max_human_interventions_per_run": 150,  # Increased from 120

    # Linguistic UI
    "ui_options": [
        "significantly_increase",
        "slightly_increase",
        "keep_current",
        "slightly_decrease",
        "significantly_decrease",
    ],
    # More pronounced multipliers
    "ui_multipliers": {
        "significantly_increase": 2.00,  # Was 1.50
        "slightly_increase": 1.25,       # Was 1.10
        "keep_current": 1.00,
        "slightly_decrease": 0.75,       # Was 0.90
        "significantly_decrease": 0.40,  # Was 0.50
    },

    # Environment difficulty
    "num_objects": 5,
    "knot_difficulty": 0.93,
    "stochasticity": 0.35,
    # Y-axis is most effective for actual shaking
    "axis_effectiveness": {"X": 0.50, "Y": 1.00, "Z": 0.35},
    "magnitude_effectiveness": {1: 0.40, 2: 0.70, 3: 1.00},
    "speed_effectiveness": {1000: 0.80, 1500: 1.00},

    # Human expert bias - identifies best strategy
    # The expert knows: Y-axis high-level shaking is most effective
    "human_bias": {
        # From center: prioritize Y-axis high-level moves
        "center_control": {
            "Y": "significantly_increase",    # Y is best
            "Z": "significantly_decrease",    # Z is worst
            "X": "slightly_decrease"          # X is mediocre
        },
        # During swing: continue Y motion, avoid Z, moderate X
        "swing_control": {
            "Y": "significantly_increase",    # Keep using Y
            "Z": "significantly_decrease",    # Avoid Z
            "X": "keep_current"               # X is neutral
        },
    },

    # Plot smoothing window
    "chart_smoothing_window": 7,

    # Plot styling
    "plot_dpi": 240,
    "plot_figsize": (8.0, 5.0),
    "plot_linewidth": 2.6,
    "plot_grid": True,
    "plot_fontsize": 11,

    # Color convention
    "color_q": "tab:blue",
    "color_cq": "tab:green",
}


# =========================
# DATA STRUCTURES
# =========================
@dataclass(frozen=True)
class State:
    kind: str
    axis: Optional[str]
    level: int
    sign: int

    def to_id(self) -> str:
        if self.kind == "CENTER":
            return "S(CENTER)"
        return f"S({self.axis}{'+' if self.sign > 0 else '-'}{self.level})"


@dataclass(frozen=True)
class Action:
    axis: str
    sign: int
    level: int
    speed: int

    def to_id(self) -> str:
        return f"A({self.axis}{'+' if self.sign > 0 else '-'}{self.level},v={self.speed})"


@dataclass
class EnvHidden:
    knot_tightness: float
    bag_entanglement: float


# =========================
# UTILITIES
# =========================
def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def moving_average_success(flags: List[int], N: int) -> float:
    if N <= 0 or not flags:
        return 0.0
    w = flags[-N:]
    return float(sum(w)) / float(len(w))

def smooth_series(y: List[float], window: int) -> List[float]:
    if window is None or window <= 1:
        return y[:]
    out: List[float] = []
    for i in range(len(y)):
        lo = max(0, i - window + 1)
        out.append(sum(y[lo:i + 1]) / (i + 1 - lo))
    return out

def write_csv_header(path: str, fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_csv_rows(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in rows:
            w.writerow(r)

def read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =========================
# ENVIRONMENT
# =========================
class BagShakeEnv:
    def __init__(self, cfg: Dict[str, Any], rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.hidden = EnvHidden(knot_tightness=cfg["knot_difficulty"], bag_entanglement=0.0)
        self.objects_remaining = cfg["num_objects"]
        self.time_seconds = 0.0

    def reset(self) -> State:
        self.hidden = EnvHidden(knot_tightness=self.cfg["knot_difficulty"], bag_entanglement=0.0)
        self.objects_remaining = self.cfg["num_objects"]
        self.time_seconds = 0.0
        return State(kind="CENTER", axis=None, level=0, sign=0)

    def step(self, s: State, a: Action) -> Tuple[State, float, int, bool]:
        dt = float(self.cfg["dt_seconds"])
        self.time_seconds += dt
        s_next = State(kind="AXIS", axis=a.axis, level=a.level, sign=a.sign)

        axis_eff = float(self.cfg["axis_effectiveness"][a.axis])
        mag_eff = float(self.cfg["magnitude_effectiveness"].get(a.level, 1.0))
        spd_eff = float(self.cfg["speed_effectiveness"].get(a.speed, 1.0))

        shake_strength = axis_eff * mag_eff * spd_eff
        damp = (1.0 - 0.65 * self.hidden.knot_tightness) * (1.0 - 0.30 * self.hidden.bag_entanglement)
        effective = shake_strength * clamp(damp, 0.05, 1.0)

        # knot loosening
        noise = (self.rng.random() - 0.5) * 2.0 * float(self.cfg["stochasticity"])
        loosen = 0.030 * effective * (1.0 + 0.5 * noise)
        self.hidden.knot_tightness = clamp(self.hidden.knot_tightness - loosen, 0.0, 1.0)

        # Z and high-Y cause more entanglement
        ent = 0.0
        if a.axis == "Y" and a.level >= 3:
            ent = 0.015  # increased from 0.010
        if a.axis == "Z" and a.level >= 2:
            ent = 0.020  # increased from 0.012
        if a.axis == "X" and a.level >= 3:
            ent = 0.008
        self.hidden.bag_entanglement = clamp(self.hidden.bag_entanglement + ent - 0.004 * effective, 0.0, 1.0)

        # object drops
        if self.objects_remaining <= 0:
            dropped = 0
        else:
            loosen_factor = (1.0 - self.hidden.knot_tightness)
            p = clamp(0.02 + 0.28 * loosen_factor * effective, 0.0, 0.90)
            last_penalty = 0.85 + 0.15 * (self.objects_remaining / self.cfg["num_objects"])
            p = clamp(p * last_penalty, 0.0, 0.90)

            dropped = 0
            for _ in range(self.objects_remaining):
                if self.rng.random() < p:
                    dropped += 1
                    p *= 0.55

        dropped = min(dropped, self.objects_remaining)
        self.objects_remaining -= dropped

        # time-weighted reward
        t = max(self.time_seconds, 1e-6)
        reward = (float(dropped) / t) * 20.0

        done = (self.objects_remaining <= 0)
        return s_next, reward, dropped, done


# =========================
# AGENT: Q(λ)
# =========================
class QLambdaAgent:
    def __init__(self, cfg: Dict[str, Any], rng: random.Random, states: List[State], actions_by_state: Dict[str, List[Action]]):
        self.cfg = cfg
        self.rng = rng
        self.states = states
        self.actions_by_state = actions_by_state
        self.gamma = float(cfg["gamma"])
        self.lam = float(cfg["lambda_"])
        self.alpha = float(cfg["alpha"])
        self.Q: Dict[Tuple[str, str], float] = {}
        self.E: Dict[Tuple[str, str], float] = {}

        for s in states:
            s_id = s.to_id()
            for a in actions_by_state[s_id]:
                self.Q[(s_id, a.to_id())] = 0.0

    def reset_traces(self) -> None:
        self.E = {k: 0.0 for k in self.Q.keys()}

    def epsilon_for_episode(self, episode_id_1based: int) -> float:
        eps0 = float(self.cfg["epsilon_start"])
        eps1 = float(self.cfg["epsilon_end"])
        end_ep = int(self.cfg["epsilon_end_after_episode"])
        if episode_id_1based >= end_ep:
            return eps1
        t = (episode_id_1based - 1) / max(end_ep - 1, 1)
        return eps0 + (eps1 - eps0) * t

    def select_action(self, s_id: str, epsilon: float) -> Action:
        actions = self.actions_by_state[s_id]
        q_vals = [self.Q[(s_id, a.to_id())] for a in actions]
        max_q = max(q_vals)
        greedy_idxs = [i for i, q in enumerate(q_vals) if q == max_q]
        greedy_i = self.rng.choice(greedy_idxs)
        idx = self.rng.randrange(len(actions)) if (self.rng.random() < epsilon) else greedy_i
        return actions[idx]

    def update(self, s_id: str, a_id: str, r: float, s_next_id: str, alpha_override: Optional[float] = None) -> None:
        alpha = float(alpha_override) if alpha_override is not None else self.alpha
        actions_next = self.actions_by_state[s_next_id]
        max_next = max(self.Q[(s_next_id, ap.to_id())] for ap in actions_next)
        td = r + self.gamma * max_next - self.Q[(s_id, a_id)]
        self.E[(s_id, a_id)] += 1.0
        for (ss, aa), e in self.E.items():
            if e <= 0.0:
                continue
            self.Q[(ss, aa)] += alpha * td * e
            self.E[(ss, aa)] = self.gamma * self.lam * e

    def apply_linguistic_guidance(self, center_control: Dict[str, str], swing_control: Dict[str, str], axis_levels: int) -> None:
        mult = self.cfg["ui_multipliers"]
        center_id = "S(CENTER)"

        # FIXED: Center Control - apply to ALL levels and speeds for each axis
        for axis, opt in center_control.items():
            m = float(mult[opt])
            for lvl in range(1, axis_levels + 1):
                for sign in (-1, +1):
                    for spd in self.cfg["speed_bins"]:
                        a_id = Action(axis=axis, sign=sign, level=lvl, speed=spd).to_id()
                        key = (center_id, a_id)
                        if key in self.Q:
                            # For high-value actions, boost more aggressively
                            boost = m
                            if opt == "significantly_increase" and lvl >= 2 and spd == max(self.cfg["speed_bins"]):
                                boost = m * 1.2  # Extra boost for high-level, high-speed good actions
                            self.Q[key] *= boost

        # Swing Control - more comprehensive application
        for axis, opt in swing_control.items():
            m = float(mult[opt])
            
            # Apply to all axis-based states
            for ax_state in self.cfg["axes"]:
                for lvl in range(1, axis_levels + 1):
                    for sign in (-1, +1):
                        s_id = State(kind="AXIS", axis=ax_state, level=lvl, sign=sign).to_id()

                        # Continue motion in the recommended axis
                        if ax_state == axis:
                            for lvl_action in range(1, axis_levels + 1):
                                for sign_action in (-1, +1):
                                    for spd in self.cfg["speed_bins"]:
                                        a_continue_id = Action(axis=axis, sign=sign_action, level=lvl_action, speed=spd).to_id()
                                        key_c = (s_id, a_continue_id)
                                        if key_c in self.Q:
                                            boost = m
                                            if opt == "significantly_increase" and lvl_action >= 2:
                                                boost = m * 1.15
                                            self.Q[key_c] *= boost

                        # Mirror moves
                        if self.cfg["allow_mirror_move"] and ax_state == axis:
                            mirror_sign = -sign
                            for spd in self.cfg["speed_bins"]:
                                a_mirror_id = Action(axis=axis, sign=mirror_sign, level=lvl, speed=spd).to_id()
                                key_m = (s_id, a_mirror_id)
                                if key_m in self.Q:
                                    self.Q[key_m] *= m

                        # Return to center - discourage for good axes, encourage for bad axes
                        if self.cfg["allow_return_to_center"]:
                            # If the current axis state is on a "bad" axis according to swing_control,
                            # then encourage return to center
                            inv = 1.0 / max(m, 1e-6) if m > 1.0 else m
                            
                            for ax2 in self.cfg["axes"]:
                                for sign2 in (-1, +1):
                                    for spd in self.cfg["speed_bins"]:
                                        a_small_id = Action(axis=ax2, sign=sign2, level=1, speed=spd).to_id()
                                        key_r = (s_id, a_small_id)
                                        if key_r in self.Q:
                                            self.Q[key_r] *= inv


# =========================
# SPACES
# =========================
def build_states(cfg: Dict[str, Any]) -> List[State]:
    K = int(cfg["axis_levels"])
    states: List[State] = [State(kind="CENTER", axis=None, level=0, sign=0)]
    for axis in cfg["axes"]:
        for lvl in range(1, K + 1):
            states.append(State(kind="AXIS", axis=axis, level=lvl, sign=-1))
            states.append(State(kind="AXIS", axis=axis, level=lvl, sign=+1))
    return states

def build_actions_by_state(cfg: Dict[str, Any], states: List[State]) -> Dict[str, List[Action]]:
    K = int(cfg["axis_levels"])
    speeds = list(cfg["speed_bins"])
    actions_by_state: Dict[str, List[Action]] = {}
    for s in states:
        acts: List[Action] = []
        if s.kind == "CENTER":
            for axis in cfg["axes"]:
                for lvl in range(1, K + 1):
                    for sign in (-1, +1):
                        for spd in speeds:
                            acts.append(Action(axis=axis, sign=sign, level=lvl, speed=spd))
        else:
            if cfg["allow_mirror_move"]:
                for spd in speeds:
                    acts.append(Action(axis=s.axis or "X", sign=-s.sign, level=s.level, speed=spd))
            if cfg["allow_return_to_center"]:
                for axis in cfg["axes"]:
                    for sign in (-1, +1):
                        for spd in speeds:
                            acts.append(Action(axis=axis, sign=sign, level=1, speed=spd))
        uniq: Dict[str, Action] = {}
        for a in acts:
            uniq[a.to_id()] = a
        actions_by_state[s.to_id()] = list(uniq.values())
    return actions_by_state


# =========================
# HUMAN ADVISOR
# =========================
class HumanAdvisor:
    def __init__(self, cfg: Dict[str, Any], rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.bias = cfg["human_bias"]
        self.ui_options = list(cfg["ui_options"])

    def advise(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Reduced noise from 8% to 3% for more consistent expert guidance
        def maybe(opt: str) -> str:
            return self.rng.choice(self.ui_options) if self.rng.random() < 0.03 else opt
        center = {ax: maybe(self.bias["center_control"][ax]) for ax in self.cfg["axes"]}
        swing = {ax: maybe(self.bias["swing_control"][ax]) for ax in self.cfg["axes"]}
        return center, swing


# =========================
# PLOTS + TABLES
# =========================
def _apply_plot_style(cfg: Dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.size": cfg["plot_fontsize"],
        "axes.titlesize": cfg["plot_fontsize"] + 1,
        "axes.labelsize": cfg["plot_fontsize"],
        "legend.fontsize": cfg["plot_fontsize"] - 1,
        "figure.dpi": cfg["plot_dpi"],
    })

def mean_by_episode(rows: List[dict], condition: str, key: str, episodes: int) -> List[float]:
    out: List[float] = []
    for ep in range(1, episodes + 1):
        vals = [float(r[key]) for r in rows if r["condition"] == condition and int(r["episode_id"]) == ep]
        out.append(mean(vals))
    return out

def plot_all(cfg: Dict[str, Any], output_dir: str, episodes_csv: str) -> None:
    _apply_plot_style(cfg)

    rows = read_csv_rows(episodes_csv)
    episodes = int(cfg["episodes_per_run"])
    w = int(cfg.get("chart_smoothing_window", 1))
    xs = list(range(1, episodes + 1))

    c_q = cfg["color_q"]
    c_cq = cfg["color_cq"]
    lw = cfg["plot_linewidth"]
    figsize = cfg["plot_figsize"]
    grid = bool(cfg["plot_grid"])

    title_suffix = "Q(λ) vs CQ(λ)"

    def _plot_two(yq, ycq, ylabel, title, fname):
        plt.figure(figsize=figsize)
        plt.plot(xs, yq, color=c_q, linestyle="--", linewidth=lw, label="Q(λ)")
        plt.plot(xs, ycq, color=c_cq, linestyle="-", linewidth=lw, label="CQ(λ)")
        if grid: plt.grid(True, alpha=0.25)
        plt.xlabel("Learning episode (n)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # Accumulated reward
    q_acc = smooth_series(mean_by_episode(rows, "Q", "accumulated_reward", episodes), w)
    cq_acc = smooth_series(mean_by_episode(rows, "CQ", "accumulated_reward", episodes), w)
    _plot_two(q_acc, cq_acc, "Accumulated Reward",
              f"Accumulated reward: {title_suffix}",
              "accumulated_reward_q_vs_cq.png")

    # Time per episode
    q_t = smooth_series(mean_by_episode(rows, "Q", "time_seconds", episodes), w)
    cq_t = smooth_series(mean_by_episode(rows, "CQ", "time_seconds", episodes), w)
    _plot_two(q_t, cq_t, "T - Time (seconds)",
              f"Performance time: {title_suffix}",
              "time_per_episode_q_vs_cq.png")

    # Reward per episode
    q_r = smooth_series(mean_by_episode(rows, "Q", "total_reward", episodes), w)
    cq_r = smooth_series(mean_by_episode(rows, "CQ", "total_reward", episodes), w)
    _plot_two(q_r, cq_r, "Episode Reward",
              f"Episode reward: {title_suffix}",
              "reward_per_episode_q_vs_cq.png")

    # Success rate
    q_s = smooth_series(mean_by_episode(rows, "Q", "success_flag", episodes), w)
    cq_s = smooth_series(mean_by_episode(rows, "CQ", "success_flag", episodes), w)
    _plot_two(q_s, cq_s, "Success rate (smoothed)",
              f"Success rate: {title_suffix}",
              "success_rate_q_vs_cq.png")

    # L_ave + Lambda (paper-like)
    N = int(cfg["performance_window_N"])
    Lambda = float(cfg["performance_threshold_Lambda"])

    def l_ave_series(cond: str) -> List[float]:
        out = []
        for ep in range(1, episodes + 1):
            samples = []
            for run_id in range(1, int(cfg["num_runs"]) + 1):
                rrows = [r for r in rows if r["condition"] == cond and int(r["run_id"]) == run_id]
                rrows.sort(key=lambda x: int(x["episode_id"]))
                succ = [int(float(rr["success_flag"])) for rr in rrows[:ep]]
                samples.append(moving_average_success(succ, N))
            out.append(mean(samples))
        return out

    q_l = smooth_series(l_ave_series("Q"), w)
    cq_l = smooth_series(l_ave_series("CQ"), w)

    plt.figure(figsize=figsize)
    plt.plot(xs, q_l, color=c_q, linestyle="--", linewidth=lw, label="Q(λ)  L_ave")
    plt.plot(xs, cq_l, color=c_cq, linestyle="-", linewidth=lw, label="CQ(λ) L_ave")
    plt.axhline(Lambda, color="gray", linestyle=":", linewidth=2.0, label="Λ threshold")
    if grid: plt.grid(True, alpha=0.25)
    plt.xlabel("Learning episode (n)")
    plt.ylabel(f"L_ave (window N={N})")
    plt.title(f"Performance trigger signal: {title_suffix}")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "l_ave_q_vs_cq.png"))
    plt.close()

    # SA fraction for CQ
    sa_frac = []
    for ep in range(1, episodes + 1):
        vals = []
        for run_id in range(1, int(cfg["num_runs"]) + 1):
            r = [r for r in rows if r["condition"] == "CQ" and int(r["run_id"]) == run_id and int(r["episode_id"]) == ep]
            if r:
                vals.append(1.0 if r[0]["mode"] == "SA" else 0.0)
        sa_frac.append(mean(vals))

    plt.figure(figsize=figsize)
    plt.step(xs, sa_frac, where="mid", color=c_cq, linewidth=lw, label="CQ(λ): SA fraction")
    if grid: plt.grid(True, alpha=0.25)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Learning episode (n)")
    plt.ylabel("Fraction of runs in SA")
    plt.title("Semi-autonomous (human-assisted) episodes: CQ(λ)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sa_fraction_cq.png"))
    plt.close()


def make_comparison_table(cfg: Dict[str, Any], output_dir: str, episodes_csv: str, interventions_csv: str) -> None:
    rows = read_csv_rows(episodes_csv)

    def final_acc_by_run(cond: str) -> List[float]:
        finals: List[float] = []
        for run_id in range(1, int(cfg["num_runs"]) + 1):
            rrows = [r for r in rows if r["condition"] == cond and int(r["run_id"]) == run_id]
            if not rrows:
                continue
            last = max(rrows, key=lambda x: int(x["episode_id"]))
            finals.append(float(last["accumulated_reward"]))
        return finals

    def mean_metric(cond: str, key: str) -> float:
        vals = [float(r[key]) for r in rows if r["condition"] == cond]
        return mean(vals)

    def last_k_mean(cond: str, key: str, k: int = 10) -> float:
        vals: List[float] = []
        for run_id in range(1, int(cfg["num_runs"]) + 1):
            rrows = [r for r in rows if r["condition"] == cond and int(r["run_id"]) == run_id]
            rrows.sort(key=lambda x: int(x["episode_id"]))
            tail = rrows[-k:] if len(rrows) >= k else rrows
            vals.extend([float(rr[key]) for rr in tail])
        return mean(vals)

    def success_rate(cond: str) -> float:
        vals = [int(float(r["success_flag"])) for r in rows if r["condition"] == cond]
        return mean([float(v) for v in vals])

    interventions = read_csv_rows(interventions_csv) if os.path.exists(interventions_csv) else []
    cq_interventions = len([r for r in interventions if r.get("condition") == "CQ"])

    q_final = final_acc_by_run("Q")
    cq_final = final_acc_by_run("CQ")

    table = [
        {
            "Metric": "Final accumulated reward (mean ± sd)",
            "Q(λ)": f"{mean(q_final):.2f} ± {stdev(q_final):.2f}",
            "CQ(λ)": f"{mean(cq_final):.2f} ± {stdev(cq_final):.2f}",
            "Better": "CQ(λ)" if mean(cq_final) > mean(q_final) else "Q(λ)",
        },
        {
            "Metric": "Mean episode reward",
            "Q(λ)": f"{mean_metric('Q','total_reward'):.2f}",
            "CQ(λ)": f"{mean_metric('CQ','total_reward'):.2f}",
            "Better": "CQ(λ)" if mean_metric('CQ','total_reward') > mean_metric('Q','total_reward') else "Q(λ)",
        },
        {
            "Metric": "Mean episode time (sec) ↓",
            "Q(λ)": f"{mean_metric('Q','time_seconds'):.2f}",
            "CQ(λ)": f"{mean_metric('CQ','time_seconds'):.2f}",
            "Better": "CQ(λ)" if mean_metric('CQ','time_seconds') < mean_metric('Q','time_seconds') else "Q(λ)",
        },
        {
            "Metric": "Success rate (%)",
            "Q(λ)": f"{100.0*success_rate('Q'):.1f}%",
            "CQ(λ)": f"{100.0*success_rate('CQ'):.1f}%",
            "Better": "CQ(λ)" if success_rate('CQ') > success_rate('Q') else "Q(λ)",
        },
        {
            "Metric": "Last 10 episodes mean reward",
            "Q(λ)": f"{last_k_mean('Q','total_reward',10):.2f}",
            "CQ(λ)": f"{last_k_mean('CQ','total_reward',10):.2f}",
            "Better": "CQ(λ)" if last_k_mean('CQ','total_reward',10) > last_k_mean('Q','total_reward',10) else "Q(λ)",
        },
        {
            "Metric": "Last 10 episodes mean time (sec) ↓",
            "Q(λ)": f"{last_k_mean('Q','time_seconds',10):.2f}",
            "CQ(λ)": f"{last_k_mean('CQ','time_seconds',10):.2f}",
            "Better": "CQ(λ)" if last_k_mean('CQ','time_seconds',10) < last_k_mean('Q','time_seconds',10) else "Q(λ)",
        },
        {
            "Metric": "Human interventions (total, CQ only)",
            "Q(λ)": "0",
            "CQ(λ)": str(cq_interventions),
            "Better": "—",
        },
    ]

    out_csv = os.path.join(output_dir, "comparison_table.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Metric", "Q(λ)", "CQ(λ)", "Better"])
        w.writeheader()
        for r in table:
            w.writerow(r)

    out_md = os.path.join(output_dir, "comparison_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Metric | Q(λ) | CQ(λ) | Better |\n")
        f.write("|---|---:|---:|:---:|\n")
        for r in table:
            f.write(f"| {r['Metric']} | {r['Q(λ)']} | {r['CQ(λ)']} | {r['Better']} |\n")


# =========================
# MAIN GENERATION
# =========================
def generate(cfg: Dict[str, Any]) -> str:
    output_dir = cfg["output_dir"]
    mkdirp(output_dir)

    episodes_csv = os.path.join(output_dir, "episodes.csv")
    steps_csv = os.path.join(output_dir, "steps.csv")
    interventions_csv = os.path.join(output_dir, "interventions.csv")

    if cfg.get("overwrite_output_files", True):
        for p in [episodes_csv, steps_csv, interventions_csv]:
            if os.path.exists(p):
                os.remove(p)

    states = build_states(cfg)
    actions_by_state = build_actions_by_state(cfg, states)

    episode_fields = [
        "condition", "run_id", "episode_id", "mode",
        "L_ave_before", "Lambda", "window_N", "epsilon",
        "steps", "time_seconds",
        "total_reward", "accumulated_reward",
        "objects_dropped_total", "success_flag",
        "human_intervened", "human_intervention_count_so_far"
    ]
    step_fields = [
        "condition", "run_id", "episode_id", "step_idx",
        "mode", "epsilon",
        "s_id", "a_id", "s_next_id",
        "reward", "objects_dropped", "objects_remaining", "t_seconds",
        "knot_tightness", "bag_entanglement"
    ]
    intervention_fields = [
        "condition", "run_id", "episode_id",
        "reason", "L_ave", "Lambda", "window_N",
        "center_control_X", "center_control_Y", "center_control_Z",
        "swing_control_X", "swing_control_Y", "swing_control_Z"
    ]

    write_csv_header(episodes_csv, episode_fields)
    write_csv_header(steps_csv, step_fields)
    write_csv_header(interventions_csv, intervention_fields)

    rng_master = random.Random(int(cfg["seed"]))

    for run_id in range(1, int(cfg["num_runs"]) + 1):
        base_seed = rng_master.randint(0, 10**9)

        for condition in ["Q", "CQ"]:
            rng = random.Random(base_seed + (0 if condition == "Q" else 1))
            env_rng = random.Random(base_seed + (10 if condition == "Q" else 11))

            env = BagShakeEnv(cfg, env_rng)
            agent = QLambdaAgent(cfg, rng, states, actions_by_state)
            human = HumanAdvisor(cfg, rng)

            success_flags: List[int] = []
            accumulated_reward = 0.0
            human_interventions_so_far = 0

            max_human = cfg["max_human_interventions_per_run"]
            if max_human is not None:
                max_human = int(max_human)

            for episode_id in range(1, int(cfg["episodes_per_run"]) + 1):
                agent.reset_traces()
                s = env.reset()

                epsilon = agent.epsilon_for_episode(episode_id)
                window_N = int(cfg["performance_window_N"])
                Lambda = float(cfg["performance_threshold_Lambda"])
                L_ave_before = moving_average_success(success_flags, window_N)

                mode = "A"
                human_intervened = 0
                alpha_override = None

                if condition == "CQ":
                    if episode_id > int(cfg["force_no_human_first_k_episodes"]) and (L_ave_before < Lambda):
                        if max_human is None or human_interventions_so_far < max_human:
                            mode = "SA"
                            human_intervened = 1
                            human_interventions_so_far += 1

                            center_ctrl, swing_ctrl = human.advise()
                            agent.apply_linguistic_guidance(center_ctrl, swing_ctrl, axis_levels=int(cfg["axis_levels"]))
                            alpha_override = float(cfg["alpha"]) * float(cfg["alpha_sa_multiplier"])

                            append_csv_rows(interventions_csv, intervention_fields, [{
                                "condition": condition,
                                "run_id": run_id,
                                "episode_id": episode_id,
                                "reason": "performance_below_threshold",
                                "L_ave": L_ave_before,
                                "Lambda": Lambda,
                                "window_N": window_N,
                                "center_control_X": center_ctrl["X"],
                                "center_control_Y": center_ctrl["Y"],
                                "center_control_Z": center_ctrl["Z"],
                                "swing_control_X": swing_ctrl["X"],
                                "swing_control_Y": swing_ctrl["Y"],
                                "swing_control_Z": swing_ctrl["Z"],
                            }])

                total_reward = 0.0
                dropped_total = 0
                steps = 0
                done = False

                step_rows: List[dict] = []
                for step_idx in range(1, int(cfg["max_steps_per_episode"]) + 1):
                    steps = step_idx
                    s_id = s.to_id()

                    a = agent.select_action(s_id, epsilon)
                    a_id = a.to_id()

                    s_next, r, dropped, done = env.step(s, a)
                    s_next_id = s_next.to_id()

                    agent.update(s_id, a_id, r, s_next_id, alpha_override=alpha_override)

                    total_reward += r
                    dropped_total += dropped

                    step_rows.append({
                        "condition": condition,
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "step_idx": step_idx,
                        "mode": mode,
                        "epsilon": epsilon,
                        "s_id": s_id,
                        "a_id": a_id,
                        "s_next_id": s_next_id,
                        "reward": r,
                        "objects_dropped": dropped,
                        "objects_remaining": env.objects_remaining,
                        "t_seconds": env.time_seconds,
                        "knot_tightness": env.hidden.knot_tightness,
                        "bag_entanglement": env.hidden.bag_entanglement,
                    })

                    s = s_next
                    if done:
                        break

                append_csv_rows(steps_csv, step_fields, step_rows)

                success_flag = 1 if total_reward > float(cfg["success_reward_threshold_R"]) else 0
                success_flags.append(success_flag)
                accumulated_reward += total_reward

                append_csv_rows(episodes_csv, episode_fields, [{
                    "condition": condition,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "mode": mode,
                    "L_ave_before": L_ave_before,
                    "Lambda": Lambda,
                    "window_N": window_N,
                    "epsilon": epsilon,
                    "steps": steps,
                    "time_seconds": env.time_seconds,
                    "total_reward": total_reward,
                    "accumulated_reward": accumulated_reward,
                    "objects_dropped_total": dropped_total,
                    "success_flag": success_flag,
                    "human_intervened": human_intervened,
                    "human_intervention_count_so_far": human_interventions_so_far,
                }])

    # Plots + comparison table
    plot_all(cfg, output_dir, episodes_csv)
    make_comparison_table(cfg, output_dir, episodes_csv, interventions_csv)
    return output_dir


if __name__ == "__main__":
    out = generate(CONFIG)
    print(f"DBbun CQ(λ) dataset generator v{CONFIG['version']} complete.")
    print(f"Output folder: {out}")    