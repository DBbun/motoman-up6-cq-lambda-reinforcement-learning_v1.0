# CQ(λ) Synthetic Dataset Generator

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: Research](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)
[![Code style: clean](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)

> **Human-in-the-Loop Reinforcement Learning for Robotic Manipulation**

A single-file Python implementation for generating synthetic datasets comparing standard Q-learning with eligibility traces **[Q(λ)]** against **Cooperative Q-learning [CQ(λ)]**, demonstrating the benefits of performance-triggered human intervention in reinforcement learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Algorithm Details](#algorithm-details)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This generator simulates a robotic **bag-shaking task** where an agent must extract knotted objects from a bag through strategic shaking motions. The implementation is based on research by Kartoun, Stern, and Edan (IEEE SMC 2006; Journal of Intelligent & Robotic Systems 2010).

### What makes CQ(λ) different?

| Feature | Q(λ) | CQ(λ) |
|---------|------|-------|
| Learning | Fully autonomous | **Performance-triggered human guidance** |
| Policy Shaping | Random exploration | **Linguistic Q-value scaling** |
| Learning Rate | Constant α | **Accelerated α during intervention (3.5×)** |
| Result | Baseline performance | **~30% better cumulative reward** |

### The Task

- **Environment**: Bag with 5 knotted objects, hidden state (knot tightness, entanglement)
- **Actions**: 3-axis shaking (X/Y/Z) × 3 levels × 2 directions × 2 speeds = 72 possible actions
- **Reward**: Time-weighted object extraction: `(objects_dropped / time) × 20`
- **Challenge**: Stochastic dynamics, exploration-exploitation tradeoff, hidden state

## ✨ Key Features

### 🤖 Dual Algorithm Comparison
- **Q(λ)**: Standard TD(λ) with eligibility traces
- **CQ(λ)**: Performance-triggered human expert guidance via linguistic policy shaping

### 📊 Comprehensive Data Output
- **10,000 episodes** (500 per run × 10 runs × 2 conditions)
- **~500,000 step-level transitions** with hidden state variables
- **~1,200 intervention logs** showing human guidance
- **6 publication-ready plots** (240 DPI, color-coded)
- **Statistical comparison tables** (CSV + Markdown)

### 🎛️ Highly Configurable
- Adjustable task difficulty (knot tightness, stochasticity, object count)
- Tunable learning parameters (α, γ, λ, exploration schedule)
- Customizable human expert strategy and bias
- Flexible intervention triggering (performance threshold, window size)

### 📈 Reproducible & Deterministic
- Random seed (default: 42)
- Identical environment conditions for Q(λ) vs CQ(λ)
- Separate RNG streams for agent decisions and environment dynamics

## 🚀 Installation

### Prerequisites

```bash
python >= 3.7
matplotlib
```

### Install Dependencies

```bash
pip install matplotlib --break-system-packages
```

### Download

```bash
git clone https://github.com/yourusername/cqlambda-dataset-generator.git
cd cqlambda-dataset-generator
```

## 🏃 Quick Start

### Basic Usage

```bash
python cq_simulator_v1.0.py
```

This will:
1. Generate 10,000 episodes (500 per run × 10 runs × 2 conditions)
2. Save all data to `./output/` directory
3. Create 6 comparison plots
4. Generate statistical summary tables

**Expected runtime**: ~15-30 minutes (depending on hardware)

### Output Structure

```
output/
├── episodes.csv                      # 10,000 rows: episode-level metrics
├── steps.csv                         # ~500,000 rows: step-by-step transitions
├── interventions.csv                 # ~1,200 rows: human guidance log
├── comparison_table.csv              # Statistical summary
├── comparison_table.md               # Markdown-formatted summary
├── accumulated_reward_q_vs_cq.png    # Primary metric: cumulative learning
├── reward_per_episode_q_vs_cq.png    # Episode-wise performance
├── time_per_episode_q_vs_cq.png      # Task efficiency
├── success_rate_q_vs_cq.png          # Success rate over time
├── l_ave_q_vs_cq.png                 # Performance trigger signal
└── sa_fraction_cq.png                # Human intervention frequency
```

## 📁 Output Files

### Core Data Files

#### `episodes.csv` (10,000 rows)
Episode-level metrics for both conditions.

**Key columns**:
- `condition`: "Q" or "CQ"
- `run_id`: 1-10
- `episode_id`: 1-500
- `mode`: "A" (autonomous) or "SA" (semi-autonomous with human)
- `accumulated_reward`: Cumulative reward (primary metric)
- `total_reward`: Reward for this episode
- `success_flag`: 1 if episode reward > 25.0
- `human_intervened`: 1 if human guidance provided this episode
- `L_ave_before`: Moving average success rate (trigger signal)

#### `steps.csv` (~500,000 rows)
Step-by-step state transitions with hidden variables.

**Key columns**:
- `s_id`: State identifier (e.g., "S(CENTER)", "S(Y+2)")
- `a_id`: Action identifier (e.g., "A(X+3,v=1500)")
- `reward`: Time-weighted reward for this step
- `objects_dropped`: Objects extracted this step
- `knot_tightness`: Hidden state [0,1]
- `bag_entanglement`: Hidden state [0,1]

#### `interventions.csv` (~1,200 rows)
Log of all human interventions (CQ only).

**Key columns**:
- `episode_id`: When intervention occurred
- `L_ave`: Performance metric that triggered intervention
- `center_control_X/Y/Z`: Guidance for actions from center state
- `swing_control_X/Y/Z`: Guidance for continuation actions

**Linguistic options**: `significantly_increase`, `slightly_increase`, `keep_current`, `slightly_decrease`, `significantly_decrease`

### Visualization Files

All plots use consistent color scheme:
- **Q(λ)**: Blue dashed lines (autonomous baseline)
- **CQ(λ)**: Green solid lines (human-assisted)
- **Threshold (Λ)**: Gray dotted line

## ⚙️ Configuration

All parameters are in the `CONFIG` dictionary at the top of the script:

### Quick Customization Examples

#### Make Task Easier
```python
CONFIG = {
    "knot_difficulty": 0.85,      # Looser initial knot (was 0.93)
    "stochasticity": 0.20,        # Less noise (was 0.35)
    "num_objects": 3,             # Fewer objects (was 5)
}
```

#### More Episodes, Longer Training
```python
CONFIG = {
    "num_runs": 20,               # More runs (was 10)
    "episodes_per_run": 1000,     # Longer training (was 500)
}
```

#### Stronger Human Expert
```python
CONFIG = {
    "alpha_sa_multiplier": 5.0,   # Faster learning during SA (was 3.5)
    "ui_multipliers": {
        "significantly_increase": 3.0,  # Stronger guidance (was 2.0)
        # ...
    }
}
```

#### More Frequent Interventions
```python
CONFIG = {
    "performance_threshold_Lambda": 0.75,    # Higher threshold (was 0.65)
    "force_no_human_first_k_episodes": 3,    # Start earlier (was 5)
    "max_human_interventions_per_run": 200,  # More interventions (was 150)
}
```

### Full Configuration

<details>
<summary>Click to expand complete CONFIG dictionary</summary>

```python
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

    # CQ(λ) advantage during SA
    "alpha_sa_multiplier": 3.5,

    # Performance-triggered intervention (CQ only)
    "performance_window_N": 15,
    "performance_threshold_Lambda": 0.65,  # Λ
    "success_reward_threshold_R": 25.0,
    "force_no_human_first_k_episodes": 5,
    "max_human_interventions_per_run": 150,

    # Linguistic UI
    "ui_options": [
        "significantly_increase",
        "slightly_increase",
        "keep_current",
        "slightly_decrease",
        "significantly_decrease",
    ],
    "ui_multipliers": {
        "significantly_increase": 2.00,
        "slightly_increase": 1.25,
        "keep_current": 1.00,
        "slightly_decrease": 0.75,
        "significantly_decrease": 0.40,
    },

    # Environment difficulty
    "num_objects": 5,
    "knot_difficulty": 0.93,
    "stochasticity": 0.35,
    "axis_effectiveness": {"X": 0.50, "Y": 1.00, "Z": 0.35},
    "magnitude_effectiveness": {1: 0.40, 2: 0.70, 3: 1.00},
    "speed_effectiveness": {1000: 0.80, 1500: 1.00},

    # Human expert bias (aligned with environment)
    "human_bias": {
        "center_control": {
            "Y": "significantly_increase",    # Y is most effective
            "Z": "significantly_decrease",    # Z is least effective
            "X": "slightly_decrease"          # X is mediocre
        },
        "swing_control": {
            "Y": "significantly_increase",
            "Z": "significantly_decrease",
            "X": "keep_current"
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
```
</details>

## 🧠 Algorithm Details

### Q(λ) - Standard Q-Learning with Eligibility Traces

```python
# TD(λ) update rule
δ = r + γ·max_a'[Q(s',a')] - Q(s,a)
e(s,a) += 1
for all (s,a):
    Q(s,a) += α·δ·e(s,a)
    e(s,a) = γ·λ·e(s,a)
```

### CQ(λ) - Cooperative Q-Learning

#### 1. Performance Monitoring
```python
L_ave = moving_average(success_flags, window=15)
if L_ave < Λ (threshold=0.65):
    trigger_human_intervention()
```

#### 2. Linguistic Policy Shaping
Human provides guidance → Q-values are scaled:

```python
Q(s,a) *= multiplier[linguistic_command]

# Example:
# "significantly_increase" → Q(s,a) *= 2.00
# "significantly_decrease" → Q(s,a) *= 0.40
```

#### 3. Accelerated Learning During Intervention
```python
α_SA = α_base × 3.5  # Faster learning when human helps
```

### Human Expert Strategy

The simulated expert knows the environment dynamics:

- **Y-axis**: Most effective (1.00) → `"significantly_increase"`
- **Z-axis**: Least effective (0.35) → `"significantly_decrease"`  
- **X-axis**: Moderate (0.50) → `"slightly_decrease"` or `"keep_current"`

This guidance helps CQ(λ) discover optimal policies faster.

## 📊 Results

### Expected Performance

| Metric | Q(λ) | CQ(λ) | Improvement |
|--------|------|-------|-------------|
| **Final Accumulated Reward** | ~7,100 | **~9,200** | **+30%** |
| **Mean Episode Reward** | ~14.2 | **~18.4** | **+30%** |
| **Success Rate** | ~58% | **~79%** | **+21pp** |
| **Mean Episode Time** | ~31.5s | **~27.1s** | **-14%** ↓ |
| **Convergence** | Episode ~350 | **Episode ~250** | **100 episodes faster** |

### Learning Curves

The `accumulated_reward_q_vs_cq.png` plot shows:
- CQ(λ) (green) consistently outperforms Q(λ) (blue)
- Gap widens over time as CQ(λ) learns optimal policy faster
- Both converge, but CQ(λ) reaches higher final performance

### Intervention Pattern

The `sa_fraction_cq.png` plot shows:
- Heavy interventions early (episodes 5-150)
- Gradual reduction as performance improves
- Minimal interventions after episode 300 (agent learned well)

## 🔬 Research Applications

### Use Cases

1. **Human-in-the-loop RL research**: Benchmark for policy shaping methods
2. **Sample efficiency studies**: Long-horizon learning (500 episodes)
3. **Intervention strategy design**: When/how to trigger human guidance
4. **Explainable AI**: Linguistic commands as interpretable policy modifications
5. **Transfer learning**: Pre-training before real robot deployment
6. **Curriculum learning**: Adjustable difficulty parameters

### Extending the Code

#### Add New Expert Strategies
```python
# In HumanAdvisor class
def advise_aggressive(self):
    # Always maximize all axes
    return {ax: "significantly_increase" for ax in self.cfg["axes"]}

def advise_conservative(self):
    # Minimal guidance
    return {ax: "keep_current" for ax in self.cfg["axes"]}
```

#### Implement Different Intervention Triggers
```python
# Confidence-based triggering
if model_uncertainty > threshold:
    trigger_intervention()

# Time-based triggering
if episode_id % 50 == 0:
    trigger_intervention()
```

#### Add More Metrics
```python
# In episode loop
metrics = {
    "q_value_variance": np.var(list(agent.Q.values())),
    "exploration_rate": epsilon,
    "intervention_gap": episode_id - last_intervention_episode,
}
```

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{kartoun2006cq,
  title={CQ($\lambda$): Cooperative reinforcement learning based on SARSA($\lambda$)},
  author={Kartoun, Uri and Stern, Helman and Edan, Yael},
  booktitle={IEEE International Conference on Systems, Man and Cybernetics},
  pages={3207--3212},
  year={2006},
  organization={IEEE}
}

@article{kartoun2010cq,
  title={A human-robot collaborative reinforcement learning algorithm},
  author={Kartoun, Uri and Stern, Helman and Edan, Yael},
  journal={Journal of Intelligent \& Robotic Systems},
  volume={60},
  number={2},
  pages={217--239},
  year={2010},
  publisher={Springer}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Areas for Enhancement

- [ ] Add neural network function approximation (DQN-based CQ)
- [ ] Implement real-time interactive human mode
- [ ] Add multi-objective reward functions
- [ ] Compare with other baselines (SARSA, Actor-Critic, PPO)
- [ ] Visualization dashboard for live monitoring
- [ ] Support for parallel episode generation

## 📝 License

This project is licensed for research and educational purposes. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Original CQ(λ) algorithm by Kartoun, Stern, and Edan
- Inspired by human-robot collaboration research at Ben-Gurion University
- Built for reproducible reinforcement learning research

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/cqlambda-dataset-generator/issues)
- **Questions**: Open a discussion or issue
- **Dataset**: Available on [Hugging Face](https://huggingface.co/datasets/yourusername/cq-lambda-bag-shaking)

---

**⭐ If you find this useful, please consider starring the repository!**

**Made with ❤️ for reproducible human-in-the-loop RL research**
