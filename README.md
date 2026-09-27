# Cooperative Smart Traffic Management Using Multi-Agent Reinforcement Learning

Adaptive traffic signal control for a real nine-intersection network in **Assiut City, Egypt**, built on
Multi-Agent Proximal Policy Optimization (MAPPO) with Graph Transformer communication, GRU temporal
encoding, and Lagrangian constrained reinforcement learning — simulated in SUMO.

**Demo video:** https://youtu.be/YoSbFgaoOPg

This repository contains the full research codebase: the MAPPO training/evaluation pipeline, a
matched Fixed-Time baseline for comparison, an offline agent-decision analyzer, and the SUMO network
the whole system is built on.

---

## Table of contents

- [Key features](#key-features)
- [System architecture](#system-architecture)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Tech stack](#tech-stack)
- [Reported results](#reported-results)
- [Known issues](#known-issues)
- [License](#license)
- [Authors](#authors)

---

## Key features

- **Cooperative MAPPO** with Centralized Training, Decentralized Execution (CTDE) — a shared actor
  across homogeneous agents, a centralized critic over all agents.
- **Graph Transformer communication** — multi-head attention (8 heads, 4 layers) over a distance-weighted
  adjacency matrix derived from the *real* `.net.xml` topology, not an assumed grid.
- **GRU temporal encoding** — 3-layer, 128-dim recurrent state so agents reason over traffic history,
  not just the current frame.
- **Topology recovery from the network file** — `sumo_topology.py` parses the SUMO network and builds
  TLS adjacency through five staged strategies: direct edge connection, BFS through non-signalised
  junctions, geographic proximity within a distance threshold, minimum-degree backfill for leaf
  nodes, and nearest-component stitching — so no agent is ever left isolated.
- **Multi-objective reward** — weighted sum over waiting time (0.35), queue length (0.25), fairness (0.20),
  worst-lane load (0.10) and starvation avoidance (0.10).
- **Constrained RL (Lagrangian PPO)** — adaptive multipliers keep CO₂ and delay under explicit
  thresholds instead of trading them off blindly.
- **Vehicle-type awareness** — emergency, heavy and police vehicles are classified separately and
  affect reward and metric accounting.
- **Matched Fixed-Time baseline** — the comparison arm shares the MAPPO config's decision interval,
  seed, step length and normalisation constants, so the two runs see identical traffic.
- **Per-intersection ranking** with a Network Efficiency Score (`throughput / (1 + delay)`) so weak
  junctions are identifiable rather than averaged away.

---

## System architecture

```
                       ┌──────────────────────────┐
                       │   SUMO simulation        │
                       │   (9 TLS, TraCI)         │
                       └────────────┬─────────────┘
                                    │ per-agent observations (15-dim)
                       ┌────────────▼─────────────┐
                       │   FeatureExtractor       │  + RunningMeanStd normaliser
                       └────────────┬─────────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
      ┌──────────▼──────────┐            ┌───────────▼───────────┐
      │  TemporalEncoder    │            │  GraphTransformer     │
      │  (GRU, 3×128)       │            │  (8 heads, 4 layers)  │
      │  traffic history    │            │  distance-weighted    │
      └──────────┬──────────┘            │  adjacency from       │
                 │                       │  sumo_topology.py     │
                 │                       └───────────┬───────────┘
                 └──────────────┬────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
   ┌──────────▼──────────┐            ┌───────────▼──────────┐
   │    SharedActor      │            │  CentralizedCritic   │
   │  (per-agent policy) │            │  (CTDE value fn)     │
   └──────────┬──────────┘            └───────────┬──────────┘
              │        phase selection               │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  LagrangianConstraints       │  CO₂ ≤ 3000, delay ≤ 30
              │  (adaptive multipliers)      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  MAPPOTrainer                │  GAE, 4 epochs, 128 minibatch
              │  MultiAgentBuffer            │  update_every = 512
              └──────────────────────────────┘
```

Reward and constraint metrics are recomputed every step by `RewardComputer` and
`LagrangianConstraints`; `MultiAgentBuffer` stores the rollout and computes GAE advantages before
`_ppo_update` performs the clipped surrogate update.

---

## Repository structure

```
.
├── mappo/                                  # Core MAPPO implementation
│   ├── mappo_optimized_4.py                # ★ Main entry point — training + evaluation
│   ├── mappo_evaluator.py                  # Evaluate a saved checkpoint (greedy, no training)
│
├── baselines/
│   ├── sumo_topology/
│   │   └── sumo_topology.py                # Parses .net.xml → TLS graph & adjacency matrix
│   └── fixed_time/
│       └── fixed_time_for_mappo_4.py       # Fixed-Time baseline (comparison arm)
│
├── analysis/
│   └── agent_analyzer.py                   # Offline CSV decision analyzer + charts
│
├── tests/                                  # Topology sanity scripts (see Known issues)
│   ├── test_1_basic.py                     # Basic TLS adjacency output
│   ├── test_2_visual.py                    # Topology graph rendering
│   ├── test_2_visual_debug.py              # Rendering debug variant
│   ├── test_3_sanity.py                    # Sanity checks
│   ├── test_4_gat_edge.py                  # GATv2 edge-index construction
│   ├── test_5_distances.py                 # Inter-signal distance checks
│   ├── test_6_consistency.py               # Adjacency symmetry / self-loop / isolation
│   └── test_7_geo_proximity.py             # Prints torch version (not a test — see below)
│
├── sumo/                                   # Simulation assets
│   ├── project (2).sumocfg                 # ★ Simulation config
│   ├── network/
│   │   └── Version 3 George.net.xml        # 1296 junctions, 4153 edges, 9 TLS
│   ├── routes/
│   │   └── project (1).rou.xml             # Demand: 42 flows, 5 vehicle types
│   ├── detectors/
│   │   ├── project (2).add.xml             # 98 laneAreaDetectors
│   │   └── e2_0.xml … e2_94.xml            # Detector outputs 
│   ├── netedit_sessions/                   # netedit project
│   └── source_osm/                         # Raw OSM import: nodes, edges, connections
│    
│
├── requirements.txt                        # Dependencies
├── LICENSE                                 # MIT
└── .gitattributes                          # Line endings pinned to LF
```

---

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **SUMO 1.26+** — the network files were generated with Eclipse SUMO 1.26.0
- A CUDA GPU is optional but strongly recommended; the code auto-detects and falls back to CPU

### 2. Install SUMO and set `SUMO_HOME`

`traci` and `sumolib` ship inside the SUMO distribution and are **not** pip-installable.

```bash
# Linux / macOS
export SUMO_HOME=/path/to/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

```powershell
# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
```

Every entry point also appends `$SUMO_HOME/tools` to `sys.path` at import time and raises a clear
error if `SUMO_HOME` is unset.

### 3. Install Python dependencies

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
pip install -r requirements.txt
```

`torch>=2.4` is a hard floor — the code calls `torch.amp.autocast('cuda', ...)` and
`GradScaler('cuda', ...)`, whose device-type-first signature arrived in PyTorch 2.4.

---

## Usage

All commands are run from the repository root.

### Train MAPPO and evaluate afterwards

```bash
python mappo/mappo_optimized_4.py
```

`main()` takes no command-line arguments — it builds a default `MAPPOConfig`, trains, then runs a
post-training evaluation and writes `mappo_training.png` and `mappo_comparison.png` into
`mappo/saved_models/`. To change behaviour, edit the `MAPPOConfig` dataclass at the top of the file
(all hyperparameters live there: `total_steps`, `decision_interval`, `seed`, `device`, reward
weights, constraint thresholds, and so on).

### Evaluate a saved checkpoint

```bash
python mappo/mappo_evaluator.py --checkpoint mappo/saved_models/mappo_final.pt
python mappo/mappo_evaluator.py --checkpoint <path> --steps 100000
python mappo/mappo_evaluator.py --gui
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | latest in `saved_models/` | Path to a `.pt` checkpoint |
| `--sumo-cfg` | see Known issues | Path to the `.sumocfg` |
| `--steps` | `50000` | Evaluation steps |
| `--gui` | off | Run with the SUMO GUI |
| `--seed` | `42` | Random seed |

### Run the Fixed-Time baseline

```bash
python baselines/fixed_time/fixed_time_for_mappo_4.py --results-save-path results/
```

| Flag | Description |
|---|---|
| `--sumo-cfg` | Path to the `.sumocfg` |
| `--results-save-path` | Output directory |
| `--total-steps` | Simulation steps |
| `--use-gui` | Run with the SUMO GUI |

Produces `fixedtime_simulation_results.png` and `fixedtime_intersection_comparison.png`.

### Analyze agent decisions

```bash
python analysis/agent_analyzer.py --csv session.csv --report full --plots results/
```

Takes a CSV export of a session and classifies every phase decision — was the chosen phase correct
given the traffic state, which phases are over/used per intersection, what the network-level
consequences were, and where the systematic errors are. `rich` and `matplotlib` are optional; without
them the analyzer falls back to plain printing and skips charts.

---

## Tech stack

Derived from the actual imports across the codebase:

| Component | Library | Role |
|---|---|---|
| Deep learning | `torch>=2.4` | Networks, PPO update, AMP, `torch.compile` |
| Numerics | `numpy>=1.24` | Array ops throughout |
| Simulation | **SUMO 1.26+** / `traci` | Traffic simulation and live control loop |
| Analysis | `pandas>=2.0` *(optional)* | CSV session analysis |
| Terminal output | `rich>=13.0` *(optional)* | Formatted reports |
| Plotting | `matplotlib>=3.7` | Training curves, per-intersection comparisons |
| Graph utilities | `networkx>=3.0` *(tests only)* | Topology rendering |
| XML parsing | `xml.etree.ElementTree` *(stdlib)* | `.net.xml` topology extraction |
| Config | `dataclasses` *(stdlib)* | `MAPPOConfig`, `BaselineConfig`, `AnalyzerConfig` |

---

## Reported results

MAPPO compared against the matched Fixed-Time baseline on the Assiut City network.

| Metric | MAPPO | Fixed-Time | Improvement |
|---|---|---|---|
| Queue Length | 1.50 | 9.10 | 83.5 % |
| Delay (s) | 1.84 | 6.68 | 72.5 % |
| Waiting Time (s) | 18.92 | 164.50 | 88.5 % |

> **Note on reproducibility:** these figures are the project's reported outcome. No trained
> checkpoints (`*.pt`), training logs, or result CSVs are committed to this repository, so the table
> cannot currently be regenerated from a fresh clone alone — you would need to retrain with
> `mappo/mappo_optimized_4.py` first. Treat the numbers as reported results, not as a
> CI-verified benchmark.

---

## Known issues

These are real and currently unresolved. They are listed rather than hidden.

1. **The scripts in `tests/` do not run as-is.** They import `sumo_topology` (now in `mappo/`) and
   load the network via the relative path `"Version 3 George.net.xml"`, which resolves only from
   `sumo/network/`. They are standalone diagnostic scripts, not a pytest suite.

2. **`tests/test_7_geo_proximity.py` is not a test** — despite its name, its entire body is
   `print(torch.__version__)` and `print(torch.version.cuda)`.

3. **The 95 `e2_*.xml` detector files are excluded by `.gitignore`** even though
   `sumo/detectors/project (2).add.xml` names them as `laneAreaDetector` outputs. They are empty
   stubs that SUMO writes into at runtime, so a fresh clone will not contain them until the
   simulation has run once. The ignore rule is inherited from the original project and is retained
   deliberately.

4. **`.gitignore` also lists six directories that do not exist in this repository**
   (`assiut_traffic_monitor/`, `dashboard/`, `dataset_1/`, `sumo_env/`, and others) — leftovers from
   an earlier project layout.

5. **Superseded assets are preserved rather than deleted.** `sumo/routes_legacy/`,
    `sumo/netedit_sessions/` (14 variants), `sumo/legacy_configs/` and `sumo/source_osm/` all
    contain older or unused files, several referencing dead absolute paths such as
    `C:\Users\hello\...`. They are kept for provenance.

6. **Two route files disagree.** `sumo/routes/project (1).rou.xml` (42 flows) is wired into the
    simulation config. `sumo/routes_legacy/project (1).rou (1).xml` (5 trips, 10 flows) and
    `project.rou (1).xml` (1 trip) are earlier iterations and are not used.

---

## License

Released under the [MIT License](LICENSE).

Developed as part of the Work-Based Professional Project course — Sphinx University, Faculty of
Computers and Artificial Intelligence, 2026.

## Authors

George Amgad · Samaan Melad · Amir Roshdy · Mena Tharwot · Maria Soliman · Youstina Bassim ·
Mahmoud Amr · Amgad Ayman

Faculty of Computers and Artificial Intelligence, Sphinx University, 2026
