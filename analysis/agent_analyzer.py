"""
=============================================================================
 agent_analyzer.py — MAPPO Decision Analyzer
=============================================================================
 Reads the exported CSV from the dashboard and analyzes every agent decision:
   - Was the chosen phase the correct one given the traffic state?
   - Which phases are being over/under-used per intersection?
   - What were the network consequences of each decision?
   - Where are the agent's systematic mistakes?
   - Generates an improvement report with specific recommendations.

 Usage:
   python agent_analyzer.py --csv session.csv
   python agent_analyzer.py --csv session.csv --mode MAPPO --episodes 5
   python agent_analyzer.py --csv session.csv --intersection Gomharia
   python agent_analyzer.py --csv session.csv --report full
=============================================================================
"""

import os
import sys
import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional rich console output ────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import track
    from rich import box as rbox
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class _FallbackConsole:
        def print(self, *a, **k): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}\n")
    console = _FallbackConsole()

# ── Optional matplotlib for charts ──────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — mirrors MAPPOConfig thresholds                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class AnalyzerConfig:
    # Reward weights (from MAPPOConfig)
    reward_w_wait:       float = 0.35
    reward_w_queue:      float = 0.25
    reward_w_fairness:   float = 0.20
    reward_w_max_lane:   float = 0.10
    reward_w_starvation: float = 0.10

    # Lagrangian thresholds
    co2_threshold:   float = 3000.0
    delay_threshold: float = 30.0

    # Phase decision thresholds
    high_queue_threshold:  float = 8.0    # veh — queue considered high
    high_delay_threshold:  float = 25.0   # s   — delay considered high
    low_speed_threshold:   float = 2.5    # m/s — speed considered low

    # Decision quality thresholds
    good_nes_threshold: float = 0.50
    warn_nes_threshold: float = 0.25

    # Starvation: if same phase held for this many consecutive steps
    starvation_steps: int = 6


CFG = AnalyzerConfig()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVATION FEATURE NAMES (from FeatureExtractor.extract)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

OBS_FEATURES = [
    "mean_queue_norm",   # 0  mean halting / max_cars
    "max_queue_norm",    # 1  max halting lane / max_cars
    "count_norm",        # 2  total vehicle count / max_cars
    "mean_occ",          # 3  mean lane occupancy
    "mean_speed",        # 4  mean speed / max_speed
    "heavy_ratio",       # 5  heavy vehicle ratio
    "emergency",         # 6  emergency vehicle flag
    "wt_norm",           # 7  waiting time / max_wait
    "co2_norm",          # 8  CO2 / max_co2
    "jam_norm",          # 9  jam factor
    "phase_norm",        # 10 current phase / (num_phases-1)
    "dur_norm",          # 11 phase duration / max_green
    "throughput",        # 12 vehicle count / max_cars
    "time_norm",         # 13 sim time / 3600
    "weather_factor",    # 14 weather multiplier
]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DECISION QUALITY CLASSIFIER                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DecisionClassifier:
    """
    Classifies each agent decision as GOOD / ACCEPTABLE / MISTAKE
    based on the traffic state at that step and the resulting outcome.

    Logic:
      - MISTAKE: high queue + high delay → agent held/chose a phase that
                 did not serve the most congested approach
      - MISTAKE: emergency vehicle present → phase did not change to
                 serve emergency (phase stayed same for >2 steps)
      - MISTAKE: starvation — same phase held for starvation_steps+ steps
                 while other approaches have high queue
      - MISTAKE: CO2 above threshold AND delay above threshold simultaneously
                 (Lagrangian constraint should have prevented this)
      - ACCEPTABLE: moderate conditions, no clear signal of wrong choice
      - GOOD: queue decreasing, delay low, NES improving
    """

    def classify(self, row: pd.Series, prev_row: Optional[pd.Series],
                 phase_history: list) -> dict:
        result = {
            "verdict":  "GOOD",
            "score":    1.0,
            "reasons":  [],
            "severity": 0,   # 0=good, 1=warn, 2=mistake
        }

        q    = row.get("queue",      0)
        d    = row.get("delay",      0)
        nes  = row.get("nes",        0)
        spd  = row.get("speed",      0)
        co2  = row.get("co2",        0) * 1000  # back to kg → mg/s scale
        emg  = row.get("emergency",  0) if "emergency" in row.index else 0
        ph   = int(row.get("phase",  0))

        # ── Check 1: high queue + high delay (agent not relieving congestion)
        if q > CFG.high_queue_threshold and d > CFG.high_delay_threshold:
            result["reasons"].append(
                f"High queue ({q:.1f} veh) + high delay ({d:.1f}s) — "
                f"phase {ph} may not be serving most congested approach")
            result["severity"] = max(result["severity"], 2)

        # ── Check 2: NES below warning threshold
        if nes < CFG.warn_nes_threshold:
            result["reasons"].append(
                f"Very low NES ({nes:.3f}) — intersection is near-gridlock")
            result["severity"] = max(result["severity"], 2)
        elif nes < CFG.good_nes_threshold:
            result["reasons"].append(
                f"Below-target NES ({nes:.3f}) — partial congestion")
            result["severity"] = max(result["severity"], 1)

        # ── Check 3: phase starvation (same phase too long)
        if len(phase_history) >= CFG.starvation_steps:
            last_n = phase_history[-CFG.starvation_steps:]
            if all(p == ph for p in last_n) and q > CFG.high_queue_threshold:
                result["reasons"].append(
                    f"Phase starvation — phase {ph} held for "
                    f"{CFG.starvation_steps}+ steps while queue={q:.1f}")
                result["severity"] = max(result["severity"], 2)

        # ── Check 4: low speed → vehicles not moving despite green
        if spd < CFG.low_speed_threshold and q > 5:
            result["reasons"].append(
                f"Low speed ({spd:.2f} m/s) with queue {q:.1f} — "
                f"phase {ph} not clearing effectively")
            result["severity"] = max(result["severity"], 1)

        # ── Check 5: CO2 + delay both above Lagrangian threshold
        if co2 > CFG.co2_threshold and d > CFG.delay_threshold:
            result["reasons"].append(
                f"Both CO₂ ({co2:.0f}mg/s) and delay ({d:.1f}s) exceed "
                f"Lagrangian thresholds — constraint not active enough")
            result["severity"] = max(result["severity"], 1)

        # ── Assign verdict ──────────────────────────────────────────────────
        if result["severity"] == 2:
            result["verdict"] = "MISTAKE"
            result["score"]   = 0.0
        elif result["severity"] == 1:
            result["verdict"] = "ACCEPTABLE"
            result["score"]   = 0.5
        else:
            result["verdict"] = "GOOD"
            result["score"]   = 1.0

        return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NETWORK IMPACT ANALYZER                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class NetworkImpactAnalyzer:
    """
    For each decision, measures how it affected the NETWORK (not just local).
    A decision is locally good but network-bad if:
      - It reduces local queue but pushes traffic to already-congested neighbors
      - The network NES drops in the N steps following this decision
    """

    def analyze_spillover(self, df: pd.DataFrame,
                          intersection_id: str,
                          step: int,
                          window: int = 5) -> dict:
        """
        Look at network-average NES in the [step, step+window] window
        compared to [step-window, step]. Negative delta = decision
        caused network degradation.
        """
        before = df[
            (df["step"] >= step - window) & (df["step"] < step)
        ]["nes"].mean()
        after = df[
            (df["step"] > step) & (df["step"] <= step + window)
        ]["nes"].mean()

        if pd.isna(before) or pd.isna(after):
            return {"delta_nes": 0.0, "spillover": False}

        delta = after - before
        return {
            "delta_nes": round(delta, 4),
            "spillover": delta < -0.05  # 5% NES drop = spillover
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE BIAS DETECTOR                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PhaseBiasDetector:
    """
    Detects whether the agent has learned a biased phase preference.
    If one phase is chosen significantly more often than others at a given
    intersection, that may indicate the agent is ignoring certain approaches.
    """

    def detect(self, phase_counts: dict, total: int) -> dict:
        if total == 0 or not phase_counts:
            return {"biased": False, "dominant_phase": None, "dominance": 0}

        max_phase = max(phase_counts, key=phase_counts.get)
        max_count = phase_counts[max_phase]
        dominance = max_count / total

        # Uniform = 1/N phases each. Biased if dominant > 2×uniform
        n_phases = len(phase_counts)
        uniform_share = 1.0 / n_phases if n_phases > 0 else 1.0
        biased = dominance > 2.5 * uniform_share

        return {
            "biased":          biased,
            "dominant_phase":  max_phase,
            "dominance":       round(dominance, 3),
            "phase_counts":    phase_counts,
            "uniform_share":   round(uniform_share, 3),
            "n_phases":        n_phases,
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  IMPROVEMENT RECOMMENDER                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ImprovementRecommender:
    """
    Takes the analysis results and generates concrete, actionable
    recommendations for improving the MAPPO agent.
    """

    def recommend(self, analysis: dict) -> list:
        recs = []

        # ── Starvation-heavy intersections ──────────────────────────────────
        for iid, stats in analysis["per_intersection"].items():
            if stats["starvation_rate"] > 0.15:
                recs.append({
                    "priority":  "HIGH",
                    "target":    iid,
                    "issue":     f"Phase starvation ({stats['starvation_rate']:.0%} of steps)",
                    "fix":       "Increase reward_w_fairness or reward_w_starvation in MAPPOConfig. "
                                 "Add a starvation penalty that grows with consecutive same-phase count.",
                    "reward_param": "reward_w_starvation",
                    "suggested_value": min(CFG.reward_w_starvation + 0.05, 0.25),
                })

        # ── High mistake rate intersections ─────────────────────────────────
        for iid, stats in analysis["per_intersection"].items():
            if stats["mistake_rate"] > 0.20:
                recs.append({
                    "priority":  "HIGH",
                    "target":    iid,
                    "issue":     f"High mistake rate ({stats['mistake_rate']:.0%})",
                    "fix":       "Review observation features for this intersection. "
                                 "If it is a cluster node (Kobry1/Kobry2), the GRU may need "
                                 "longer history — increase episode_steps or add a neighbor "
                                 "queue feature explicitly.",
                    "reward_param": "reward_w_queue",
                    "suggested_value": min(CFG.reward_w_queue + 0.05, 0.40),
                })

        # ── Phase bias ───────────────────────────────────────────────────────
        for iid, stats in analysis["per_intersection"].items():
            bias = stats.get("phase_bias", {})
            if bias.get("biased"):
                recs.append({
                    "priority":  "MEDIUM",
                    "target":    iid,
                    "issue":     f"Phase bias: phase {bias['dominant_phase']} used "
                                 f"{bias['dominance']:.0%} of steps "
                                 f"(uniform = {bias['uniform_share']:.0%})",
                    "fix":       "Increase phase_diversity_coef in MAPPOConfig (currently 0.02). "
                                 "This adds an entropy bonus that discourages over-concentration "
                                 "on one phase.",
                    "reward_param": "phase_diversity_coef",
                    "suggested_value": 0.05,
                })

        # ── Lagrangian constraint violations ─────────────────────────────────
        lag_rate = analysis["global"]["lagrangian_violation_rate"]
        if lag_rate > 0.10:
            recs.append({
                "priority":  "MEDIUM",
                "target":    "ALL",
                "issue":     f"Lagrangian constraints violated {lag_rate:.0%} of steps",
                "fix":       "Reduce co2_threshold or delay_threshold in MAPPOConfig, "
                             "or increase lagrangian_lr from 1e-3 to 3e-3 to make "
                             "the Lagrangian multiplier respond faster to violations.",
                "reward_param": "lagrangian_lr",
                "suggested_value": 3e-3,
            })

        # ── Low overall NES ──────────────────────────────────────────────────
        if analysis["global"]["mean_nes"] < CFG.warn_nes_threshold:
            recs.append({
                "priority":  "HIGH",
                "target":    "NETWORK",
                "issue":     f"Mean NES {analysis['global']['mean_nes']:.3f} below warning threshold",
                "fix":       "Consider curriculum learning: start training at 50% vehicle "
                             "density and gradually increase. The agent may be stuck in a "
                             "local optimum learned under high-density conditions.",
                "reward_param": "total_steps",
                "suggested_value": "increase + curriculum",
            })

        # ── No improvement over episodes ─────────────────────────────────────
        ep_trend = analysis["global"].get("nes_episode_trend", 0)
        if ep_trend < 0:
            recs.append({
                "priority":  "HIGH",
                "target":    "TRAINING",
                "issue":     f"NES is declining across episodes (trend={ep_trend:+.4f}/ep)",
                "fix":       "Check for reward instability. Consider reducing actor_lr "
                             "from 3e-4 to 1e-4 and increasing ppo_epochs from 4 to 6. "
                             "Also verify that the GRU hidden state is being reset "
                             "correctly at episode boundaries.",
                "reward_param": "actor_lr",
                "suggested_value": 1e-4,
            })

        # Sort by priority
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recs.sort(key=lambda r: order.get(r["priority"], 3))
        return recs


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ANALYZER                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AgentAnalyzer:
    """
    Main orchestrator. Loads CSV, runs all analysis components,
    and produces a structured report.
    """

    def __init__(self, csv_path: str, mode_filter: str = None,
                 episode_filter: int = None, intersection_filter: str = None):
        self.csv_path           = csv_path
        self.mode_filter        = mode_filter
        self.episode_filter     = episode_filter
        self.intersection_filter = intersection_filter

        self.classifier  = DecisionClassifier()
        self.impact      = NetworkImpactAnalyzer()
        self.bias_det    = PhaseBiasDetector()
        self.recommender = ImprovementRecommender()

        self.df       = None
        self.analysis = {}

    # ── Load and validate ────────────────────────────────────────────────────
    def load(self):
        console.print(f"[bold]Loading:[/bold] {self.csv_path}" if RICH
                      else f"Loading: {self.csv_path}")

        required_cols = {"step", "id", "phase", "queue", "delay", "nes"}
        df = pd.read_csv(self.csv_path)

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Available: {list(df.columns)}\n"
                "Make sure you exported from the latest dashboard version "
                "(E key or ⬇ CSV button)."
            )

        # Apply filters
        if self.mode_filter:
            df = df[df["mode"].str.upper() == self.mode_filter.upper()]
        if self.episode_filter is not None:
            df = df[df["episode"] >= self.episode_filter]
        if self.intersection_filter:
            mask = (df["label"].str.lower() == self.intersection_filter.lower()) | \
                   (df["id"] == self.intersection_filter)
            df = df[mask]

        df = df.sort_values(["id", "step"]).reset_index(drop=True)

        # Derived columns
        if "label" not in df.columns:
            df["label"] = df["id"]
        if "throughput" not in df.columns:
            df["throughput"] = df["nes"] * (1 + df["delay"])

        self.df = df
        console.print(
            f"  Loaded {len(df):,} rows · "
            f"{df['id'].nunique()} intersections · "
            f"{df['step'].max():,} steps · "
            f"{df['episode'].nunique() if 'episode' in df.columns else '?'} episodes"
            if RICH else
            f"  {len(df):,} rows, {df['id'].nunique()} intersections"
        )
        return self

    # ── Core analysis ────────────────────────────────────────────────────────
    def analyze(self):
        df = self.df
        per_intersection = {}
        all_decisions    = []
        network_df       = df.groupby("step")[["queue","delay","nes","speed"]].mean()

        intersections = df["id"].unique()
        iterable = track(intersections, description="Analyzing...") \
                   if RICH else intersections

        for iid in iterable:
            sub = df[df["id"] == iid].sort_values("step").reset_index(drop=True)
            label = sub["label"].iloc[0] if "label" in sub.columns else iid

            phase_history  = []
            phase_counts   = defaultdict(int)
            decisions      = []
            starvation_ct  = 0
            mistake_ct     = 0

            for i, row in sub.iterrows():
                ph = int(row.get("phase", 0))
                phase_counts[ph] += 1
                phase_history.append(ph)

                prev = sub.iloc[i-1] if i > 0 else None
                verdict = self.classifier.classify(row, prev, phase_history)
                impact  = self.impact.analyze_spillover(
                    network_df.reset_index(), iid, int(row["step"]))

                if verdict["verdict"] == "MISTAKE":
                    mistake_ct += 1

                # Starvation check
                if (len(phase_history) >= CFG.starvation_steps and
                        all(p == ph for p in phase_history[-CFG.starvation_steps:])):
                    starvation_ct += 1

                decisions.append({
                    "step":        int(row["step"]),
                    "episode":     int(row.get("episode", 1)),
                    "intersection": label,
                    "phase":       ph,
                    "queue":       float(row.get("queue", 0)),
                    "delay":       float(row.get("delay", 0)),
                    "nes":         float(row.get("nes",   0)),
                    "speed":       float(row.get("speed", 0)),
                    "co2":         float(row.get("co2",   0)),
                    "verdict":     verdict["verdict"],
                    "score":       verdict["score"],
                    "reasons":     verdict["reasons"],
                    "delta_nes":   impact["delta_nes"],
                    "spillover":   impact["spillover"],
                })

            n = len(sub)
            bias = self.bias_det.detect(dict(phase_counts), n)

            per_intersection[iid] = {
                "label":           label,
                "total_decisions": n,
                "mistake_rate":    round(mistake_ct / max(n, 1), 4),
                "starvation_rate": round(starvation_ct / max(n, 1), 4),
                "mean_nes":        round(sub["nes"].mean(), 4),
                "mean_queue":      round(sub["queue"].mean(), 4),
                "mean_delay":      round(sub["delay"].mean(), 4),
                "phase_counts":    dict(phase_counts),
                "phase_bias":      bias,
                "decisions":       decisions,
            }
            all_decisions.extend(decisions)

        # ── Global stats ─────────────────────────────────────────────────────
        total     = len(all_decisions)
        mistakes  = sum(1 for d in all_decisions if d["verdict"] == "MISTAKE")
        spillovers= sum(1 for d in all_decisions if d.get("spillover"))

        # NES trend across episodes
        nes_by_ep = (df.groupby("episode")["nes"].mean()
                     if "episode" in df.columns else pd.Series(dtype=float))
        if len(nes_by_ep) >= 2:
            x = np.arange(len(nes_by_ep))
            nes_trend = float(np.polyfit(x, nes_by_ep.values, 1)[0])
        else:
            nes_trend = 0.0

        # Lagrangian violations
        lag_violations = df[
            (df["co2"] * 1000 > CFG.co2_threshold) &
            (df["delay"] > CFG.delay_threshold)
        ]

        global_stats = {
            "total_decisions":         total,
            "mistake_count":           mistakes,
            "mistake_rate":            round(mistakes / max(total, 1), 4),
            "spillover_count":         spillovers,
            "spillover_rate":          round(spillovers / max(total, 1), 4),
            "mean_nes":                round(df["nes"].mean(), 4),
            "mean_delay":              round(df["delay"].mean(), 4),
            "mean_queue":              round(df["queue"].mean(), 4),
            "lagrangian_violation_rate": round(len(lag_violations) / max(len(df), 1), 4),
            "nes_episode_trend":       round(nes_trend, 6),
            "total_steps":             int(df["step"].max()),
            "total_episodes":          int(df["episode"].nunique()) if "episode" in df.columns else 1,
        }

        # ── Recommendations ──────────────────────────────────────────────────
        self.analysis = {
            "global":           global_stats,
            "per_intersection": per_intersection,
            "all_decisions":    all_decisions,
        }
        recommendations = self.recommender.recommend(self.analysis)
        self.analysis["recommendations"] = recommendations

        return self

    # ── Print report ─────────────────────────────────────────────────────────
    def print_report(self, level: str = "full"):
        a  = self.analysis
        g  = a["global"]
        pi = a["per_intersection"]

        # ── Header ───────────────────────────────────────────────────────────
        if RICH:
            console.rule("[bold cyan]MAPPO Agent Decision Analysis Report[/bold cyan]")
        else:
            console.rule("MAPPO Agent Decision Analysis Report")

        print(f"\n  Total decisions : {g['total_decisions']:,}")
        print(f"  Total steps     : {g['total_steps']:,}")
        print(f"  Episodes        : {g['total_episodes']}")
        print(f"  Mean NES        : {g['mean_nes']:.4f}")
        print(f"  Mean Delay      : {g['mean_delay']:.2f}s")
        print(f"  Mean Queue      : {g['mean_queue']:.2f} veh")
        print(f"  Mistake rate    : {g['mistake_rate']:.1%}")
        print(f"  Spillover rate  : {g['spillover_rate']:.1%}")
        print(f"  Lagrangian violations: {g['lagrangian_violation_rate']:.1%}")
        print(f"  NES trend /ep   : {g['nes_episode_trend']:+.5f}")
        print()

        # ── Per-intersection table ────────────────────────────────────────────
        if RICH:
            console.rule("[bold]Per-Intersection Summary[/bold]")
            tbl = Table(box=rbox.SIMPLE_HEAVY, show_footer=False,
                        style="dim", header_style="bold cyan")
            tbl.add_column("Intersection",  style="white",  min_width=18)
            tbl.add_column("Mean NES",       justify="right")
            tbl.add_column("Mean Q",         justify="right")
            tbl.add_column("Mean D",         justify="right")
            tbl.add_column("Mistake %",      justify="right")
            tbl.add_column("Starvation %",   justify="right")
            tbl.add_column("Bias",           justify="center")
            tbl.add_column("Dom. Phase",     justify="center")

            sorted_ids = sorted(pi, key=lambda k: pi[k]["mean_nes"], reverse=True)
            for iid in sorted_ids:
                s = pi[iid]
                bias = s["phase_bias"]
                nes_col  = ("[green]" if s["mean_nes"] > CFG.good_nes_threshold
                             else "[yellow]" if s["mean_nes"] > CFG.warn_nes_threshold
                             else "[red]") + f"{s['mean_nes']:.3f}"
                mis_col  = ("[red]" if s["mistake_rate"] > .20 else
                             "[yellow]" if s["mistake_rate"] > .10 else
                             "[green]") + f"{s['mistake_rate']:.1%}"
                star_col = ("[red]" if s["starvation_rate"] > .15 else
                             "[yellow]" if s["starvation_rate"] > .05 else
                             "[green]") + f"{s['starvation_rate']:.1%}"
                tbl.add_row(
                    s["label"], nes_col,
                    f"{s['mean_queue']:.1f}", f"{s['mean_delay']:.1f}s",
                    mis_col, star_col,
                    ("[red]YES" if bias["biased"] else "[green]NO"),
                    str(bias.get("dominant_phase", "—")),
                )
            console.print(tbl)
        else:
            print(f"{'Intersection':<22} {'NES':>6} {'Q':>6} {'D':>7} {'Err%':>6} {'Star%':>6} {'Bias':>5}")
            print("─" * 65)
            for iid in sorted(pi, key=lambda k: pi[k]["mean_nes"], reverse=True):
                s = pi[iid]
                bias = s["phase_bias"]
                print(f"{s['label']:<22} "
                      f"{s['mean_nes']:>6.3f} "
                      f"{s['mean_queue']:>6.1f} "
                      f"{s['mean_delay']:>6.1f}s "
                      f"{s['mistake_rate']:>6.1%} "
                      f"{s['starvation_rate']:>6.1%} "
                      f"{'YES' if bias['biased'] else 'no':>5}")

        # ── Top mistakes (full report only) ──────────────────────────────────
        if level == "full":
            print()
            if RICH:
                console.rule("[bold red]Top Decision Mistakes[/bold red]")
            else:
                console.rule("Top Decision Mistakes")

            mistakes = [d for d in a["all_decisions"] if d["verdict"] == "MISTAKE"]
            mistakes_sorted = sorted(
                mistakes, key=lambda d: d["queue"] + d["delay"], reverse=True)[:20]

            for d in mistakes_sorted:
                print(f"\n  Step {d['step']:>6} | Ep {d['episode']:>3} | "
                      f"{d['intersection']:<22} | Phase {d['phase']} | "
                      f"Q={d['queue']:.1f} D={d['delay']:.1f}s "
                      f"NES={d['nes']:.3f} ΔNetNES={d['delta_nes']:+.4f}")
                for r in d["reasons"]:
                    print(f"           → {r}")

        # ── Recommendations ───────────────────────────────────────────────────
        print()
        if RICH:
            console.rule("[bold yellow]Improvement Recommendations[/bold yellow]")
        else:
            console.rule("Improvement Recommendations")

        recs = a["recommendations"]
        if not recs:
            print("  No critical issues found. Agent performance is satisfactory.")
        else:
            for i, r in enumerate(recs, 1):
                pri_sym = {"HIGH": "!!!", "MEDIUM": " ! ", "LOW": " · "}.get(r["priority"], " ? ")
                print(f"\n  [{pri_sym}] {r['priority']} — {r['target']}")
                print(f"       Issue : {r['issue']}")
                print(f"       Fix   : {r['fix']}")
                print(f"       Param : {r['reward_param']} → {r['suggested_value']}")

    # ── Save JSON report ──────────────────────────────────────────────────────
    def save_json(self, out_path: str):
        out = {
            "global":          self.analysis["global"],
            "per_intersection": {
                k: {kk: vv for kk, vv in v.items() if kk != "decisions"}
                for k, v in self.analysis["per_intersection"].items()
            },
            "recommendations": self.analysis["recommendations"],
            "top_mistakes": sorted(
                [d for d in self.analysis["all_decisions"] if d["verdict"] == "MISTAKE"],
                key=lambda d: d["queue"] + d["delay"], reverse=True
            )[:50],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved → {out_path}")

    # ── Plot charts ───────────────────────────────────────────────────────────
    def plot(self, out_dir: str = "."):
        if not MATPLOTLIB:
            print("matplotlib not installed — skipping charts.")
            return

        os.makedirs(out_dir, exist_ok=True)
        pi = self.analysis["per_intersection"]
        df = self.df

        # ── Chart 1: Mistake rate per intersection ────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        labels  = [pi[k]["label"] for k in pi]
        rates   = [pi[k]["mistake_rate"] for k in pi]
        colors  = ["#e53935" if r > .20 else "#fb8c00" if r > .10 else "#43a047"
                   for r in rates]
        bars = ax.bar(labels, rates, color=colors, edgecolor="#1a1a1a", linewidth=0.5)
        ax.axhline(0.20, color="#e53935", linestyle="--", linewidth=1,
                   label="High threshold (20%)")
        ax.axhline(0.10, color="#fb8c00", linestyle="--", linewidth=1,
                   label="Warning threshold (10%)")
        ax.set_title("Decision Mistake Rate per Intersection", fontsize=13, fontweight="bold")
        ax.set_ylabel("Mistake Rate")
        ax.set_ylim(0, min(max(rates) * 1.3 + 0.05, 1.0))
        ax.legend(fontsize=9)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mistake_rate.png"), dpi=150)
        plt.close()
        print(f"  Chart saved → {out_dir}/mistake_rate.png")

        # ── Chart 2: Phase distribution heatmap ──────────────────────────────
        max_phases = max(
            max(pi[k]["phase_counts"].keys(), default=0) for k in pi) + 1
        matrix = np.zeros((len(pi), max_phases))
        row_labels = []
        for ri, k in enumerate(pi):
            row_labels.append(pi[k]["label"])
            total = pi[k]["total_decisions"]
            for ph, cnt in pi[k]["phase_counts"].items():
                if ph < max_phases:
                    matrix[ri, ph] = cnt / max(total, 1)

        fig, ax = plt.subplots(figsize=(max(6, max_phases * 1.2), len(pi) * 0.7 + 1.5))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(max_phases))
        ax.set_xticklabels([f"Phase {i}" for i in range(max_phases)], fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        plt.colorbar(im, ax=ax, label="Fraction of steps")
        for ri in range(len(pi)):
            for ci in range(max_phases):
                v = matrix[ri, ci]
                ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="black" if v < 0.5 else "white")
        ax.set_title("Phase Usage Distribution (fraction of steps per intersection)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase_heatmap.png"), dpi=150)
        plt.close()
        print(f"  Chart saved → {out_dir}/phase_heatmap.png")

        # ── Chart 3: NES over episodes ───────────────────────────────────────
        if "episode" in df.columns and df["episode"].nunique() > 1:
            ep_nes = df.groupby("episode")["nes"].mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ep_nes.index, ep_nes.values, color="#1565c0",
                    linewidth=2, marker="o", markersize=4)
            z = np.polyfit(ep_nes.index, ep_nes.values, 1)
            trend_y = np.poly1d(z)(ep_nes.index)
            ax.plot(ep_nes.index, trend_y, "--", color="#e53935",
                    linewidth=1.5, label=f"Trend ({z[0]:+.5f}/ep)")
            ax.axhline(CFG.good_nes_threshold, color="#43a047",
                       linestyle=":", linewidth=1, label="Good threshold (0.50)")
            ax.axhline(CFG.warn_nes_threshold, color="#fb8c00",
                       linestyle=":", linewidth=1, label="Warning threshold (0.25)")
            ax.set_title("Network Efficiency Score per Episode", fontsize=13, fontweight="bold")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean NES")
            ax.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "nes_episodes.png"), dpi=150)
            plt.close()
            print(f"  Chart saved → {out_dir}/nes_episodes.png")

        # ── Chart 4: Network impact of decisions (delta NES) ─────────────────
        mistakes  = [d for d in self.analysis["all_decisions"] if d["verdict"] == "MISTAKE"]
        good      = [d for d in self.analysis["all_decisions"] if d["verdict"] == "GOOD"]
        fig, ax   = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-0.3, 0.3, 31)
        ax.hist([d["delta_nes"] for d in good],    bins=bins, alpha=0.6,
                color="#43a047", label="GOOD decisions")
        ax.hist([d["delta_nes"] for d in mistakes], bins=bins, alpha=0.6,
                color="#e53935", label="MISTAKE decisions")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Network NES Impact of GOOD vs MISTAKE Decisions",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("ΔNetwork NES (post - pre decision)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "decision_impact.png"), dpi=150)
        plt.close()
        print(f"  Chart saved → {out_dir}/decision_impact.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="MAPPO Agent Decision Analyzer — "
                    "reads dashboard CSV export and analyzes agent behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze full session:
    python agent_analyzer.py --csv session.csv

  Analyze only MAPPO mode from episode 5 onward:
    python agent_analyzer.py --csv session.csv --mode MAPPO --episodes 5

  Focus on one intersection:
    python agent_analyzer.py --csv session.csv --intersection Gomharia

  Full report + save JSON + generate charts:
    python agent_analyzer.py --csv session.csv --report full --json out.json --plots plots/
        """,
    )
    p.add_argument("--csv",          required=True,          help="Path to exported CSV file")
    p.add_argument("--mode",         default=None,           help="Filter by mode: MAPPO or FIXED")
    p.add_argument("--episodes",     type=int, default=None, help="Analyze from this episode onward")
    p.add_argument("--intersection", default=None,           help="Focus on one intersection by name or ID")
    p.add_argument("--report",       choices=["brief","full"], default="full",
                   help="brief = summary table only; full = + top mistakes (default: full)")
    p.add_argument("--json",         default=None,           help="Save JSON report to this path")
    p.add_argument("--plots",        default=None,           help="Save charts to this directory")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    analyzer = (
        AgentAnalyzer(
            csv_path             = args.csv,
            mode_filter          = args.mode,
            episode_filter       = args.episodes,
            intersection_filter  = args.intersection,
        )
        .load()
        .analyze()
    )

    analyzer.print_report(level=args.report)

    if args.json:
        analyzer.save_json(args.json)

    if args.plots:
        analyzer.plot(out_dir=args.plots)


if __name__ == "__main__":
    main()
