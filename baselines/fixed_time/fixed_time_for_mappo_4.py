import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field

# ── SUMO setup ──────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable.\n"
        "Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo"
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


# ── Project paths (resolved relative to this file, not hardcoded) ──
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))          # baselines/fixed_time/
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(_BASELINE_DIR))    # repository root/
DEFAULT_SUMO_CFG = os.path.join(_PROJECT_ROOT, "sumo", "project (2).sumocfg")
DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


# ╔═══════════════════════════════════════════════════════════════╗
# ║                    1. CONFIGURATION                          ║
# ╚═══════════════════════════════════════════════════════════════╝

@dataclass
class BaselineConfig:
    """Configuration for the Fixed-Time baseline — mirrors MAPPOConfig exactly."""

    # ── Simulation Schedule (ALIGNED with MAPPO) ─────────────
    decision_interval: int = 5        # [ALIGNED] same as MAPPO decision_interval
    total_steps:       int = 20_000  # [FIX] was 200_000 → must equal MAPPO total_steps
    log_interval:      int = 100      # print stats every N sim steps

    # ── Normalization Constants (same as MAPPO) ──────────────
    max_cars:   float = 300.0
    max_wait:   float = 600.0
    max_spikes: float = 50.0
    max_co2:    float = 5_000.0
    max_jam:    float = 100.0
    max_speed:  float = 13.89   # ~50 km/h

    # ── Vehicle Types (ALIGNED with MAPPO) ───────────────────
    emergency_types: frozenset = field(
        default_factory=lambda: frozenset({
            "emergency", "ambulance", "fire", "fire_department",
            "authority", "police", "emergency vehicle",
        })
    )
    heavy_types: frozenset = field(
        # [FIX] "buks" was a typo — corrected to "bus"; added "coach","delivery"
        default_factory=lambda: frozenset({
            "truck", "bus", "trailer", "coach", "delivery",
        })
    )

    # ═════════════════════════════════════════════════════════
    #  ★ Defaults resolve relative to the repository root. Override
    #    with --sumo-cfg / --results-save-path, or edit these two lines.
    # ═════════════════════════════════════════════════════════
    sumo_cfg:          str   = DEFAULT_SUMO_CFG
    results_save_path: str   = DEFAULT_RESULTS_DIR
    # [FIX] was 0.1 → must equal MAPPO sim_step_length so all time-based
    # metrics (waiting, delay, trip time) are on the same scale
    sim_step_length:   float = 0.1
    use_gui:           bool  = False
    # [FIX] Added seed — must match MAPPO seed so both runs see identical traffic
    seed:              int   = 42
    # ═════════════════════════════════════════════════════════

    def __post_init__(self):
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.makedirs(self.results_save_path, exist_ok=True)
        sim_seconds = self.total_steps * self.sim_step_length
        print(f"[BASELINE CONFIG] total_steps      : {self.total_steps:,}")
        print(f"[BASELINE CONFIG] sim_step_length  : {self.sim_step_length} s")
        print(f"[BASELINE CONFIG] sim duration     : {sim_seconds:,.0f} s "
              f"({sim_seconds/3600:.2f} h)")
        print(f"[BASELINE CONFIG] seed             : {self.seed}")
        print(f"[BASELINE CONFIG] results_save_path: {self.results_save_path}")


# ╔═══════════════════════════════════════════════════════════════╗
# ║             2. LANE-LEVEL METRIC HELPERS                     ║
# ║         (identical to MAPPO_CTDE_Unified.py)                 ║
# ╚═══════════════════════════════════════════════════════════════╝

def _get_lane_vehicles(lanes):
    """Get all vehicle IDs on the given lanes."""
    vehs = []
    for lane in lanes:
        try:
            vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
        except Exception:
            pass
    return vehs


def _get_halting_count(lanes):
    total = 0
    for lane in lanes:
        try:
            total += traci.lane.getLastStepHaltingNumber(lane)
        except Exception:
            pass
    return total


def _get_vehicle_count(lanes):
    total = 0
    for lane in lanes:
        try:
            total += traci.lane.getLastStepVehicleNumber(lane)
        except Exception:
            pass
    return total


def _get_occupancy(lanes):
    if not lanes:
        return 0.0
    occs = []
    for lane in lanes:
        try:
            occs.append(traci.lane.getLastStepOccupancy(lane))
        except Exception:
            pass
    return float(np.mean(occs)) / 100.0 if occs else 0.0


def _get_mean_speed(lanes):
    if not lanes:
        return 0.0
    speeds = []
    for lane in lanes:
        try:
            speeds.append(traci.lane.getLastStepMeanSpeed(lane))
        except Exception:
            pass
    return float(np.mean(speeds)) if speeds else 0.0


def _get_waiting_time_raw(vehs):
    """Total waiting time (seconds) for a list of vehicles."""
    wt = 0.0
    for v in vehs:
        try:
            wt += traci.vehicle.getWaitingTime(v)
        except Exception:
            pass
    return wt


def _get_delay_raw(vehs):
    """Mean delay (seconds) for a list of vehicles."""
    delays = []
    for v in vehs:
        try:
            actual = traci.vehicle.getAccumulatedWaitingTime(v)
            ideal = traci.vehicle.getDistance(v) / max(traci.vehicle.getMaxSpeed(v), 0.1)
            delays.append(max(0.0, actual - ideal))
        except Exception:
            pass
    return float(np.mean(delays)) if delays else 0.0


def _get_co2_raw(vehs):
    co2 = 0.0
    for v in vehs:
        try:
            co2 += traci.vehicle.getCO2Emission(v)
        except Exception:
            pass
    return co2


def _get_accel_spikes(vehs, threshold=2.5):
    spikes = 0
    for v in vehs:
        try:
            if abs(traci.vehicle.getAcceleration(v)) > threshold:
                spikes += 1
        except Exception:
            pass
    return spikes


# ╔═══════════════════════════════════════════════════════════════╗
# ║             3. FIXED-TIME ENVIRONMENT                        ║
# ╚═══════════════════════════════════════════════════════════════╝

class FixedTimeEnv:
    """
    SUMO environment that runs with DEFAULT fixed-time signal programs.
    No RL actions — signals follow their original timing from the .net.xml.
    Collects the SAME metrics as SUMOMultiAgentEnv for fair comparison.
    """

    def __init__(self, config: BaselineConfig):
        self.cfg = config
        self.tls_ids = []
        self.num_agents = 0
        self.agent_info = {}
        self.sim_step = 0
        self._trip_log = {}
        self._completed_trips = []   # [FIX] stores durations of COMPLETED trips (like MAPPO)
        self._is_open = False

    # ── START ────────────────────────────────────────────────

    def start(self):
        """Start SUMO and discover traffic lights & detectors."""
        if self._is_open:
            try:
                traci.close()
            except Exception:
                pass

        sumo_binary = "sumo-gui.exe" if self.cfg.use_gui else "sumo.exe"
        binary_path = os.path.join(os.environ["SUMO_HOME"], "bin", sumo_binary)
        sumo_cmd = [
            binary_path,
            "-c", self.cfg.sumo_cfg,
            "--step-length",          str(self.cfg.sim_step_length),
            "--delay",                "0",
            "--lateral-resolution",   "0",
            # [FIX] Pass seed → identical traffic demand every run (matches MAPPO)
            "--seed",                 str(self.cfg.seed),
            # [FIX] Added to match MAPPO's sumo_cmd exactly
            "--ignore-route-errors",
            "--no-warnings",
            "--error-log",            os.path.join(self.cfg.results_save_path, "sumo_errors.log"),
            "--duration-log.disable",
            "--no-step-log",
            "--collision.action",     "warn",
            "--time-to-teleport",     "300",
            "--time-to-teleport.highways", "0",
        ]
        traci.start(sumo_cmd)
        self._is_open = True
        self.sim_step = 0
        self._trip_log = {}
        self._completed_trips = []   # [FIX] reset on every start

        # ── Discover TLS and detectors ──────────────────────
        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.num_agents = len(self.tls_ids)
        detector_ids = list(traci.lanearea.getIDList())

        print(f"[ENV] Found {self.num_agents} traffic lights: "
              f"{[t[:30] for t in self.tls_ids]}")
        print(f"[ENV] Found {len(detector_ids)} detectors")

        # ── Build mappings ──────────────────────────────────
        tls_det_map = self._build_detector_map(detector_ids)
        det_to_lane = self._build_det_to_lane_map(detector_ids)

        # ── Build per-agent info ────────────────────────────
        self.agent_info = {}

        for i, tls_id in enumerate(self.tls_ids):
            tls_key = f"TLS_{i + 1}"
            dets = tls_det_map.get(tls_key, [])
            # [FIX] MAPPO never falls back to ALL detectors — it uses an empty
            # lane list and then merges incoming lanes; do the same here
            lanes = self._get_lanes_for_detectors(dets, det_to_lane)
            # [FIX] Add TLS approach lanes not covered by detectors (mirrors MAPPO)
            lanes = self._merge_incoming_lanes(tls_id, lanes)
            num_phases = len(
                traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
            )

            self.agent_info[tls_id] = {
                'index': i,
                'lanes': lanes,
                'num_phases': num_phases,
            }
            print(f"  {tls_id[:35]} → {len(lanes)} lanes, {num_phases} phases (FIXED)")

        # ── Print the fixed-time program for each TLS ───────
        print(f"\n[FIXED-TIME] Signal programs:")
        for tls_id in self.tls_ids:
            programs = traci.trafficlight.getAllProgramLogics(tls_id)
            if programs:
                phases = programs[0].phases
                durations = [f"{p.duration:.0f}s" for p in phases]
                print(f"  {tls_id[:35]} → phases: {durations}")

        return {
            'num_agents': self.num_agents,
            'tls_ids': self.tls_ids,
        }

    # ── RUN SIMULATION ───────────────────────────────────────

    def run(self):
        """
        Run the full simulation with fixed-time signals.
        Collect metrics at the same intervals as MAPPO.
        """
        cfg = self.cfg

        # ── Metric accumulators ──────────────────────────────
        agent_data = {
            tls_id: {
                'wt': [], 'q': [], 'th': [], 'd': [],
                'co2': [], 'spd': [], 'occ': [], 'nveh': [],
            }
            for tls_id in self.tls_ids
        }

        # ── Logging lists (same format as MAPPO trainer) ─────
        log_steps = []
        log_queue = []
        log_wt = []
        log_delay = []
        log_trip = []
        log_co2 = []
        log_speed = []

        start_time = time.time()
        log_counter = 0

        print(f"\n[FIXED-TIME] Running simulation — {cfg.total_steps} steps "
              f"(NO RL, default signal timings)...\n")

        for sim_step in range(cfg.total_steps):
            traci.simulationStep()
            self.sim_step = sim_step + 1
            self._update_trip_log()

            # ── Collect metrics at decision intervals ────────
            if (sim_step + 1) % cfg.decision_interval == 0:
                log_counter += 1

                metrics = self._get_per_agent_metrics()

                for tls_id in self.tls_ids:
                    m = metrics[tls_id]
                    ad = agent_data[tls_id]
                    ad['wt'].append(m['waiting_time'])
                    ad['q'].append(m['queue_length'])
                    ad['th'].append(m['throughput'])
                    ad['d'].append(m['delay'])
                    ad['co2'].append(m['co2'])
                    ad['spd'].append(m['speed'])
                    ad['occ'].append(m['occupancy'])
                    ad['nveh'].append(m['num_vehicles'])

                # ── Periodic logging ─────────────────────────
                actual_step = sim_step + 1
                if actual_step % cfg.log_interval == 0:
                    # [FIX] Use mean() per agent — matches MAPPO's _log() exactly
                    # (was sum() → gave values 9× larger than MAPPO)
                    avg_queue = float(np.mean([m['queue_length'] for m in metrics.values()]))
                    avg_wt    = float(np.mean([m['waiting_time'] for m in metrics.values()]))
                    avg_delay = float(np.mean([m['delay']        for m in metrics.values()]))
                    avg_co2   = float(np.mean([m['co2']          for m in metrics.values()]))
                    avg_speed = float(np.mean([m['speed']        for m in metrics.values()]))
                    avg_thru  = float(np.mean([m['throughput']   for m in metrics.values()]))
                    trip_time = self._get_mean_trip_time()
                    pct       = 100.0 * actual_step / cfg.total_steps

                    log_steps.append(actual_step)
                    log_queue.append(avg_queue)
                    log_wt.append(avg_wt)
                    log_delay.append(avg_delay)
                    log_trip.append(trip_time)
                    log_co2.append(avg_co2)
                    log_speed.append(avg_speed)

                    # [FIX] Print format mirrors MAPPO _log() for side-by-side comparison
                    print(
                        f"[{pct:5.1f}%] Step {actual_step:7,d} | "
                        f"Q/int {avg_queue:5.1f} | "
                        f"W/int {avg_wt:7.1f}s | "
                        f"D/int {avg_delay:5.1f}s | "
                        f"T {trip_time:5.1f}s | "
                        f"Thru/int {avg_thru:4.1f} | "
                        f"CO2/int {avg_co2:7.0f} | "
                        f"Spd {avg_speed:4.1f} | "
                        f"[FIXED-TIME]"
                    )

            # ── Check if simulation ended ────────────────────
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"\n[FIXED-TIME] Simulation ended at step {sim_step + 1} "
                      f"(no more vehicles)")
                break

        elapsed = time.time() - start_time
        print(f"\n[FIXED-TIME] Simulation complete. Time: {elapsed:.0f}s")

        return {
            'agent_data': agent_data,
            'log_steps': log_steps,
            'log_queue': log_queue,
            'log_wt': log_wt,
            'log_delay': log_delay,
            'log_trip': log_trip,
            'log_co2': log_co2,
            'log_speed': log_speed,
            'elapsed': elapsed,
        }

    # ── CLOSE ────────────────────────────────────────────────

    def close(self):
        if self._is_open:
            try:
                traci.close()
            except Exception:
                pass
            self._is_open = False

    # ── METRICS ──────────────────────────────────────────────

    def _get_per_agent_metrics(self):
        """Get detailed metrics for each intersection (same as MAPPO)."""
        metrics = {}
        for tls_id in self.tls_ids:
            ai = self.agent_info[tls_id]
            lanes = ai['lanes']
            vehs = _get_lane_vehicles(lanes)

            metrics[tls_id] = {
                'waiting_time':  _get_waiting_time_raw(vehs),
                'queue_length':  _get_halting_count(lanes),
                'throughput':    _get_vehicle_count(lanes),
                'delay':         _get_delay_raw(vehs),
                'co2':           _get_co2_raw(vehs),
                'speed':         _get_mean_speed(lanes),
                'occupancy':     _get_occupancy(lanes),
                'num_vehicles':  len(vehs),
            }
        return metrics

    def _get_mean_trip_time(self):
        """[FIX] Mean trip time of COMPLETED trips only — mirrors MAPPO's get_mean_trip_time().
        Old version averaged (now - departure) for vehicles still in network → wrong metric."""
        if not self._completed_trips:
            return 0.0
        return float(np.mean(self._completed_trips))

    # ── INTERNAL HELPERS ─────────────────────────────────────

    def _update_trip_log(self):
        """[FIX] Record duration of completed trips — mirrors MAPPO's _update_trip_log().
        Old version popped the vehicle without recording duration → trip time was lost."""
        t = self.sim_step * self.cfg.sim_step_length
        for v in traci.simulation.getDepartedIDList():
            self._trip_log[v] = t
        for v in traci.simulation.getArrivedIDList():
            if v in self._trip_log:
                dur = t - self._trip_log.pop(v)
                if dur > 0:
                    self._completed_trips.append(dur)

    @staticmethod
    def _build_detector_map(detector_ids):
        """[FIX] Added regex fallback — mirrors MAPPO's _build_detector_map exactly.
        If detectors don't start with 'det_TLS_', the regex catches other naming conventions."""
        import re
        tls_det_map = defaultdict(list)
        vc = 0
        for det_id in detector_ids:
            if not det_id.startswith("det_TLS_"):
                continue
            rest  = det_id[len("det_"):]
            parts = rest.split("det_")
            for part in parts:
                tokens = part.split("_")
                if len(tokens) >= 2 and tokens[0] == "TLS":
                    tls_key = f"TLS_{tokens[1]}"
                    tls_det_map[tls_key].append(det_id)
                    vc += 1
        if vc > 0:
            return dict(tls_det_map)
        # Regex fallback — handles TLS_1, TLS-1, tls1, etc.
        for det_id in detector_ids:
            match = re.search(r'[Tt][Ll][Ss][_\-]?(\d+)', det_id)
            if match:
                tls_det_map[f"TLS_{match.group(1)}"].append(det_id)
                vc += 1
        if vc > 0:
            print(f"[INFO] Detector map: regex fallback matched {vc} detectors")
        elif detector_ids:
            print(f"[WARN] No detectors matched any naming convention. "
                  f"Sample: {detector_ids[:3]}")
        return dict(tls_det_map)

    @staticmethod
    def _build_det_to_lane_map(detector_ids):
        det_to_lane = {}
        for det_id in detector_ids:
            try:
                det_to_lane[det_id] = traci.lanearea.getLaneID(det_id)
            except Exception:
                det_to_lane[det_id] = None
        return det_to_lane

    @staticmethod
    def _get_lanes_for_detectors(detectors, det_to_lane):
        lanes = []
        for d in detectors:
            lane = det_to_lane.get(d)
            if lane and lane not in lanes:
                lanes.append(lane)
        return lanes

    @staticmethod
    def _merge_incoming_lanes(tls_id, existing_lanes):
        """[FIX] Identical to MAPPO's _merge_incoming_lanes().
        Adds TLS approach lanes not covered by any detector span so every
        approaching vehicle is counted — critical for fair metric comparison."""
        merged = list(existing_lanes)
        try:
            links = traci.trafficlight.getControlledLinks(tls_id)
            for link_list in links:
                for link in link_list:
                    if len(link) >= 1 and link[0] not in merged:
                        merged.append(link[0])
        except Exception:
            pass
        return merged


# ╔═══════════════════════════════════════════════════════════════╗
# ║             4. EVALUATOR & RANKING                           ║
# ║         (same format as MAPPO IntersectionEvaluator)         ║
# ╚═══════════════════════════════════════════════════════════════╝

class BaselineEvaluator:
    """Evaluates Fixed-Time results with same format as MAPPO evaluator."""

    def __init__(self, config: BaselineConfig):
        self.cfg = config

    def compute_results(self, env, sim_results):
        """Compute ranking and NES from simulation results."""
        agent_data = sim_results['agent_data']

        # ── Per-agent summaries ──────────────────────────────
        agent_summaries = {}
        for tls_id in env.tls_ids:
            ad = agent_data[tls_id]
            agent_summaries[tls_id] = {
                'mean_waiting':    float(np.mean(ad['wt'])) if ad['wt'] else 0.0,
                'mean_queue':      float(np.mean(ad['q'])) if ad['q'] else 0.0,
                'mean_throughput': float(np.mean(ad['th'])) if ad['th'] else 0.0,
                'mean_delay':      float(np.mean(ad['d'])) if ad['d'] else 0.0,
                'mean_co2':        float(np.mean(ad['co2'])) if ad['co2'] else 0.0,
                'mean_speed':      float(np.mean(ad['spd'])) if ad['spd'] else 0.0,
            }

        # ── Compute ranking ─────────────────────────────────
        ranking = []
        for tls_id, s in agent_summaries.items():
            score = s['mean_throughput'] / (1.0 + s['mean_delay'])
            ranking.append({
                'tls_id':     tls_id,
                'score':      score,
                'waiting':    s['mean_waiting'],
                'queue':      s['mean_queue'],
                'throughput': s['mean_throughput'],
                'delay':      s['mean_delay'],
                'co2':        s['mean_co2'],
                'speed':      s['mean_speed'],
            })
        ranking.sort(key=lambda x: x['score'], reverse=True)

        # ── Network Efficiency Score ─────────────────────────
        nes = float(np.mean([r['score'] for r in ranking])) if ranking else 0.0

        trip_time = env._get_mean_trip_time()

        return {
            'agent_summaries': agent_summaries,
            'ranking': ranking,
            'nes': nes,
            'trip_time': trip_time,
        }

    # ── Print ranking table (same format as MAPPO) ───────────

    def print_ranking_table(self, results, label="Fixed-Time Baseline"):
        ranking = results['ranking']
        nes = results['nes']
        trip_time = results['trip_time']

        print(f"\n{'=' * 100}")
        print(f"  {label} — INTERSECTION RANKING")
        print(f"{'=' * 100}")
        print(f"  {'Rank':<6} {'TLS ID':<35} {'Wait(s)':<10} {'Queue':<8} "
              f"{'Thru':<8} {'Delay(s)':<10} {'Speed':<8} {'Score':<8}")
        print(f"  {'-' * 94}")

        for i, r in enumerate(ranking):
            rank_label = "★ BEST" if i == 0 else (
                "✗ WORST" if i == len(ranking) - 1 else f"  #{i+1}")
            tls_short = r['tls_id'][:33]
            print(
                f"  {rank_label:<6} {tls_short:<35} "
                f"{r['waiting']:>8.1f}  {r['queue']:>6.1f}  "
                f"{r['throughput']:>6.1f}  {r['delay']:>8.1f}  "
                f"{r['speed']:>6.1f}  {r['score']:>6.4f}"
            )

        print(f"  {'-' * 94}")
        print(f"  Network Efficiency Score (NES) : {nes:.4f}")
        print(f"  Mean Trip Time                 : {trip_time:.1f}s")
        print(f"{'=' * 100}\n")

    # ── Plot simulation metrics (same format as MAPPO) ───────

    @staticmethod
    def plot_simulation_curves(sim_results, save_path):
        """Generate 4-panel visualization matching MAPPO training plots."""
        steps = sim_results['log_steps']
        if not steps:
            print("[EVAL] No logs to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle("FIXED-TIME Baseline — Simulation Results (No RL)",
                     fontsize=15, fontweight='bold')

        # Panel 1: Queue Length
        axes[0, 0].plot(steps, sim_results['log_queue'],
                        color='#FF9800', linewidth=1.5)
        axes[0, 0].set_title("Total Queue (All Intersections)", fontsize=12)
        axes[0, 0].set_xlabel("Simulation Step")
        axes[0, 0].set_ylabel("Halting Vehicles")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(steps, sim_results['log_queue'],
                                alpha=0.1, color='#FF9800')

        # Panel 2: CO2 Emissions
        axes[0, 1].plot(steps, sim_results['log_co2'],
                        color='#795548', linewidth=1.5)
        axes[0, 1].set_title("Total CO2 Emissions", fontsize=12)
        axes[0, 1].set_xlabel("Simulation Step")
        axes[0, 1].set_ylabel("CO2 (mg)")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Waiting Time & Delay
        axes[1, 0].plot(steps, sim_results['log_wt'], color='#F44336',
                        linewidth=1.5, label="Total Waiting Time")
        axes[1, 0].plot(steps, sim_results['log_delay'], color='#9C27B0',
                        linewidth=1.5, label="Mean Delay")
        axes[1, 0].set_title("Waiting Time & Delay", fontsize=12)
        axes[1, 0].set_xlabel("Simulation Step")
        axes[1, 0].set_ylabel("Seconds")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: Mean Trip Time
        axes[1, 1].plot(steps, sim_results['log_trip'],
                        color='#4CAF50', linewidth=1.5)
        axes[1, 1].set_title("Mean Trip Time", fontsize=12)
        axes[1, 1].set_xlabel("Simulation Step")
        axes[1, 1].set_ylabel("Seconds")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(save_path, "fixedtime_simulation_results.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[PLOT] Simulation curves saved to {out_path}")

    # ── Plot intersection comparison (same as MAPPO) ─────────

    @staticmethod
    def plot_intersection_comparison(ranking, save_path):
        """Bar chart comparing all intersections (same format as MAPPO)."""
        if not ranking:
            return

        tls_names = [r['tls_id'][:20] for r in ranking]
        scores = [r['score'] for r in ranking]
        delays = [r['delay'] for r in ranking]
        throughputs = [r['throughput'] for r in ranking]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("FIXED-TIME Baseline — Per-Intersection Comparison",
                     fontsize=14, fontweight='bold')

        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, len(ranking)))

        # Score
        axes[0].barh(tls_names, scores, color=colors)
        axes[0].set_title("Efficiency Score (higher = better)")
        axes[0].set_xlabel("Score")
        axes[0].invert_yaxis()

        # Delay
        colors_delay = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(ranking)))
        sorted_by_delay = sorted(range(len(delays)), key=lambda i: delays[i])
        axes[1].barh(
            [tls_names[i] for i in sorted_by_delay],
            [delays[i] for i in sorted_by_delay],
            color=[colors_delay[j] for j in range(len(sorted_by_delay))]
        )
        axes[1].set_title("Mean Delay (lower = better)")
        axes[1].set_xlabel("Seconds")
        axes[1].invert_yaxis()

        # Throughput
        sorted_by_thru = sorted(range(len(throughputs)),
                                key=lambda i: throughputs[i], reverse=True)
        axes[2].barh(
            [tls_names[i] for i in sorted_by_thru],
            [throughputs[i] for i in sorted_by_thru],
            color=plt.cm.Blues(np.linspace(0.8, 0.3, len(ranking)))
        )
        axes[2].set_title("Mean Throughput (higher = better)")
        axes[2].set_xlabel("Vehicles")
        axes[2].invert_yaxis()

        plt.tight_layout()
        out_path = os.path.join(save_path, "fixedtime_intersection_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[PLOT] Intersection comparison saved to {out_path}")


# ╔═══════════════════════════════════════════════════════════════╗
# ║                      5. MAIN                                 ║
# ╚═══════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-Time Baseline for MAPPO Comparison"
    )
    parser.add_argument("--sumo-cfg", type=str, default=None,
                        help="Path to SUMO .sumocfg file")
    parser.add_argument("--results-save-path", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total simulation steps")
    parser.add_argument("--use-gui", action="store_true",
                        help="Use SUMO GUI instead of headless")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build config ─────────────────────────────────────────
    config = BaselineConfig()

    if args.sumo_cfg:
        config.sumo_cfg = args.sumo_cfg
    if args.results_save_path:
        config.results_save_path = args.results_save_path
    if args.total_steps:
        config.total_steps = args.total_steps
    if args.use_gui:
        config.use_gui = True

    # ── Print header ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FIXED-TIME BASELINE")
    print(f"  Traffic Signal Control WITHOUT RL")
    print(f"{'=' * 60}")
    print(f"  ● Using default SUMO signal programs")
    print(f"  ● No reinforcement learning")
    print(f"  ● No training — just simulation")
    print(f"  ● Same metrics as MAPPO-CTDE for comparison")
    print(f"{'=' * 60}\n")

    # ── Create environment & run ─────────────────────────────
    env = FixedTimeEnv(config)
    evaluator = BaselineEvaluator(config)

    env.start()
    sim_results = env.run()

    # ── Compute ranking & NES ────────────────────────────────
    eval_results = evaluator.compute_results(env, sim_results)

    # ── Print ranking table ──────────────────────────────────
    evaluator.print_ranking_table(eval_results, label="Fixed-Time Baseline")

    # ── Plot simulation curves ───────────────────────────────
    evaluator.plot_simulation_curves(sim_results, config.results_save_path)

    # ── Plot intersection comparison ─────────────────────────
    evaluator.plot_intersection_comparison(
        eval_results['ranking'], config.results_save_path
    )

    # ── Print final summary ──────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  FIXED-TIME BASELINE — FINAL SUMMARY")
    print(f"{'=' * 65}")
    if sim_results['log_delay']:
        avg_delay = np.mean(sim_results['log_delay'])
        avg_wt    = np.mean(sim_results['log_wt'])
        avg_queue = np.mean(sim_results['log_queue'])
        avg_trip  = np.mean(sim_results['log_trip'])
        avg_co2   = np.mean(sim_results['log_co2'])
        avg_speed = np.mean(sim_results['log_speed'])
        # [FIX] All averages are per-agent (mean), matching MAPPO's _log()
        print(f"  Mean Delay         : {avg_delay:>10.2f} s   (avg/intersection)")
        print(f"  Mean Waiting Time  : {avg_wt:>10.2f} s   (avg/intersection)")
        print(f"  Mean Queue Length  : {avg_queue:>10.1f} veh (avg/intersection)")
        print(f"  Mean CO2           : {avg_co2:>10.0f}     (avg/intersection)")
        print(f"  Mean Speed         : {avg_speed:>10.2f} m/s")
    else:
        print("  (no log entries recorded)")
    # [FIX] Use completed trips only — matches MAPPO's get_mean_trip_time()
    trip_time_final = env._get_mean_trip_time()
    n_completed     = len(env._completed_trips)
    print(f"  Mean Trip Time     : {trip_time_final:>10.2f} s   ({n_completed} completed trips)")
    print(f"  Network Efficiency : {eval_results['nes']:>10.4f}")
    print(f"  Best Intersection  : {eval_results['ranking'][0]['tls_id']}")
    print(f"  Worst Intersection : {eval_results['ranking'][-1]['tls_id']}")
    print(f"  Simulation Time    : {sim_results['elapsed']:>10.0f} s")
    print(f"  Seed used          : {config.seed}  ← use same value for MAPPO eval")
    print(f"{'=' * 65}")

    # ── Close ────────────────────────────────────────────────
    env.close()
    print(f"\n[DONE] Results saved to: {config.results_save_path}")
    print(f"[INFO] Compare these results with MAPPO_CTDE_Unified.py output!")


if __name__ == "__main__":
    main()