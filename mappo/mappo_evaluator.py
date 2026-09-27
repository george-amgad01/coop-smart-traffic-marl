"""
MAPPO Checkpoint Evaluator
==========================
Loads a saved MAPPO checkpoint and runs evaluation (greedy actions,
no training) for 50,000 simulation steps.  Prints the same metrics
as fixed_time_baseline.py for a fair comparison.

Usage:
    python mappo_evaluator.py
    python mappo_evaluator.py --checkpoint saved_models/mappo_step_200000.pt
    python mappo_evaluator.py --checkpoint saved_models/mappo_final.pt --steps 100000
    python mappo_evaluator.py --gui
"""

import os, sys, time, math, argparse
import numpy as np
import torch
from torch.amp import autocast

# ── SUMO setup ──────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ── Project paths (resolved relative to this file, not hardcoded) ──
_EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))      # mappo/
_PROJECT_ROOT  = os.path.dirname(_EVALUATOR_DIR)                 # repository root/
DEFAULT_SUMO_CFG = os.path.join(_PROJECT_ROOT, "sumo", "project (2).sumocfg")
DEFAULT_SAVE_DIR = os.path.join(_EVALUATOR_DIR, "saved_models")

# ── Re-use classes and helpers from MAPPO ────────────────────
from mappo_optimized_4 import (
    MAPPOConfig,
    MAPPONetwork,
    SUMOMultiAgentEnv,
    RewardComputer,
    VehicleCache,
    _get_lane_vehicles,
    _get_halting_count,
    _get_vehicle_count,
    _get_mean_speed,
    _get_waiting_time_raw,
    _get_delay_raw,
    _get_co2_raw,
)


# ─────────────────────────────────────────────────────────────
# MAPPO Evaluator
# ─────────────────────────────────────────────────────────────
class MAPPOEvaluator:
    def __init__(self, cfg, checkpoint_path, eval_steps=50_000):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.eval_steps = eval_steps
        self.device = cfg.get_device()

    def run(self):
        cfg = self.cfg
        dev = self.device

        # ── 1. Start environment & discover network topology ──
        env = SUMOMultiAgentEnv(cfg)
        obs_all, info = env.reset()

        N = info['num_agents']
        mp = info['max_phases']
        pc = info['agent_phase_counts']
        vgl = info.get('valid_greens_list', None)
        tls_ids = env.tls_ids

        print(f"[EVALUATOR] {N} agents, max_phases={mp}")

        # ── 2. Build network & load checkpoint ───────────────
        network = MAPPONetwork(cfg, N, mp, pc, valid_greens_list=vgl).to(dev)

        ck = torch.load(self.checkpoint_path, map_location=dev)
        if 'phase_masks' in ck['network']:
            del ck['network']['phase_masks']
        network.load_state_dict(ck['network'], strict=False)
        print(f"[EVALUATOR] Loaded checkpoint: {self.checkpoint_path}")

        # Load observation normalizer stats
        if 'obs_norm' in ck:
            env.feature_extractor.obs_normalizer.load_state_dict(ck['obs_norm'])
            print("[EVALUATOR] Loaded observation normalizer stats")

        # Load reward computer stats (for fair reward comparison)
        if 'reward_computers' in ck:
            for tls_id, rc_state in ck['reward_computers'].items():
                if tls_id in env.reward_computers:
                    env.reward_computers[tls_id].load_state_dict(rc_state)
            print("[EVALUATOR] Loaded reward computer stats")

        # ── 3. Run evaluation (greedy, no exploration) ────────
        network.eval()
        adj = env.adj_matrix.to(dev, non_blocking=True)
        h = network.init_hidden(dev)

        # Metric accumulators
        log_delay = []
        log_wt = []
        log_queue = []
        log_trip = []
        cumulative_reward = 0.0
        total_co2 = 0.0  # Raw sum of ALL vehicle CO2 across ALL steps

        # Per-intersection accumulators
        agent_data = {t: {'wt': [], 'q': [], 'th': [], 'd': [],
                          'co2': [], 'spd': [], 'rew': []}
                      for t in tls_ids}

        decision_interval = cfg.decision_interval
        log_interval = max(1000 // decision_interval, 1)
        sim_step = 0
        dec_step = 0
        _trip_log = {}
        t0 = time.time()

        ck_name = os.path.basename(self.checkpoint_path)
        print(f"\n[EVALUATOR] Running {self.eval_steps:,} steps "
              f"(checkpoint: {ck_name}) ...\n")

        with torch.no_grad():
            while sim_step < self.eval_steps:
                # ── Get greedy actions from network ──────────
                obs_norm = env.feature_extractor.normalize_frozen(obs_all)
                obs_t = torch.from_numpy(obs_norm).to(dev, non_blocking=True)

                with autocast('cuda', enabled=(cfg.use_amp and dev.type == 'cuda')):
                    logits, _, h, _ = network(obs_t, adj, h)

                # Greedy: argmax (no sampling, no exploration)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

                # ── Step the environment ─────────────────────
                obs_all, rewards, done, step_info = env.step(actions)
                sim_step += decision_interval
                dec_step += 1

                # ── Accumulate total CO2 from ALL vehicles ───
                total_co2 += step_info.get('step_total_co2', 0.0)

                # ── Update trip log ──────────────────────────
                t_now = sim_step * cfg.sim_step_length
                for v in traci.simulation.getDepartedIDList():
                    _trip_log[v] = t_now
                for v in traci.simulation.getArrivedIDList():
                    _trip_log.pop(v, None)

                # ── Collect per-agent metrics (same as MAPPO) ─
                m = env.get_per_agent_metrics()
                for i, tls_id in enumerate(tls_ids):
                    ad = agent_data[tls_id]
                    ad['wt'].append(m[tls_id]['waiting_time'])
                    ad['q'].append(m[tls_id]['queue_length'])
                    ad['th'].append(m[tls_id]['throughput'])
                    ad['d'].append(m[tls_id]['delay'])
                    ad['co2'].append(m[tls_id]['co2'])
                    ad['spd'].append(m[tls_id]['speed'])
                    
                    # Store Raw Reward for fair comparison
                    raw_r = env.reward_computers[tls_id].last_raw_reward
                    ad['rew'].append(raw_r)
                    
                cumulative_reward += sum([env.reward_computers[t].last_raw_reward for t in tls_ids])

                # Network-wide averages this step
                avg_queue = float(np.mean([m[t]['queue_length'] for t in tls_ids]))
                avg_wt = float(np.mean([m[t]['waiting_time'] for t in tls_ids]))
                avg_delay = float(np.mean([m[t]['delay'] for t in tls_ids]))
                tt = (float(np.mean([t_now - t for t in _trip_log.values()]))
                      if _trip_log else 0.0)

                log_delay.append(avg_delay)
                log_wt.append(avg_wt)
                log_queue.append(avg_queue)
                log_trip.append(tt)

                # Periodic logging
                if dec_step % log_interval == 0:
                    pct = 100.0 * sim_step / self.eval_steps
                    avg_speed = float(np.mean([m[t]['speed'] for t in tls_ids]))
                    print(f"[{pct:5.1f}%] Step {sim_step:7,d} | "
                          f"Rew {cumulative_reward:9.1f} | "
                          f"Q/int {avg_queue:5.1f} | W/int {avg_wt:7.1f}s | "
                          f"D/int {avg_delay:5.1f}s | T {tt:5.1f}s | "
                          f"Spd {avg_speed:4.1f}")

                if done > 0.5:
                    print("[EVALUATOR] Simulation ended (no more vehicles)")
                    break

        elapsed = time.time() - t0

        # ── Per-intersection ranking ────────────────────────
        ranking = []
        for t in tls_ids:
            ad = agent_data[t]
            mt = np.mean(ad['th'])
            md = np.mean(ad['d'])
            ranking.append({
                'tls_id': t, 'score': mt / (1.0 + md),
                'waiting': np.mean(ad['wt']), 'queue': np.mean(ad['q']),
                'throughput': mt, 'delay': md,
                'co2': np.mean(ad['co2']), 'speed': np.mean(ad['spd']),
                'reward': np.sum(ad['rew']),
            })
        ranking.sort(key=lambda x: x['score'], reverse=True)
        nes = np.mean([r['score'] for r in ranking])

        # ── Print Intersection Ranking ──────────────────────
        print(f"\n{'='*100}")
        print(f"  MAPPO EVALUATION -- INTERSECTION RANKING  "
              f"(checkpoint: {ck_name})")
        print(f"{'='*100}")
        print(f"  {'Rank':<6} {'TLS ID':<35} {'Wait(s)':<10} {'Queue':<8} "
              f"{'Thru':<8} {'Delay(s)':<10} {'Speed':<8} {'Score':<8}")
        print(f"  {'-'*94}")
        for i, r in enumerate(ranking):
            lbl = "BEST" if i == 0 else (
                "WORST" if i == len(ranking)-1 else f"  #{i+1}")
            print(f"  {lbl:<6} {r['tls_id'][:33]:<35} "
                  f"{r['waiting']:>8.1f}  {r['queue']:>6.1f}  "
                  f"{r['throughput']:>6.1f}  {r['delay']:>8.1f}  "
                  f"{r['speed']:>6.1f}  {r['score']:>6.4f}")
        print(f"  {'-'*94}")
        print(f"  NES: {nes:.4f} | Trip: {log_trip[-1] if log_trip else 0:.1f}s "
              f"| Reward: {cumulative_reward:.1f}")
        print(f"{'='*100}")

        # ── Print Final Summary ─────────────────────────────
        print(f"\n{'='*65}")
        print(f"  MAPPO EVALUATION RESULTS")
        print(f"  Checkpoint : {ck_name}")
        print(f"  Steps      : {sim_step:,}")
        print(f"  Time       : {elapsed:.0f}s")
        print(f"{'='*65}")
        if log_delay:
            print(f"  Mean Delay        : {np.mean(log_delay):>10.2f} s")
            print(f"  Mean Wait         : {np.mean(log_wt):>10.2f} s")
            print(f"  Mean Queue        : {np.mean(log_queue):>10.1f} veh")
            print(f"  Mean Trip Time    : {np.mean(log_trip):>10.2f} s")
        print(f"  Cumulative Reward : {cumulative_reward:>10.2f}")
        print(f"  Total CO2         : {total_co2:>10.0f} mg")
        print(f"{'='*65}")

        env.close()
        print(f"\n[EVALUATOR] Done. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        return {
            'log_delay': log_delay, 'log_wt': log_wt,
            'log_queue': log_queue, 'log_trip': log_trip,
            'cumulative_reward': cumulative_reward,
            'total_co2': total_co2,
            'ranking': ranking, 'nes': nes,
        }


# ─────────────────────────────────────────────────────────────
def _find_latest_checkpoint(save_dir):
    """Find the checkpoint with the highest step number."""
    import re
    best_step = -1
    best_path = None
    for f in os.listdir(save_dir):
        if f.startswith("mappo_step_") and f.endswith(".pt"):
            m = re.search(r'step_(\d+)', f)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best_path = os.path.join(save_dir, f)
    # Fallback to mappo_final.pt
    if best_path is None:
        final = os.path.join(save_dir, "mappo_final.pt")
        if os.path.exists(final):
            best_path = final
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="MAPPO Checkpoint Evaluator")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a .pt checkpoint file. If not given, "
                             "uses the latest checkpoint in saved_models/")
    parser.add_argument("--sumo-cfg", default=None,
                        help="Path to the SUMO .sumocfg file")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Evaluation steps (default: 50000)")
    parser.add_argument("--gui", action="store_true",
                        help="Run with SUMO GUI")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    save_dir = os.path.join(
        os.path.dirname(__file__), "saved_models")

    # Resolve checkpoint path
    if args.checkpoint:
        ck_path = args.checkpoint
        if not os.path.isabs(ck_path):
            # Try relative to project dir first, then saved_models
            if os.path.exists(ck_path):
                pass
            elif os.path.exists(os.path.join(save_dir, ck_path)):
                ck_path = os.path.join(save_dir, ck_path)
    else:
        ck_path = _find_latest_checkpoint(save_dir)

    if ck_path is None or not os.path.exists(ck_path):
        print("[ERROR] No checkpoint found!")
        print(f"  Searched: {save_dir}")
        print(f"  Use --checkpoint <path> to specify a checkpoint file.")
        sys.exit(1)

    print(f"[EVALUATOR] Using checkpoint: {ck_path}")

    cfg = MAPPOConfig(
        sumo_cfg=args.sumo_cfg or DEFAULT_SUMO_CFG,
        model_save_path=save_dir,
        use_gui=args.gui,
        seed=args.seed,
    )

    evaluator = MAPPOEvaluator(cfg, ck_path, eval_steps=args.steps)
    results = evaluator.run()


if __name__ == "__main__":
    main()
