import os
import sys
import re
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from torch.amp import GradScaler, autocast
import traci
import sumo_topology

# ── SUMO setup ──────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


# ── Project directory (for relative paths) ───────────────────
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))          # mappo/
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)                        # repository root/

# ── GPU Optimizations ────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_IS_WINDOWS = sys.platform.startswith("win")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ╔═══════════════════════════════════════════════════════════╗
# ║                     CONFIGURATION                         ║
# ╚═══════════════════════════════════════════════════════════╝

@dataclass
class MAPPOConfig:
    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef_start: float = 0.15
    entropy_coef_end: float = 0.02
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    grad_accum_steps: int = 1

    # ── Learning rates ──
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    lr_decay: bool = True
    lr_end_factor: float = 0.1

    # ── Training schedule ──
    ppo_epochs: int = 4
    mini_batch_size: int = 128
    update_every: int = 512
    decision_interval: int = 5
    total_steps: int = 20_000
    save_every: int = 20_000
    log_interval: int = 100
    episode_steps: int = 20_000

    # ── Network (ENLARGED) ──
    obs_dim: int = 15
    embed_dim: int = 128
    hidden_dim: int = 256
    gru_dim: int = 128
    gru_num_layers: int = 3
    gat_heads: int = 8
    gat_layers: int = 4
    ffn_dim: int = 512
    id_embed_dim: int = 32
    dropout: float = 0.1

    # ── Feature encoder depth ──
    encoder_hidden_layers: int = 3

    # ── Actor / Critic MLP depth ──
    actor_hidden_layers: int = 3
    critic_hidden_layers: int = 4

    # ── Traffic ──
    min_green: int = 10
    max_green: int = 60

    # ── Normalization ──
    max_cars: float = 300.0
    max_wait: float = 600.0
    max_co2: float = 5_000.0
    max_speed: float = 13.89

    reward_w_wait: float = 0.35
    reward_w_queue: float = 0.25
    reward_w_fairness: float = 0.20
    reward_w_max_lane: float = 0.10
    reward_w_starvation: float = 0.10
    reward_clip: float = 10.0
    starvation_memory: float = 0.95

    # ── Phase diversity ──
    phase_diversity_coef: float = 0.02
    logit_temperature: float = 1.2

    sumo_cfg: str = os.path.join(_REPO_ROOT, "sumo", "project (2).sumocfg")
    model_save_path: str = os.path.join(_PROJECT_DIR, "saved_models")
    use_gui: bool = False

    sim_step_length: float = 0.1
    weather_factor: float = 1.0
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    use_value_norm: bool = True
    use_adaptive_entropy: bool = True

    # ── Constrained RL (Lagrangian PPO) ──
    use_constrained_rl: bool = True
    co2_threshold: float = 3000.0
    delay_threshold: float = 30.0
    lagrangian_lr: float = 1e-3

    # ── GPU Speed ──
    use_amp: bool = True
    use_compile: bool = field(default_factory=lambda: not sys.platform.startswith("win"))

    def get_device(self):
        return torch.device(self.device)

    @property
    def total_decision_steps(self):
        return self.total_steps // self.decision_interval

    @property
    def total_updates(self):
        return self.total_decision_steps // self.update_every

    def __post_init__(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        set_seed(self.seed)
        amp_note = "(AMP ON)" if self.use_amp and self.device == "cuda" else ""
        compile_note = "(torch.compile ON)" if self.use_compile else ""
        print(f"[CONFIG] Device: {self.device}, Seed: {self.seed} {amp_note} {compile_note}")
        print(f"[CONFIG] obs_dim={self.obs_dim}, total_steps={self.total_steps:,}")
        print(f"[CONFIG] ENLARGED NETWORK: embed={self.embed_dim}, hidden={self.hidden_dim}, "
              f"gru_dim={self.gru_dim}x{self.gru_num_layers}L, "
              f"gat={self.gat_layers}L×{self.gat_heads}H, ffn={self.ffn_dim}")


# ╔═══════════════════════════════════════════════════════════╗
# ║                RUNNING MEAN / STD                         ║
# ╚═══════════════════════════════════════════════════════════╝

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1 and self.mean.ndim == 1:
            batch = batch.reshape(1, -1)
        elif batch.ndim == 0:
            batch = batch.reshape(1)
        bm, bv, bc = np.mean(batch, 0), np.var(batch, 0), batch.shape[0]
        delta = bm - self.mean
        total = self.count + bc
        self.mean = self.mean + delta * bc / total
        self.var = (self.var * self.count + bv * bc +
                    np.square(delta) * self.count * bc / total) / total
        self.count = total

    def normalize(self, x):
        return (np.asarray(x) - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ValueNormalizer:
    def __init__(self):
        self.rms = RunningMeanStd(shape=())

    def normalize(self, returns):
        flat = np.asarray(returns, np.float64).flatten()
        self.rms.update(flat)
        return (returns - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)

    def denormalize(self, values):
        return values * (np.sqrt(self.rms.var) + 1e-8) + self.rms.mean

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, d):
        self.rms.load_state_dict(d)


# ╔═══════════════════════════════════════════════════════════╗
# ║             VEHICLE CACHE & LANE HELPERS (CPU)            ║
# ╚═══════════════════════════════════════════════════════════╝

class VehicleCache:
    def __init__(self):
        self._cache = {}
        self._step = -1

    def get(self, lanes, sim_step):
        if sim_step != self._step:
            self._cache.clear()
            self._step = sim_step
        key = tuple(lanes)
        if key not in self._cache:
            vehs = []
            for lane in lanes:
                try:
                    vehs.extend(traci.lane.getLastStepVehicleIDs(lane))
                except Exception:
                    pass
            self._cache[key] = vehs
        return self._cache[key]


_vcache = VehicleCache()


def _get_lane_vehicles(lanes, sim_step=-1):
    if sim_step >= 0:
        return _vcache.get(lanes, sim_step)
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
    wt = 0.0
    for v in vehs:
        try:
            wt += traci.vehicle.getWaitingTime(v)
        except Exception:
            pass
    return wt


def _get_delay_raw(vehs):
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


# ╔═══════════════════════════════════════════════════════════╗
# ║                         REWARD FUNCTION                   ║
# ╚═══════════════════════════════════════════════════════════╝

class RewardComputer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._lane_ema = {}
        self._alpha = 1.0 - cfg.starvation_memory

    def compute(self, tls_id, lanes, sim_step):
        cfg = self.cfg
        vehs = _get_lane_vehicles(lanes, sim_step)
        halt = _get_halting_count(lanes)

        raw_wt = _get_waiting_time_raw(vehs) / cfg.max_wait
        raw_q = halt / cfg.max_cars

        per_lane_q = []
        for lane in lanes:
            try:
                lq = traci.lane.getLastStepHaltingNumber(lane) / cfg.max_cars
            except Exception:
                lq = 0.0
            per_lane_q.append(lq)
            prev = self._lane_ema.get(lane, 0.0)
            self._lane_ema[lane] = prev * cfg.starvation_memory + lq * self._alpha

        fairness_penalty = float(np.var(per_lane_q)) if len(per_lane_q) > 1 else 0.0
        max_lane_q = max(per_lane_q) if per_lane_q else 0.0

        starvation = 0.0
        if lanes:
            lane_emas = [self._lane_ema.get(l, 0.0) for l in lanes]
            starvation = max(lane_emas)

        reward = (
            - cfg.reward_w_wait * raw_wt
            - cfg.reward_w_queue * raw_q
            - cfg.reward_w_fairness * fairness_penalty
            - cfg.reward_w_max_lane * max_lane_q
            - cfg.reward_w_starvation * starvation
        )

        return float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))

    def state_dict(self):
        return {'lane_ema': dict(self._lane_ema)}

    def load_state_dict(self, d):
        self._lane_ema = d.get('lane_ema', {})


# ╔═══════════════════════════════════════════════════════════╗
# ║              FIXED FEATURE EXTRACTOR                      ║
# ╚═══════════════════════════════════════════════════════════╝

class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_normalizer = RunningMeanStd(shape=(cfg.obs_dim,))

    def extract(self, tls_id, lanes, prev_dur, sim_step):
        cfg = self.cfg
        vehs = _get_lane_vehicles(lanes, sim_step)

        hpl = []
        for lane in lanes:
            try:
                hpl.append(traci.lane.getLastStepHaltingNumber(lane))
            except Exception:
                hpl.append(0)

        th = sum(hpl)
        tc = _get_vehicle_count(lanes)

        mean_queue = (th / max(len(lanes), 1)) / cfg.max_cars
        max_queue = (max(hpl) if hpl else 0) / cfg.max_cars
        count_norm = tc / cfg.max_cars
        mean_occ = _get_occupancy(lanes)
        mean_speed = min(_get_mean_speed(lanes) / cfg.max_speed, 1.0)

        heavy_ratio = 0.0
        emergency = 0.0

        wt_norm = _get_waiting_time_raw(vehs) / cfg.max_wait
        co2_norm = _get_co2_raw(vehs) / cfg.max_co2
        jam_norm = min((th * 5.0) / 100.0, 1.0)

        cp = traci.trafficlight.getPhase(tls_id)
        np_ = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        phase_norm = cp / max(np_ - 1, 1)
        dur_norm = prev_dur / cfg.max_green
        throughput = tc / cfg.max_cars
        time_norm = min((sim_step * cfg.sim_step_length) / 3600.0, 1.0)

        return np.array([
            mean_queue, max_queue, count_norm, mean_occ, mean_speed,
            heavy_ratio, emergency, wt_norm, co2_norm,
            jam_norm, phase_norm, dur_norm, throughput, time_norm,
            cfg.weather_factor,
        ], dtype=np.float32)

    def normalize_batch(self, obs_batch):
        obs_batch = np.asarray(obs_batch, dtype=np.float64)
        self.obs_normalizer.update(obs_batch)
        return self.obs_normalizer.normalize(obs_batch).astype(np.float32)

    def normalize_frozen(self, obs_batch):
        return self.obs_normalizer.normalize(
            np.asarray(obs_batch, np.float64)).astype(np.float32)

    def normalize_single(self, obs):
        return self.obs_normalizer.normalize(obs).astype(np.float32)


# ╔═══════════════════════════════════════════════════════════╗
# ║        NEURAL NETWORKS  (ENLARGED MAPPO-CTDE)             ║
# ╚═══════════════════════════════════════════════════════════╝

def _build_mlp(in_dim, hidden_dim, out_dim, num_hidden_layers, dropout=0.0):
    layers = []
    layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class FeatureEncoder(nn.Module):
    def __init__(self, obs_dim, embed_dim, hidden_dim, num_hidden_layers=3, dropout=0.1):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dim, embed_dim, num_hidden_layers, dropout)

    def forward(self, x):
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # dropout applies between GRU layers (ignored when num_layers=1)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x, h=None):
        out, h_new = self.gru(x.unsqueeze(1), h)
        out = self.layer_norms[-1](out.squeeze(1))
        return out, h_new

    def forward_batch(self, x_batch):
        # x_batch: (B, N, D)
        B, N, D = x_batch.shape
        x_flat = x_batch.reshape(B * N, 1, D)
        out_flat, _ = self.gru(x_flat)
        out_flat = self.layer_norms[-1](out_flat.squeeze(1))
        return out_flat.reshape(B, N, -1)


class GraphTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn_vec = nn.Parameter(torch.randn(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_mask):
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        x_n = self.norm1(x)
        Q = self.W_q(x_n).view(B, N, H, Dh).permute(0, 2, 1, 3)
        K = self.W_k(x_n).view(B, N, H, Dh).permute(0, 2, 1, 3)
        V = self.W_v(x_n).view(B, N, H, Dh).permute(0, 2, 1, 3)

        attn_input = self.leaky_relu(Q.unsqueeze(3) + K.unsqueeze(2))
        a_vec = self.attn_vec.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        attn = (attn_input * a_vec).sum(dim=-1)

        safe_adj = adj_mask.clone()
        diag = torch.arange(min(N, safe_adj.size(0)), device=x.device)
        safe_adj[diag, diag] = 1.0
        mask = safe_adj.unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = attn.sum(dim=-1, keepdim=True)
        uniform = torch.ones_like(attn) / max(N, 1)
        attn = torch.where(row_sum > 1e-8, attn, uniform)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        x = x + self.dropout(self.W_o(out))
        x = x + self.ffn(self.norm2(x))

        return x if batched else x.squeeze(0)


class GraphTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers=4,
                 max_agents=64, dropout=0.1):
        super().__init__()
        self.pos_encoding = nn.Embedding(max_agents, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, adj_mask):
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, N, D = x.shape
        pos_ids = torch.arange(N, device=x.device)
        x = x + self.pos_encoding(pos_ids).unsqueeze(0)
        if not batched:
            x = x.squeeze(0)
        for layer in self.layers:
            x = layer(x, adj_mask)
        return self.final_norm(x)


class SharedActor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_phases, num_agents, id_embed_dim,
                 temperature=1.0, num_hidden_layers=3):
        super().__init__()
        self.temperature = temperature
        self.id_embedding = nn.Embedding(num_agents, id_embed_dim)

        # Build deeper actor MLP
        in_dim = embed_dim + id_embed_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
        self.net = nn.Sequential(*layers)

        self.phase_head = nn.Linear(hidden_dim, max_phases)
        nn.init.orthogonal_(self.phase_head.weight, gain=0.01)
        nn.init.zeros_(self.phase_head.bias)

    def forward(self, x, agent_ids, phase_masks=None):
        combined = torch.cat([x, self.id_embedding(agent_ids)], dim=-1)
        logits = self.phase_head(self.net(combined))
        logits = logits / self.temperature
        if phase_masks is not None:
            logits = logits.masked_fill(phase_masks == 0, float('-inf'))
        return logits


class CentralizedCritic(nn.Module):
    def __init__(self, gru_dim, num_agents, hidden_dim, id_embed_dim,
                 num_hidden_layers=4):
        super().__init__()
        self.num_agents = num_agents
        self.id_embedding = nn.Embedding(num_agents, id_embed_dim)
        total_input = gru_dim * num_agents + id_embed_dim

        layers = [
            nn.Linear(total_input, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4), nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.ReLU(),
        ]
        cur_dim = hidden_dim * 2
        for _ in range(num_hidden_layers - 2):
            layers += [
                nn.Linear(cur_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(),
            ]
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, all_temporal, agent_ids=None):
        if all_temporal.dim() == 2:
            N, D = all_temporal.shape
            gs = all_temporal.reshape(1, N * D).expand(N, -1)
            if agent_ids is None:
                agent_ids = torch.arange(N, device=all_temporal.device)
            return self.net(torch.cat([gs, self.id_embedding(agent_ids)], -1)).squeeze(-1)
        B, N, D = all_temporal.shape
        gs = all_temporal.reshape(B, N * D).unsqueeze(1).expand(-1, N, -1).reshape(B * N, N * D)
        ids = torch.arange(N, device=all_temporal.device).repeat(B)
        return self.net(torch.cat([gs, self.id_embedding(ids)], -1)).squeeze(-1).reshape(B, N)


class MAPPONetwork(nn.Module):
    def __init__(self, cfg, num_agents, max_phases, agent_phase_counts):
        super().__init__()
        self.num_agents = num_agents
        self.max_phases = max_phases

        self.feature_encoder = FeatureEncoder(
            cfg.obs_dim, cfg.embed_dim, cfg.hidden_dim,
            num_hidden_layers=cfg.encoder_hidden_layers,
            dropout=cfg.dropout,
        )
        self.temporal_encoder = TemporalEncoder(
            cfg.embed_dim, cfg.gru_dim,
            num_layers=cfg.gru_num_layers,
            dropout=cfg.dropout,
        )
        self.gat = GraphTransformer(
            embed_dim=cfg.gru_dim, num_heads=cfg.gat_heads,
            ffn_dim=cfg.ffn_dim, num_layers=cfg.gat_layers,
            max_agents=max(num_agents, 64), dropout=cfg.dropout,
        )
        self.actor = SharedActor(
            cfg.gru_dim, cfg.hidden_dim, max_phases,
            num_agents, cfg.id_embed_dim,
            temperature=cfg.logit_temperature,
            num_hidden_layers=cfg.actor_hidden_layers,
        )
        self.critic = CentralizedCritic(
            cfg.gru_dim, num_agents,
            cfg.hidden_dim, cfg.id_embed_dim,
            num_hidden_layers=cfg.critic_hidden_layers,
        )

        pm = torch.zeros(num_agents, max_phases)
        for i, pc in enumerate(agent_phase_counts):
            pm[i, :pc] = 1.0
        self.register_buffer('phase_masks', pm)

    def init_hidden(self, device):
        return torch.zeros(
            self.temporal_encoder.num_layers,
            self.num_agents,
            self.temporal_encoder.hidden_dim,
            device=device,
        )

    def encode(self, obs, h=None):
        return self.temporal_encoder(self.feature_encoder(obs), h)

    def encode_batch(self, obs_batch):
        return self.temporal_encoder.forward_batch(self.feature_encoder(obs_batch))

    def communicate(self, temporal, adj_mask):
        return self.gat(temporal, adj_mask)

    def get_actor_logits(self, communicated, device):
        if communicated.dim() == 2:
            N = communicated.size(0)
            return self.actor(communicated, torch.arange(N, device=device),
                              self.phase_masks)
        B, N, D = communicated.shape
        flat = communicated.reshape(B * N, D)
        ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
        masks = self.phase_masks.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        return self.actor(flat, ids, masks).reshape(B, N, -1)

    def get_values(self, temporal):
        return self.critic(temporal)

    def forward(self, obs, adj_mask, h=None):
        temporal, h_new = self.encode(obs, h)
        communicated = self.communicate(temporal, adj_mask)
        logits = self.get_actor_logits(communicated, obs.device)
        values = self.get_values(temporal)
        return logits, values, h_new, temporal


# ╔═══════════════════════════════════════════════════════════╗
# ║             MULTI-AGENT PPO BUFFER                        ║
# ╚═══════════════════════════════════════════════════════════╝

class MultiAgentBuffer:
    def __init__(self):
        self.clear()

    def store(self, obs, actions, log_probs, values, rewards, done,
              co2_val=None, delay_val=None):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(done)
        if co2_val is not None:
            self.co2_vals.append(co2_val)
        if delay_val is not None:
            self.delay_vals.append(delay_val)

    def compute_gae(self, last_values, gamma, gae_lambda):
        T = len(self.rewards)
        N = self.rewards[0].shape[0]
        last_values = np.asarray(last_values, np.float32).reshape(N)
        safe_vals = [np.asarray(v, np.float32).reshape(N) for v in self.values]
        all_vals = safe_vals + [last_values]

        advantages = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            dm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * all_vals[t + 1] * dm - all_vals[t]
            gae = delta + gamma * gae_lambda * dm * gae
            advantages[t] = gae
        returns = advantages + np.stack(safe_vals)
        return advantages, returns

    def get_mean_constraints(self):
        mc = float(np.mean(self.co2_vals)) if self.co2_vals else 0.0
        md = float(np.mean(self.delay_vals)) if self.delay_vals else 0.0
        return mc, md

    def generate_batches(self, advantages, returns, mbs):
        T = len(self.obs)
        idx = np.arange(T)
        np.random.shuffle(idx)
        for s in range(0, T, mbs):
            bi = idx[s:s + mbs]
            if len(bi) == 0:
                continue
            batch = {
                'obs': np.stack([self.obs[i] for i in bi]),
                'actions': np.stack([self.actions[i] for i in bi]),
                'log_probs': np.stack([self.log_probs[i] for i in bi]),
                'advantages': advantages[bi],
                'returns': returns[bi],
            }
            if self.co2_vals:
                batch['co2'] = np.array([self.co2_vals[i] for i in bi])
            if self.delay_vals:
                batch['delay'] = np.array([self.delay_vals[i] for i in bi])
            yield batch

    def clear(self):
        self.obs = []; self.actions = []; self.log_probs = []
        self.values = []; self.rewards = []; self.dones = []
        self.co2_vals = []; self.delay_vals = []

    def size(self):
        return len(self.rewards)


# ╔═══════════════════════════════════════════════════════════╗
# ║         LAGRANGIAN CONSTRAINTS (Constrained RL)           ║
# ╚═══════════════════════════════════════════════════════════╝

class LagrangianConstraints:
    def __init__(self, cfg):
        self.co2_threshold = cfg.co2_threshold
        self.delay_threshold = cfg.delay_threshold
        self.lr = cfg.lagrangian_lr
        self._log_lam_co2 = 0.0
        self._log_lam_delay = 0.0

    @property
    def lambda_co2(self):
        return math.exp(self._log_lam_co2)

    @property
    def lambda_delay(self):
        return math.exp(self._log_lam_delay)

    def compute_penalty(self, mean_co2, mean_delay):
        co2_v = max(0.0, mean_co2 - self.co2_threshold)
        del_v = max(0.0, mean_delay - self.delay_threshold)
        return self.lambda_co2 * co2_v + self.lambda_delay * del_v

    def update_multipliers(self, mean_co2, mean_delay):
        self._log_lam_co2 += self.lr * (mean_co2 - self.co2_threshold)
        self._log_lam_delay += self.lr * (mean_delay - self.delay_threshold)
        self._log_lam_co2 = np.clip(self._log_lam_co2, -10.0, 5.0)
        self._log_lam_delay = np.clip(self._log_lam_delay, -10.0, 5.0)

    def state_dict(self):
        return {'log_lam_co2': self._log_lam_co2, 'log_lam_delay': self._log_lam_delay}

    def load_state_dict(self, d):
        self._log_lam_co2 = d.get('log_lam_co2', 0.0)
        self._log_lam_delay = d.get('log_lam_delay', 0.0)


# ╔═══════════════════════════════════════════════════════════╗
# ║             SUMO MULTI-AGENT ENVIRONMENT                  ║
# ╚═══════════════════════════════════════════════════════════╝

class SUMOMultiAgentEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        self.reward_computers = {}
        self.tls_ids = []
        self.num_agents = 0
        self.agent_info = {}
        self.adj_matrix = None
        self.max_phases = 0
        self.agent_phase_counts = []
        self.sim_step = 0
        self._trip_log = {}
        self._is_open = False
        self._topo_neighbors = {}
        self._topo_distances = {}
        self._topo_centers = {}
        self._net_info = None

    def reset(self):
        if self._is_open:
            try:
                traci.close()
            except Exception:
                pass
            time.sleep(0.5)

        binary = "sumo-gui.exe" if self.cfg.use_gui else "sumo.exe"
        sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", binary)

        sumo_cmd = [
            sumo_binary,
            "-c", self.cfg.sumo_cfg,
            "--step-length", str(self.cfg.sim_step_length),
            "--delay", "0",
            "--lateral-resolution", "0",
            "--seed", str(self.cfg.seed),
            "--ignore-route-errors",
            "--no-warnings",
            "--error-log", os.path.join(self.cfg.model_save_path, "sumo_errors.log"),
            "--duration-log.disable",
            "--no-step-log",
            "--collision.action", "warn",
            "--time-to-teleport", "300",
            "--time-to-teleport.highways", "0",
        ]

        connected = False
        for attempt in range(3):
            try:
                traci.start(sumo_cmd)
                connected = True
                break
            except Exception as e:
                print(f"[WARN] TraCI start attempt {attempt+1} failed: {e}")
                time.sleep(1.5)
                try:
                    traci.close()
                except Exception:
                    pass

        if not connected:
            raise RuntimeError(
                "[ERROR] Could not connect to SUMO after 3 attempts.\n"
                "Check that SUMO_HOME is correct and the .sumocfg file exists.\n"
                f"  SUMO_HOME = {os.environ.get('SUMO_HOME')}\n"
                f"  sumocfg   = {self.cfg.sumo_cfg}"
            )

        self._is_open = True
        self.sim_step = 0
        self._trip_log = {}
        _vcache._step = -1

        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.num_agents = len(self.tls_ids)
        detector_ids = list(traci.lanearea.getIDList())
        print(f"[ENV] {self.num_agents} traffic lights, {len(detector_ids)} detectors")

        use_real_topology = False
        try:
            net_file = sumo_topology.extract_net_file_from_sumocfg(self.cfg.sumo_cfg)
            net_info = sumo_topology.parse_sumo_network(net_file)
            self._net_info = net_info

            topo_nbrs, topo_dists, topo_centers = \
                sumo_topology.build_tls_adjacency(self.tls_ids, net_info)
            self._topo_neighbors = topo_nbrs
            self._topo_distances = topo_dists
            self._topo_centers = topo_centers

            sumo_topology.print_topology_summary(
                self.tls_ids, topo_nbrs, topo_dists, topo_centers)
            use_real_topology = True
            print("[ENV] ✓ Real network topology loaded from .net.xml")
        except Exception as e:
            print(f"[WARN] Could not parse network topology: {e}")
            print("[WARN] Falling back to detector-based neighbor graph")

        tls_det_map = self._build_detector_map(detector_ids)
        det_to_lane = self._build_det_to_lane_map(detector_ids)
        if not use_real_topology:
            nbr_graph = self._build_neighbor_graph(self.tls_ids, detector_ids, tls_det_map)

        self.agent_info = {}
        self.agent_phase_counts = []
        self.max_phases = 0
        self.reward_computers = {t: RewardComputer(self.cfg) for t in self.tls_ids}

        for i, tls_id in enumerate(self.tls_ids):
            tls_key = f"TLS_{i + 1}"
            dets = tls_det_map.get(tls_key, detector_ids)
            lanes = self._get_lanes(dets, det_to_lane)
            if use_real_topology:
                nbrs = self._topo_neighbors.get(tls_id, [])
            else:
                nbrs = nbr_graph.get(tls_id, [])
            np_ = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
            self.max_phases = max(self.max_phases, np_)
            self.agent_phase_counts.append(np_)
            self.agent_info[tls_id] = {
                'index': i, 'lanes': lanes, 'neighbors': nbrs,
                'num_phases': np_, 'prev_dur': float(self.cfg.min_green),
                'last_switch': -self.cfg.min_green,
            }

        if use_real_topology:
            self.adj_matrix = sumo_topology.build_adjacency_matrix(
                self.tls_ids, self._topo_neighbors, self._topo_distances)
        else:
            self.adj_matrix = self._build_adj_matrix(nbr_graph)
        obs_all = self._get_all_obs()
        return obs_all, {
            'num_agents': self.num_agents, 'max_phases': self.max_phases,
            'agent_phase_counts': self.agent_phase_counts,
        }

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = {self.tls_ids[i]: int(actions[i]) for i in range(self.num_agents)}

        for idx, tls_id in enumerate(self.tls_ids):
            ai = self.agent_info[tls_id]
            if tls_id in actions:
                cur = traci.trafficlight.getPhase(tls_id)
                can = (self.sim_step - ai['last_switch']) >= self.cfg.min_green
                if actions[tls_id] != cur and can:
                    traci.trafficlight.setPhase(tls_id, actions[tls_id])
                    ai['last_switch'] = self.sim_step

        for _ in range(self.cfg.decision_interval):
            traci.simulationStep()
            self.sim_step += 1
            self._update_trip_log()

        obs_all = self._get_all_obs()
        rewards = self._compute_rewards()
        done = 1.0 if (traci.simulation.getMinExpectedNumber() <= 0 or
                       self.sim_step >= self.cfg.total_steps) else 0.0
        step_co2, step_delay = self._get_constraint_metrics()
        return obs_all, rewards, done, {
            'mean_co2': step_co2, 'mean_delay': step_delay,
        }

    def close(self):
        if self._is_open:
            try:
                traci.close()
            except Exception:
                pass
            self._is_open = False

    def _get_all_obs(self):
        return np.stack([
            self.feature_extractor.extract(
                t, self.agent_info[t]['lanes'],
                self.agent_info[t]['prev_dur'], self.sim_step
            ) for t in self.tls_ids
        ], axis=0)

    def _compute_rewards(self):
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i, tls_id in enumerate(self.tls_ids):
            ai = self.agent_info[tls_id]
            r = self.reward_computers[tls_id].compute(
                tls_id, ai['lanes'], self.sim_step)
            rewards[i] = r
        return rewards

    def _get_constraint_metrics(self):
        all_co2, all_delay = [], []
        for tls_id in self.tls_ids:
            lanes = self.agent_info[tls_id]['lanes']
            vehs = _get_lane_vehicles(lanes, self.sim_step)
            all_co2.append(_get_co2_raw(vehs))
            all_delay.append(_get_delay_raw(vehs))
        return (float(np.mean(all_co2)) if all_co2 else 0.0,
                float(np.mean(all_delay)) if all_delay else 0.0)

    def get_per_agent_metrics(self):
        metrics = {}
        for tls_id in self.tls_ids:
            lanes = self.agent_info[tls_id]['lanes']
            vehs = _get_lane_vehicles(lanes, self.sim_step)
            metrics[tls_id] = {
                'waiting_time': _get_waiting_time_raw(vehs),
                'queue_length': _get_halting_count(lanes),
                'throughput': _get_vehicle_count(lanes),
                'delay': _get_delay_raw(vehs),
                'co2': _get_co2_raw(vehs),
                'speed': _get_mean_speed(lanes),
            }
        return metrics

    def get_mean_trip_time(self):
        if not self._trip_log:
            return 0.0
        now = self.sim_step * self.cfg.sim_step_length
        return float(np.mean([now - t for t in self._trip_log.values()]))

    def compute_dynamic_adjacency(self):
        N = self.num_agents
        adj = torch.eye(N)
        for i, ti in enumerate(self.tls_ids):
            vol_i = _get_vehicle_count(self.agent_info[ti]['lanes'])
            for tj_id in self.agent_info[ti].get('neighbors', []):
                if tj_id not in self.tls_ids:
                    continue
                j = self.tls_ids.index(tj_id)
                vol_j = _get_vehicle_count(self.agent_info[tj_id]['lanes'])
                traffic_w = min(1.0, (vol_i + vol_j) / (2.0 * self.cfg.max_cars))
                dist = self._topo_distances.get(
                    (ti, tj_id), self._topo_distances.get((tj_id, ti), 500.0))
                dist_w = 1.0 / (1.0 + dist / 500.0)
                w = max(0.5 * dist_w + 0.5 * traffic_w, 0.1)
                adj[i, j] = adj[j, i] = w
        return adj

    def _update_trip_log(self):
        t = self.sim_step * self.cfg.sim_step_length
        for v in traci.simulation.getDepartedIDList():
            self._trip_log[v] = t
        for v in traci.simulation.getArrivedIDList():
            self._trip_log.pop(v, None)

    @staticmethod
    def _build_detector_map(det_ids):
        m = defaultdict(list)
        vc = 0
        for d in det_ids:
            if not d.startswith("det_TLS_"):
                continue
            for part in d[len("det_"):].split("det_"):
                tok = part.split("_")
                if len(tok) >= 2 and tok[0] == "TLS":
                    m[f"TLS_{tok[1]}"].append(d)
                    vc += 1
        if vc > 0:
            return dict(m)
        for d in det_ids:
            match = re.search(r'[Tt][Ll][Ss][_\-]?(\d+)', d)
            if match:
                m[f"TLS_{match.group(1)}"].append(d)
                vc += 1
        if vc > 0:
            print(f"[INFO] Detector mapping: regex fallback matched {vc}")
            return dict(m)
        if det_ids:
            print(f"[WARN] No detectors matched. Sample: {det_ids[:3]}")
        return dict(m)

    @staticmethod
    def _build_det_to_lane_map(det_ids):
        m = {}
        for d in det_ids:
            try:
                m[d] = traci.lanearea.getLaneID(d)
            except Exception:
                m[d] = None
        return m

    @staticmethod
    def _get_lanes(dets, d2l):
        lanes = []
        for d in dets:
            l = d2l.get(d)
            if l and l not in lanes:
                lanes.append(l)
        return lanes

    @staticmethod
    def _build_neighbor_graph(tls_ids, det_ids, tls_det_map):
        shared = set()
        for d in det_ids:
            if not d.startswith("det_TLS_"):
                continue
            parts = d[len("det_"):].split("det_")
            if len(parts) < 2:
                continue
            keys = []
            for p in parts:
                tok = p.split("_")
                if len(tok) >= 2 and tok[0] == "TLS":
                    keys.append(f"TLS_{tok[1]}")
            if len(keys) == 2:
                shared.add((keys[0], keys[1]))
                shared.add((keys[1], keys[0]))
        n2r = {f"TLS_{i+1}": t for i, t in enumerate(sorted(tls_ids))}
        g = {t: [] for t in tls_ids}
        for k1, k2 in shared:
            r1, r2 = n2r.get(k1), n2r.get(k2)
            if r1 and r2:
                if r2 not in g[r1]:
                    g[r1].append(r2)
                if r1 not in g[r2]:
                    g[r2].append(r1)
        si = sorted(tls_ids)
        for t in tls_ids:
            if not g[t]:
                idx = si.index(t)
                if idx > 0:
                    g[t].append(si[idx - 1])
                if idx < len(si) - 1:
                    g[t].append(si[idx + 1])
        return g

    def _build_adj_matrix(self, nbr_graph):
        N = self.num_agents
        adj = torch.eye(N)
        for i, ti in enumerate(self.tls_ids):
            for tj in nbr_graph.get(ti, []):
                if tj in self.tls_ids:
                    j = self.tls_ids.index(tj)
                    adj[i, j] = adj[j, i] = 1.0
        return adj


# ╔═══════════════════════════════════════════════════════════╗
# ║                      MAPPO TRAINER                        ║
# ╚═══════════════════════════════════════════════════════════╝

class MAPPOTrainer:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = cfg.get_device()
        self.feature_extractor = env.feature_extractor
        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.buffer = MultiAgentBuffer()
        self.value_normalizer = ValueNormalizer() if cfg.use_value_norm else None
        self.lagrangian = LagrangianConstraints(cfg) if cfg.use_constrained_rl else None
        self.scaler = GradScaler("cuda", enabled=(cfg.use_amp and cfg.device == "cuda"))
        self.update_count = 0
        self.grad_norms = []
        self.log_steps = []; self.log_reward = []; self.log_queue = []
        self.log_wt = []; self.log_delay = []; self.log_trip = []

    def train(self):
        cfg = self.cfg
        dev = self.device
        obs_all, info = self.env.reset()
        N = info['num_agents']
        mp = info['max_phases']
        pc = info['agent_phase_counts']

        self.network = MAPPONetwork(cfg, N, mp, pc).to(dev)

        if cfg.use_compile and not _IS_WINDOWS and hasattr(torch, 'compile') and dev.type == 'cuda':
            try:
                self.network = torch.compile(self.network)
                print("[SPEED] torch.compile() enabled")
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {e}")
        elif _IS_WINDOWS:
            print("[INFO] torch.compile() skipped — Windows detected (Triton not supported)")

        tp = sum(p.numel() for p in self.network.parameters())
        print(f"\n[MAPPO] Agents={N}, MaxPhases={mp}, Params={tp:,}, Device={dev}")
        print(f"[MAPPO] AMP={cfg.use_amp}, obs_dim={cfg.obs_dim}, "
              f"update_every={cfg.update_every}, mini_batch={cfg.mini_batch_size}")

        use_episodes = cfg.episode_steps > 0
        if use_episodes:
            n_episodes = math.ceil(cfg.total_steps / cfg.episode_steps)
            print(f"[MAPPO] Multi-episode mode: {n_episodes} episodes × "
                  f"{cfg.episode_steps:,} steps each")
        else:
            print(f"[MAPPO] Single-episode mode: {cfg.total_steps:,} steps")

        actor_p = []
        for name in ('feature_encoder', 'temporal_encoder', 'gat', 'actor'):
            module = getattr(self.network, name, None)
            if module is not None:
                actor_p.extend(list(module.parameters()))

        self.optimizer = optim.Adam([
            {'params': actor_p, 'lr': cfg.actor_lr},
            {'params': self.network.critic.parameters(), 'lr': cfg.critic_lr},
        ])
        if cfg.lr_decay:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=cfg.lr_end_factor,
                total_iters=max(cfg.total_updates, 1))

        adj = self.env.adj_matrix.to(dev, non_blocking=True)
        h = self.network.init_hidden(dev)
        cum_reward = 0.0
        dec_step = 0
        global_sim_step = 0
        episode_idx = 0
        episode_sim_step = 0
        self.update_count = 0
        t0 = time.time()

        print(f"[MAPPO] Training: {cfg.total_steps:,} sim steps ...\n")

        while global_sim_step < cfg.total_steps:

            obs_norm = self.feature_extractor.normalize_batch(obs_all)
            obs_t = torch.from_numpy(obs_norm).to(dev, non_blocking=True)

            with torch.no_grad():
                with autocast('cuda', enabled=(cfg.use_amp and dev.type == 'cuda')):
                    logits, values, h_new, _ = self.network(obs_t, adj, h)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            obs_next, rewards, done, step_info = self.env.step(actions_np)

            self.buffer.store(
                obs_norm, actions_np, log_probs.cpu().numpy(),
                values.detach().cpu().numpy(), rewards, done,
                co2_val=step_info.get('mean_co2'),
                delay_val=step_info.get('mean_delay'),
            )
            cum_reward += rewards.sum()
            h = h_new
            obs_all = obs_next

            global_sim_step += cfg.decision_interval
            episode_sim_step += cfg.decision_interval
            dec_step += 1

            if dec_step % cfg.update_every == 0 and self.buffer.size() >= 2:
                self.update_count += 1
                self._ppo_update(adj, obs_all)
                if self.scheduler:
                    self.scheduler.step()
                adj = self.env.compute_dynamic_adjacency().to(dev, non_blocking=True)

            if dec_step % max(cfg.log_interval // cfg.decision_interval, 1) == 0:
                self._log(global_sim_step, cum_reward, dec_step)
            if global_sim_step % cfg.save_every == 0 and global_sim_step > 0:
                self._save(f"step_{global_sim_step}")

            episode_limit_reached = (use_episodes and episode_sim_step >= cfg.episode_steps)
            if done > 0.5 or episode_limit_reached:
                episode_idx += 1
                reason = "SUMO done" if done > 0.5 else f"episode_steps={cfg.episode_steps:,}"

                if self.buffer.size() >= 2:
                    self.update_count += 1
                    self._ppo_update(adj, obs_all)
                    if self.scheduler:
                        self.scheduler.step()

                if global_sim_step >= cfg.total_steps:
                    break

                cfg.seed += 1
                print(f"\n[EPISODE] #{episode_idx} ended ({reason}). "
                      f"Global step: {global_sim_step:,}/{cfg.total_steps:,}. "
                      f"Starting episode #{episode_idx + 1} with seed={cfg.seed} ...\n")

                obs_all, info = self.env.reset()
                h = self.network.init_hidden(dev)
                adj = self.env.adj_matrix.to(dev, non_blocking=True)
                episode_sim_step = 0

        elapsed = time.time() - t0
        self._save("final")
        print(f"\n[MAPPO] Done. {elapsed:.0f}s ({elapsed/60:.1f}min), "
              f"{self.update_count} PPO updates, {episode_idx} episode resets")
        return {
            'log_steps': self.log_steps, 'log_reward': self.log_reward,
            'log_queue': self.log_queue, 'log_wt': self.log_wt,
            'log_delay': self.log_delay, 'log_trip': self.log_trip,
            'cumulative_reward': cum_reward, 'grad_norms': self.grad_norms,
        }

    def _ppo_update(self, adj, obs_next):
        cfg = self.cfg
        dev = self.device

        with torch.no_grad():
            on = self.feature_extractor.normalize_frozen(obs_next)
            ot = torch.from_numpy(on).to(dev, non_blocking=True)
            with autocast('cuda', enabled=(cfg.use_amp and dev.type == 'cuda')):
                _, lv, _, _ = self.network(ot, adj)
            lv = lv.cpu().numpy()

        adv, ret = self.buffer.compute_gae(lv, cfg.gamma, cfg.gae_lambda)
        if self.value_normalizer:
            ret = self.value_normalizer.normalize(ret).astype(np.float32)

        if self.lagrangian:
            mc, md = self.buffer.get_mean_constraints()
            self.lagrangian.update_multipliers(mc, md)

        for _ in range(cfg.ppo_epochs):
            for batch in self.buffer.generate_batches(adv, ret, cfg.mini_batch_size):
                self._update_step(batch, adj)
        self.buffer.clear()

    def _update_step(self, batch, adj):
        cfg = self.cfg
        dev = self.device

        obs_b = torch.from_numpy(batch['obs']).to(dev, non_blocking=True)
        act_b = torch.from_numpy(batch['actions']).long().to(dev, non_blocking=True)
        olp_b = torch.from_numpy(batch['log_probs']).to(dev, non_blocking=True)
        ret_b = torch.from_numpy(batch['returns']).to(dev, non_blocking=True)
        adv_b = torch.from_numpy(batch['advantages']).to(dev, non_blocking=True)

        if adv_b.dim() == 2 and adv_b.size(1) > 1:
            agent_mean = adv_b.mean(dim=0, keepdim=True)
            agent_std = adv_b.std(dim=0, keepdim=True) + 1e-8
            adv_b = (adv_b - agent_mean) / agent_std
        else:
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=(cfg.use_amp and dev.type == 'cuda')):
            temporal = self.network.encode_batch(obs_b)
            communicated = self.network.communicate(temporal, adj)
            nl = self.network.get_actor_logits(communicated, dev)
            nd = torch.distributions.Categorical(logits=nl)
            nlp = nd.log_prob(act_b)
            ent = nd.entropy()
            nv = self.network.get_values(temporal)

            ratio = torch.exp(nlp - olp_b)
            s1 = ratio * adv_b
            s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
            a_loss = -torch.mean(torch.min(s1, s2))

            c_loss = torch.mean((ret_b - nv) ** 2)
            e_mean = torch.mean(ent)

            progress = min(self.update_count / max(cfg.total_updates, 1), 1.0)
            base_ec = (cfg.entropy_coef_start +
                       (cfg.entropy_coef_end - cfg.entropy_coef_start) * progress)
            if cfg.use_adaptive_entropy:
                max_ent = np.log(max(self.network.max_phases, 2))
                ent_ratio = e_mean.item() / max(max_ent, 1e-8)
                adaptive_scale = np.clip(3.0 * (1.0 - ent_ratio), 0.5, 3.0)
                ec = base_ec * adaptive_scale
            else:
                ec = base_ec

            probs = nd.probs
            if probs.dim() == 3:
                B, N_ag, P = probs.shape
                flat_probs = probs.reshape(B * N_ag, P)
            else:
                flat_probs = probs
            valid_mask = (flat_probs > 0).float()
            n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            uniform = valid_mask / n_valid
            log_ratio = torch.log(flat_probs.clamp(min=1e-8) / uniform.clamp(min=1e-8))
            phase_div_loss = (flat_probs * log_ratio * valid_mask).sum(dim=-1).mean()

            total_loss = (a_loss
                          + cfg.value_loss_coef * c_loss
                          - ec * e_mean
                          + cfg.phase_diversity_coef * phase_div_loss)

            if self.lagrangian and 'co2' in batch and 'delay' in batch:
                penalty = self.lagrangian.compute_penalty(
                    float(np.mean(batch['co2'])), float(np.mean(batch['delay'])))
                total_loss = total_loss + penalty

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        gn = nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
        gn_val = float(gn)
        if math.isfinite(gn_val):
            self.grad_norms.append(gn_val)
            self.scaler.step(self.optimizer)
        else:
            self.grad_norms.append(float('nan'))
        self.scaler.update()

    def _log(self, step, cum_rew, dec_step):
        m = self.env.get_per_agent_metrics()
        tt = self.env.get_mean_trip_time()
        N = max(len(m), 1)

        avg_queue = float(np.mean([v['queue_length'] for v in m.values()]))
        avg_wt    = float(np.mean([v['waiting_time'] for v in m.values()]))
        avg_delay = float(np.mean([v['delay']        for v in m.values()]))
        avg_speed = float(np.mean([v['speed']        for v in m.values()]))
        avg_co2   = float(np.mean([v['co2']          for v in m.values()]))
        avg_thru  = float(np.mean([v['throughput']   for v in m.values()]))

        self.log_steps.append(step); self.log_reward.append(cum_rew)
        self.log_queue.append(avg_queue); self.log_wt.append(avg_wt)
        self.log_delay.append(avg_delay); self.log_trip.append(tt)

        lr = self.optimizer.param_groups[0]['lr']
        valid_gn = [g for g in self.grad_norms if not math.isnan(g)]
        gn = valid_gn[-1] if valid_gn else 0.0

        lag = ""
        if self.lagrangian:
            lag = (f" | λCO2 {self.lagrangian.lambda_co2:.3f} "
                   f"| λDel {self.lagrangian.lambda_delay:.3f}")
        pct = 100.0 * step / self.cfg.total_steps
        print(f"[{pct:5.1f}%] Step {step:7,d} | Dec {dec_step:6d} | "
              f"Rew {cum_rew:9.1f} | "
              f"Q/int {avg_queue:5.1f} | W/int {avg_wt:7.1f}s | "
              f"D/int {avg_delay:5.1f}s | T {tt:5.1f}s | "
              f"Thru/int {avg_thru:4.1f} | CO2/int {avg_co2:7.0f} | "
              f"Spd {avg_speed:4.1f} | LR {lr:.1e} | GN {gn:.2f}{lag}")

    def _save(self, tag):
        d = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_norm': self.feature_extractor.obs_normalizer.state_dict(),
            'reward_computers': {t: rc.state_dict()
                                 for t, rc in self.env.reward_computers.items()},
        }
        if self.value_normalizer:
            d['value_norm'] = self.value_normalizer.state_dict()
        if self.lagrangian:
            d['lagrangian'] = self.lagrangian.state_dict()
        path = os.path.join(self.cfg.model_save_path, f"mappo_{tag}.pt")
        torch.save(d, path)
        print(f"[SAVE] {path}")

    def load_models(self, path):
        ck = torch.load(path, map_location=self.device)
        if self.network:
            self.network.load_state_dict(ck['network'])
        if self.optimizer and 'optimizer' in ck:
            self.optimizer.load_state_dict(ck['optimizer'])
        if 'obs_norm' in ck:
            self.feature_extractor.obs_normalizer.load_state_dict(ck['obs_norm'])
        if self.value_normalizer and 'value_norm' in ck:
            self.value_normalizer.load_state_dict(ck['value_norm'])
        if self.lagrangian and 'lagrangian' in ck:
            self.lagrangian.load_state_dict(ck['lagrangian'])
        print(f"[MAPPO] Loaded {path}")


# ╔═══════════════════════════════════════════════════════════╗
# ║           INTERSECTION EVALUATOR & RANKING                ║
# ╚═══════════════════════════════════════════════════════════╝

class IntersectionEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, env, network, feat_ext, max_steps=None, label="Evaluation"):
        cfg = self.cfg
        dev = cfg.get_device()
        max_steps = max_steps or cfg.total_steps
        obs_all, info = env.reset()
        adj = env.adj_matrix.to(dev, non_blocking=True)
        h = network.init_hidden(dev)

        agent_data = {t: {'wt': [], 'q': [], 'th': [], 'd': [], 'co2': [], 'spd': [], 'rew': []}
                      for t in env.tls_ids}
        total_rew = 0.0
        print(f"\n[EVAL] {label} — {max_steps:,} steps ...")

        network.eval()
        with torch.no_grad():
            for ss in range(0, max_steps, cfg.decision_interval):
                on = feat_ext.normalize_frozen(obs_all)
                ot = torch.from_numpy(on).to(dev, non_blocking=True)
                with autocast('cuda', enabled=(cfg.use_amp and dev.type == 'cuda')):
                    logits, _, h, _ = network(ot, adj, h)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()
                obs_all, rewards, done, _ = env.step(actions)
                total_rew += rewards.sum()

                m = env.get_per_agent_metrics()
                for i, t in enumerate(env.tls_ids):
                    ad = agent_data[t]
                    ad['wt'].append(m[t]['waiting_time'])
                    ad['q'].append(m[t]['queue_length'])
                    ad['th'].append(m[t]['throughput'])
                    ad['d'].append(m[t]['delay'])
                    ad['co2'].append(m[t]['co2'])
                    ad['spd'].append(m[t]['speed'])
                    ad['rew'].append(rewards[i])
                if done > 0.5:
                    break
        network.train()

        ranking = []
        for t in env.tls_ids:
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
        tt = env.get_mean_trip_time()

        print(f"\n{'='*100}")
        print(f"  {label} — INTERSECTION RANKING")
        print(f"{'='*100}")
        print(f"  {'Rank':<6} {'TLS ID':<35} {'Wait(s)':<10} {'Queue':<8} "
              f"{'Thru':<8} {'Delay(s)':<10} {'Speed':<8} {'Score':<8}")
        print(f"  {'-'*94}")
        for i, r in enumerate(ranking):
            lbl = "★ BEST" if i == 0 else ("✗ WORST" if i == len(ranking)-1 else f"  #{i+1}")
            print(f"  {lbl:<6} {r['tls_id'][:33]:<35} "
                  f"{r['waiting']:>8.1f}  {r['queue']:>6.1f}  {r['throughput']:>6.1f}  "
                  f"{r['delay']:>8.1f}  {r['speed']:>6.1f}  {r['score']:>6.4f}")
        print(f"  {'-'*94}")
        print(f"  NES: {nes:.4f} | Trip: {tt:.1f}s | Reward: {total_rew:.1f}")
        print(f"{'='*100}\n")

        return {'ranking': ranking, 'nes': nes, 'total_reward': total_rew, 'trip_time': tt}

    @staticmethod
    def plot_training(logs, save_path):
        if not logs['log_steps']:
            return
        s = logs['log_steps']
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("MAPPO-CTDE — Training Results (Enhanced Deep Network)",
                     fontsize=15, fontweight='bold')

        axes[0, 0].plot(s, logs['log_reward'], '#2196F3', lw=1.5)
        axes[0, 0].fill_between(s, logs['log_reward'], alpha=0.1, color='#2196F3')
        axes[0, 0].set_title("Cumulative Reward"); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(s, logs['log_queue'], '#FF9800', lw=1.5)
        axes[0, 1].set_title("Total Queue (vehicles)"); axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(s, logs['log_wt'], '#F44336', lw=1.5)
        axes[0, 2].set_title("Total Waiting Time (s)"); axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(s, logs['log_delay'], '#9C27B0', lw=1.5)
        axes[1, 0].set_title("Mean Delay (s)"); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(s, logs['log_trip'], '#4CAF50', lw=1.5)
        axes[1, 1].set_title("Mean Trip Time (s)"); axes[1, 1].grid(True, alpha=0.3)

        if logs.get('grad_norms'):
            gn = logs['grad_norms']
            window = max(len(gn) // 50, 1)
            smoothed = np.convolve(gn, np.ones(window)/window, mode='valid')
            axes[1, 2].plot(smoothed, '#795548', lw=1.5)
            axes[1, 2].set_title("Grad Norm (smoothed)"); axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(save_path, "mappo_training.png")
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[PLOT] Saved to {out}")

    @staticmethod
    def plot_comparison(ranking, save_path):
        if not ranking:
            return
        names = [r['tls_id'][:20] for r in ranking]
        scores = [r['score'] for r in ranking]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, len(ranking)))
        ax.barh(names, scores, color=colors)
        ax.set_title("Intersection Efficiency Score (NES)", fontsize=13)
        ax.set_xlabel("Score (throughput / (1 + delay))")
        ax.invert_yaxis()
        plt.tight_layout()
        out = os.path.join(save_path, "mappo_comparison.png")
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[PLOT] Saved to {out}")


# ╔═══════════════════════════════════════════════════════════╗
# ║                       MAIN                                ║
# ╚═══════════════════════════════════════════════════════════╝

def main():
    cfg = MAPPOConfig()

    if cfg.device == "cuda":
        print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    env = SUMOMultiAgentEnv(cfg)
    evaluator = IntersectionEvaluator(cfg)
    trainer = MAPPOTrainer(cfg, env)

    logs = trainer.train()
    evaluator.plot_training(logs, cfg.model_save_path)

    print(f"\n{'='*65}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*65}")
    if logs['log_delay']:
        print(f"  Mean Delay        : {np.mean(logs['log_delay']):>10.2f} s")
        print(f"  Mean Wait         : {np.mean(logs['log_wt']):>10.2f} s")
        print(f"  Mean Queue        : {np.mean(logs['log_queue']):>10.1f} veh")
        print(f"  Mean Trip Time    : {np.mean(logs['log_trip']):>10.2f} s")
    print(f"  Cumulative Reward : {logs['cumulative_reward']:>10.2f}")
    if logs.get('grad_norms'):
        print(f"  Avg Grad Norm     : {np.mean(logs['grad_norms']):>10.4f}")
        print(f"  Final Grad Norm   : {logs['grad_norms'][-1]:>10.4f}")
    print(f"{'='*65}")

    print("\n[INFO] Running post-training evaluation ...")
    results = evaluator.evaluate(
        env, trainer.network, env.feature_extractor,
        label="Post-Training Evaluation"
    )
    evaluator.plot_comparison(results['ranking'], cfg.model_save_path)

    print(f"\n{'='*65}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"  Network Efficiency Score : {results['nes']:>10.4f}")
    print(f"  Mean Trip Time           : {results['trip_time']:>10.1f} s")
    print(f"  Best Intersection        : {results['ranking'][0]['tls_id']}")
    print(f"    Score  : {results['ranking'][0]['score']:.4f}")
    print(f"    Wait   : {results['ranking'][0]['waiting']:.1f} s")
    print(f"    Delay  : {results['ranking'][0]['delay']:.1f} s")
    print(f"    Speed  : {results['ranking'][0]['speed']:.2f} m/s")
    print(f"  Worst Intersection       : {results['ranking'][-1]['tls_id']}")
    print(f"    Score  : {results['ranking'][-1]['score']:.4f}")
    print(f"{'='*65}")

    env.close()
    print(f"\n[DONE] Models saved to: {cfg.model_save_path}")

if __name__ == "__main__":
    main()