"""Network topologies built on the deterministic radio MAC simulator."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .mac import BROADCAST, Frame, MacSimulator, frame_bytes

TOPOLOGIES = ("star", "broadcast", "flood", "gossip", "relay")

@dataclass(frozen=True)
class NetworkConfig:
    topology: str
    gateway: str = "gs"
    relay: str | None = None
    ttl: int = 4
    gossip_g: float = 0.5
    forward_delay_s: float = 0.0
    gateway_forwards: bool = False

@dataclass(frozen=True)
class Reception:
    origin: str
    origin_seq: int
    rx: str
    created_s: float
    first_rx_s: float
    hops: int

@dataclass(frozen=True)
class Message:
    origin: str
    origin_seq: int
    created_s: float

class Network:
    """Simulate message dissemination over a MAC, with deterministic forwarding."""
    def __init__(self, positions, channel, phy, mac_cfg, net_cfg: NetworkConfig, *, seed: int,
                 walls=None, noise_floor_fn=None):
        if net_cfg.topology not in TOPOLOGIES:
            raise ValueError("unknown topology")
        if net_cfg.gateway not in positions:
            raise ValueError("gateway must be in positions")
        if net_cfg.relay is not None and net_cfg.relay not in positions:
            raise ValueError("relay must be in positions")
        if net_cfg.topology == "relay" and net_cfg.relay is None:
            raise ValueError("relay topology requires relay")
        if not isinstance(net_cfg.ttl, int) or net_cfg.ttl < 0:
            raise ValueError("ttl must be nonnegative")
        if not math.isfinite(net_cfg.gossip_g) or not 0 <= net_cfg.gossip_g <= 1:
            raise ValueError("gossip_g must be in [0, 1]")
        if not math.isfinite(net_cfg.forward_delay_s) or net_cfg.forward_delay_s < 0:
            raise ValueError("forward_delay_s must be nonnegative")
        self.positions = {str(k): tuple(v) for k, v in positions.items()}
        self.net_cfg = net_cfg
        self.sim = MacSimulator(self.positions, channel, phy, mac_cfg, seed=seed,
                                walls=walls, noise_floor_fn=noise_floor_fn)
        # The MAC uses children 0 to 2 of SeedSequence(seed); the network's own
        # streams take children 3 (gossip choices) and 4 (origination phases and jitter).
        streams = np.random.SeedSequence(seed).spawn(5)
        self._gossip_rng = np.random.default_rng(streams[3])
        self._origin_rng = np.random.default_rng(streams[4])
        self._next_seq: dict[str, int] = {n: 0 for n in sorted(self.positions)}
        self.messages: dict[tuple[str, int], Message] = {}
        self.receptions: dict[tuple[str, int, str], Reception] = {}
        self._forwarded = 0
        self._periodic: list[dict] = []
        self.sim.on_receive = self._on_receive

    def _validate_node(self, node: str) -> None:
        if node not in self.positions:
            raise ValueError("unknown node")

    def _frame_dst(self, node: str) -> str:
        return self.net_cfg.gateway if self.net_cfg.topology == "star" else BROADCAST

    def originate(self, node: str, t_s: float, payload_bytes: int) -> tuple[str, int]:
        self._validate_node(node)
        if not math.isfinite(t_s) or t_s < self.sim.now_s - 1e-12:
            raise ValueError("invalid origination time")
        frame_bytes(payload_bytes, self.sim.phy)
        seq = self._next_seq[node]
        self._next_seq[node] = seq + 1
        msg = Message(node, seq, float(t_s))
        self.messages[(node, seq)] = msg
        frame = Frame(node, self._frame_dst(node), payload_bytes, float(t_s),
                      seq=seq, origin=node, origin_seq=seq, ttl=0, hops=0)
        self.sim.enqueue(node, frame, float(t_s))
        return (node, seq)

    def add_periodic_origin(self, node, rate_hz, payload_bytes, *, start_s=0.0,
                            stop_s=math.inf, phase_s=None, jitter_s=0.0) -> None:
        self._validate_node(node)
        if (not math.isfinite(rate_hz) or rate_hz <= 0 or not math.isfinite(start_s)
                or start_s < self.sim.now_s or math.isnan(stop_s)):
            raise ValueError("invalid periodic source")
        if stop_s < start_s or jitter_s < 0 or not math.isfinite(jitter_s):
            raise ValueError("invalid periodic source")
        period = 1.0 / rate_hz
        if phase_s is None:
            phase_s = float(self._origin_rng.uniform(0.0, period))
        if not (0 <= phase_s < period):
            raise ValueError("phase_s must be in one period")
        frame_bytes(payload_bytes, self.sim.phy)
        self._periodic.append({"node": node, "rate": float(rate_hz), "payload": int(payload_bytes),
                               "start": float(start_s), "stop": float(stop_s), "phase": float(phase_s),
                               "jitter": float(jitter_s), "k": 0})

    def _on_receive(self, sim, rx: str, frame: Frame, t: float) -> None:
        if frame.origin is None or frame.origin_seq is None:
            return
        origin, oseq = frame.origin, frame.origin_seq
        if rx == origin:
            return
        key = (origin, oseq, rx)
        if key in self.receptions:
            return
        hops = int(frame.hops) + 1
        topo = self.net_cfg.topology
        if topo in ("flood", "gossip") and hops > self.net_cfg.ttl:
            return
        self.receptions[key] = Reception(origin, oseq, rx, frame.created_s, float(t), hops)
        if topo == "relay":
            relay = self.net_cfg.relay
            if rx == relay and origin not in (relay, self.net_cfg.gateway):
                f = Frame(relay, self.net_cfg.gateway, frame.payload_bytes, frame.created_s,
                          seq=oseq, origin=origin, origin_seq=oseq, ttl=frame.ttl, hops=1)
                sim.enqueue(relay, f, t + self.net_cfg.forward_delay_s)
                self._forwarded += 1
            return
        if topo not in ("flood", "gossip"):
            return
        if hops >= self.net_cfg.ttl:
            return
        if rx == self.net_cfg.gateway and not self.net_cfg.gateway_forwards:
            return
        if topo == "gossip" and self._gossip_rng.random() >= self.net_cfg.gossip_g:
            return
        f = Frame(rx, BROADCAST, frame.payload_bytes, frame.created_s,
                  seq=oseq, origin=origin, origin_seq=oseq, ttl=frame.ttl, hops=hops)
        sim.enqueue(rx, f, t + self.net_cfg.forward_delay_s)
        self._forwarded += 1

    def run_until(self, t_s: float) -> None:
        if not math.isfinite(t_s) or t_s < self.sim.now_s - 1e-12:
            raise ValueError("run_until time must be finite and nondecreasing")
        for spec in self._periodic:
            period = 1.0 / spec["rate"]
            while spec["start"] + spec["phase"] + spec["k"] * period <= t_s + 1e-12:
                base = spec["start"] + spec["phase"] + spec["k"] * period
                spec["k"] += 1
                send = base + (float(self._origin_rng.uniform(0.0, spec["jitter"])) if spec["jitter"] else 0.0)
                if send <= spec["stop"] + 1e-12:
                    self.originate(spec["node"], send, spec["payload"])
        self.sim.run_until(t_s)

    def data_transmissions(self) -> int:
        return int(sum(r.tries for r in self.sim.frame_records.values()))

    def forwarded_frames(self) -> int:
        return self._forwarded

    def summary(self, t0_s: float, t1_s: float, *, dest: str | None = None) -> dict:
        if not math.isfinite(t0_s) or not math.isfinite(t1_s) or t1_s < t0_s:
            raise ValueError("invalid summary interval")
        dest = self.net_cfg.gateway if dest is None else dest
        self._validate_node(dest)
        selected = [m for m in self.messages.values() if t0_s <= m.created_s < t1_s]
        lats, hops = [], []
        for m in selected:
            rec = self.receptions.get((m.origin, m.origin_seq, dest))
            if rec is not None:
                lats.append(rec.first_rx_s - m.created_s)
                hops.append(rec.hops)
        n = len(selected)
        return {
            "n_messages": n,
            "n_delivered": len(lats),
            "delivery_ratio": (len(lats) / n if n else 0.0),
            "latencies_s": sorted(lats),
            "mean_hops": (sum(hops) / len(hops) if hops else math.nan),
            "transmissions_per_message": (self.data_transmissions() / len(self.messages) if self.messages else 0.0),
            "busy_fraction": self.sim.channel_busy_fraction(t0_s, t1_s),
        }
