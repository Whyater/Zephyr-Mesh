from types import SimpleNamespace

from sim.radio.mac import MacConfig, PhyConfig
from sim.radio.network import Network, NetworkConfig


class FlatChannel:
    sensitivity_dbm = -98.0
    cfg = SimpleNamespace(sensitivity_dbm=-98.0, noise_floor_dbm=-95.0)

    def sample(self, tx, rx, distance_m, t_s, **kwargs):
        if tx == "gs" or rx == "gs":
            good = True
        elif tx in ("d", "r") or rx in ("d", "r"):
            good = True
        else:
            good = abs(int(tx[1:]) - int(rx[1:])) <= 1
        return SimpleNamespace(rssi_dbm=-40.0 if good else -150.0, snr_db=55.0, per=0.0)

    def received_power_dbm(self, tx, rx, distance_m, t_s, **kwargs):
        return self.sample(tx, rx, distance_m, t_s).rssi_dbm


def _chain(n=4, topology="flood", seed=4, **kwargs):
    pos = {f"n{i}": (float(i), 0.0, 0.0) for i in range(n)}
    phy = PhyConfig(cw_min=0)
    mac = MacConfig(backoff=False, queue_capacity=100)
    net = Network(pos, FlatChannel(), phy, mac,
                  NetworkConfig(topology, gateway=f"n{n-1}", ttl=n, **kwargs), seed=seed)
    net.originate("n0", 0.0, 10)
    net.run_until(1.0)
    return net


def test_flood_chain_hops_and_transmissions():
    net = _chain()
    assert [net.receptions[('n0', 0, f'n{i}')].hops for i in range(1, 4)] == [1, 2, 3]
    assert net.forwarded_frames() == 2
    assert net.data_transmissions() == 3


def test_relay_direct_copy_is_deduplicated():
    pos = {"d": (0.0, 0.0, 0.0), "r": (1.0, 0.0, 0.0), "gs": (2.0, 0.0, 0.0)}
    phy = PhyConfig(cw_min=0)
    mac = MacConfig(backoff=False, queue_capacity=20)
    net = Network(pos, FlatChannel(), phy, mac, NetworkConfig("relay", relay="r"), seed=2)
    net.originate("d", 0.0, 10)
    net.run_until(1.0)
    assert net.receptions[("d", 0, "gs")].hops == 1


def test_same_seed_is_deterministic():
    a = _chain(seed=9, topology="gossip", gossip_g=0.5)
    b = _chain(seed=9, topology="gossip", gossip_g=0.5)
    assert a.receptions == b.receptions
    variants = [_chain(seed=s, topology="gossip", gossip_g=0.5).receptions for s in range(10, 15)]
    assert any(r != a.receptions for r in variants)


def test_gossip_line_probability():
    reached = 0
    for seed in range(200):
        net = _chain(seed=seed, topology="gossip", gossip_g=0.5)
        reached += ("n0", 0, "n3") in net.receptions
    assert abs(reached / 200.0 - 0.25) < 0.12


def test_summary_numbers():
    net = _chain(n=3)
    out = net.summary(0.0, 1.0, dest="n2")
    assert out["n_messages"] == 1
    assert out["n_delivered"] == 1
    assert out["mean_hops"] == 2
    assert out["latencies_s"] == sorted(out["latencies_s"])
