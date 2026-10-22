"""Brute-force fixed-lag retrodiction reference."""
import numpy as np

def replay_packets(packets, window, q_acc=0.1, r_std=0.05, start_time=None, start_position=None):
    """Replay accepted packets from scratch after each arrival; return state history.

    Packets are (stamp, z, receive_time). Acceptance and duplicate/window rules follow the fixed-lag retrodiction rule.
    """
    from .ref_cv import predict, _update
    W=float(window); accepted=[]; history=[]; newest=-np.inf; outputs=[]
    # Fixed filter start (after the t = 0 correction): an optional constant-velocity prior at start_time.
    origin_time = None if start_time is None else float(start_time)
    origin_position = None if start_position is None else np.asarray(start_position, float).copy()

    def replay_current():
        """Replay retained entries from the original prior."""
        if origin_time is None:
            return None
        x = np.zeros(6, dtype=float)
        x[:3] = origin_position
        P = np.eye(6, dtype=float)
        current_time = origin_time
        for stamp, value, _ in sorted(accepted, key=lambda a: a[0]):
            x, P = predict(x, P, current_time, stamp, q_acc)
            x, P = _update(x, P, value, r_std)
            current_time = stamp
        return x, P

    for packet in packets:
        s=float(packet[0]); z=np.asarray(packet[1],float); recv=packet[2] if len(packet)>2 else None
        drop=False
        stamps=[a[0] for a in accepted]
        if any(abs(s-a)<1e-12 for a in stamps):
            drop=True
        elif s>newest:
            if origin_time is None:
                origin_time = s
                origin_position = z.copy()
            entry=(s,z,recv)
            accepted.append(entry); history.append(entry); newest=s
        elif s<newest and s>=newest-W:
            history.sort(key=lambda a:a[0])
            i = next(i for i, a in enumerate(history) if a[0] > s)
            before_stamp = origin_time if i == 0 else history[i-1][0]
            if s > before_stamp or (i == 0 and s == before_stamp):  # t = 0 correction: fuse into a fresh filter
                entry=(s,z,recv)
                accepted.append(entry); history.insert(i, entry)
            else:
                drop=True
        else: drop=True
        if not drop:
            cutoff = newest - W
            history = [a for a in history if a[0] >= cutoff or a[0] == newest]
        current = replay_current()
        outputs.append(current if current is not None else (None, None))
    return outputs

def run_retro(packets,window,q_acc=0.1,r_std=0.05,start_time=None,start_position=None):
    """Replay packets after each arrival with fixed-lag retrodiction."""
    return replay_packets(packets,window,q_acc,r_std,start_time,start_position)  # t = 0 correction: pass the filter start through
retrodict = replay_packets
