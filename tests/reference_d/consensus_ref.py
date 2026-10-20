"""Independent consensus and graph references."""
import numpy as np


def laplacian_ref(adjacency):
    A = np.asarray(adjacency, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def consensus_eigh(L, eps, x0, k):
    L = np.asarray(L, dtype=float); x0 = np.asarray(x0, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(L)
    M = eigvecs @ np.diag((1.0 - eps * eigvals) ** k) @ eigvecs.T
    return M @ x0


def consensus_matrix_power(L, eps, x0, k):
    L = np.asarray(L, dtype=float); x0 = np.asarray(x0, dtype=float)
    return np.linalg.matrix_power(np.eye(L.shape[0]) - eps * L, int(k)) @ x0


def consensus_ref(L, eps, x0, k):
    """Return both independent forms of the closed form consensus iterate."""
    return consensus_eigh(L, eps, x0, k), consensus_matrix_power(L, eps, x0, k)


def rate_ref(L, eps):
    """Slowest per-step factor of the disagreement: every mode except the single all-equal one.

    A disconnected graph has more than one zero eigenvalue; those extra modes never decay (factor 1).
    """
    eigvals = np.sort(np.linalg.eigvalsh(np.asarray(L, dtype=float)))
    rest = eigvals[1:]
    return float(np.max(np.abs(1.0 - eps * rest))) if rest.size else 0.0
