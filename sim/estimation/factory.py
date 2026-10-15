"""Build any estimator from a model name.

The engine asks for "cv", "ca", "ct" or "imm" and gets an Estimator started
at the target's true start with zero velocity. Each model module is imported
only when it is asked for, so a sweep that uses only the CV arms never loads
the others.
"""
from __future__ import annotations

from .base import MODELS, Estimator, EstimatorSettings


def _estimator_class(model: str):
    if model == "cv":
        from .cv import CVEstimator
        return CVEstimator
    if model == "ca":
        from .ca import CAEstimator
        return CAEstimator
    if model == "ct":
        from .ct import CTEstimator
        return CTEstimator
    if model == "imm":
        from .imm import IMMEstimator
        return IMMEstimator
    raise ValueError(f"model must be one of {MODELS}")


def make_estimator(model: str, settings: EstimatorSettings, start_position, *, q_factor: float = 1.0,
                   r_std_m: float | None = None, model_options=()) -> Estimator:
    """New estimator for ``model``.

    ``q_factor`` multiplies every process-noise intensity of the model
    (q_acc, q_jerk and q_turn). ``r_std_m`` None means
    ``settings.measurement_std_m``. ``model_options`` is a dict or a sequence
    of (key, value) pairs passed on as keywords, for example
    (("known_turn_rate", 1.0),) for the CT exact-model check.
    """
    cls = _estimator_class(model)
    options = dict(model_options)
    return cls(start_position, settings, q_factor=q_factor, r_std_m=r_std_m, **options)
