"""The moving coop simulator must refuse estimators it cannot share honestly.

Only the CV estimator exposes a full 6-state track (``.tracker`` with P). For
ca, ct and imm, ``CoopSwarmSimulator._full_track`` used to fill the velocity
block of the shared covariance with an identity placeholder, so a run labelled
with one of those estimators fused and shared a made-up velocity uncertainty.
The config now rejects them instead of running them that way.
"""
from __future__ import annotations

import pytest

from sim.coop.simulator import CoopSwarmConfig


@pytest.mark.parametrize("model", ["ca", "ct", "imm"])
def test_non_cv_estimator_is_rejected(model):
    with pytest.raises(ValueError, match="only supports estimator 'cv'"):
        CoopSwarmConfig(estimator=model)


def test_cv_estimator_is_still_accepted():
    assert CoopSwarmConfig(estimator="cv").estimator == "cv"


def test_unknown_estimator_is_still_rejected():
    with pytest.raises(ValueError):
        CoopSwarmConfig(estimator="kalman9000")
