"""Public simulator contracts.

The physics and replay modules remain importable by their existing paths.  The
typed adapter, authority, and failsafe objects are re-exported here so future
hardware integrations can depend on one stable import surface.
"""

from sim.adapters import Capabilities, DroneAdapter, DroneState, SimAdapter
from sim.authority import (
    AuthorityArbiter,
    AuthorityDecision,
    CommandKind,
    CommandSource,
    ControlCommand,
)
from sim.failsafe import (
    FailsafeAction,
    FailsafeConfig,
    FailsafeController,
    FailsafeDecision,
    FailsafeState,
)

__all__ = [
    "AuthorityArbiter",
    "AuthorityDecision",
    "Capabilities",
    "CommandKind",
    "CommandSource",
    "ControlCommand",
    "DroneAdapter",
    "DroneState",
    "FailsafeAction",
    "FailsafeConfig",
    "FailsafeController",
    "FailsafeDecision",
    "FailsafeState",
    "SimAdapter",
]
