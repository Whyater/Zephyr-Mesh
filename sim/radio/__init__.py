"""Radio and network layer for the Zephyr Mesh simulator (simulation only).

The older ``sim.link.SimulatedLink`` treats delay and loss as knobs.  This
package explains where they come from:

- ``channel``: distance and obstacles -> signal strength -> packet errors;
- ``mac``: number of senders and send rate -> airtime, collisions, queueing;
- ``seq`` and ``replay``: sequence numbers, duplicates, reordering, replays;
- ``network``: topologies, relays and flooding;
- ``heartbeat`` and ``interference``: lost-link and degraded-link detection;
- ``model``: one interface that wraps either the old abstract link or this
  radio model, so callers switch with one config field.

Every default that comes from outside is literature-derived (see
``params``), never measured.  No ESP32 hardware has been tested.
"""
