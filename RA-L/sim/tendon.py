"""
Pull-only tendons for every SOFA scene.

A SoftRobots CableConstraint holds its commanded displacement in BOTH directions
unless minForce is set ("If unspecified no minimum value will be considered"),
so a cable can push on the body. A real tendon cannot: it pulls, or goes slack.
Every scene creates its cables through this wrapper, which sets minForce = 0.

Found 2026-09-29: with the bilateral default, 403 of the 408 four-section design
poses were held in shape by at least one pushing cable. Sweeps run before this
date (cylinder, two-section, two-section CAD, the first four-section cloud run)
used pushing cables.

TDCR_ALLOW_PUSH=1 restores the old bilateral cables, for comparisons only.
"""
import os

from softrobots.actuators import PullingCable as _PullingCable

ALLOW_PUSH = os.environ.get("TDCR_ALLOW_PUSH") == "1"


def PullingCable(*args, **kwargs):
    """softrobots.actuators.PullingCable, pull-only (minForce = 0)."""
    node = _PullingCable(*args, **kwargs)
    if not ALLOW_PUSH:
        node.CableConstraint.minForce = 0.0
    return node
