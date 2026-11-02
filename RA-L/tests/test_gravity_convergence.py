"""The gravity fixed point must not stop while the axial strains are still moving: a straight robot does not rotate, so
a convergence test on rotations alone stopped after one relaxed half step (half the self-weight compression)."""
import numpy as np

from tdcr.eb_prior import predict_gravity
from tdcr.twin import Twin


def test_straight_upright_compression_does_not_depend_on_the_relaxation():
    for n in (2, 4):
        tw = Twin(n, "upright")
        dl = np.zeros((1, 3 * n))
        half = predict_gravity(tw.eb, dl, tw.stations, g_vec=tw.g_scene, mass=tw.mass, relax=0.5, tol=1e-10, iters=400)
        full = predict_gravity(tw.eb, dl, tw.stations, g_vec=tw.g_scene, mass=tw.mass, relax=1.0, tol=1e-10, iters=400)
        assert half[3]["converged"] and full[3]["converged"]
        assert abs(half[0][0, -1, 1] - full[0][0, -1, 1]) < 1e-6           # mm, tip height


def test_live_twin_compresses_the_straight_stack_fully():
    tw = Twin(2, "upright")                                                    # live tolerance 1e-4
    tight = Twin(2, "upright")
    C, *_ = predict_gravity(tight.eb, np.zeros((1, 6)), tight.stations, g_vec=tight.g_scene, mass=tight.mass,
                            relax=1.0, tol=1e-12, iters=400)
    assert abs(tw.shape(np.zeros(6), frame="scene").positions[-1, 1] - C[0, -1, 1]) < 0.005
