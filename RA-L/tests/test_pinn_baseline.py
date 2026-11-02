"""PINN baseline (analysis/pinn_baseline.py): its differentiable pieces must reproduce the exact model, or a poor PINN
result could be a bug rather than a finding."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from analysis import pinn_baseline as pb                                   # noqa: E402
from analysis.module_compliance import load_set                            # noqa: E402
from tdcr.eb_prior import _march, predict_gravity                          # noqa: E402


@pytest.fixture(scope="module")
def geo():
    return pb.Geometry(2)


def test_torch_march_matches_the_compiled_march(geo):
    rng = np.random.default_rng(0)
    nc = geo.nc
    w = np.zeros((3, nc, 3))
    w[..., 0], w[..., 2] = rng.normal(0, 3e-3, (3, nc)), rng.normal(0, 3e-3, (3, nc))
    sh = rng.uniform(0, 1e-3, (3, nc))
    _, _, pe, Re = _march(w, sh, np.zeros((3, nc, 2)), float(geo.cell), np.zeros(nc))
    pc_t, Rc_t, pe_t = pb.march(torch.tensor(w), torch.tensor(sh), geo.cell)
    np.testing.assert_allclose(pe_t.numpy(), pe, atol=1e-9)


def test_physics_residual_vanishes_on_the_exact_solution(geo):
    """Loads read back from the exact gravity solve satisfy equilibrium and pull-only compatibility."""
    d = load_set("std6_2s_sym_train")
    dl = d["dl"][:4]
    eb = geo.eb
    C, R, lam, info = predict_gravity(eb, dl, geo.stations, g_vec=d["g"], E=pb.E_FEM, mass=d["mass"], tol=1e-12,
                                      iters=600)
    assert info["converged"]
    w_t, sh_t = eb.strains(lam)
    w = w_t + info["state"]["wg"]
    sh = sh_t + info["state"]["shg"]
    loads = pb.loads_from_strains(geo, torch.tensor(w), torch.tensor(sh))          # effective loads (m_x, m_z, n)
    res = pb.residuals(geo, loads, torch.tensor(lam), torch.tensor(dl), torch.tensor(d["g"], dtype=torch.float64),
                       d["mass"], log_scales=torch.zeros(3, dtype=torch.float64))
    assert res["equilibrium"].abs().max().item() < 1e-6
    assert res["complementarity"].abs().max().item() < 1e-6


def test_inverse_scales_act_on_notched_cells_only(geo):
    s = torch.tensor([np.log(20.0), np.log(5.0), np.log(5.0)], dtype=torch.float64)
    f = pb.compliance_factors(geo, s)
    plain = ~geo.cut
    assert torch.allclose(f[:, plain], torch.ones_like(f[:, plain]))
    assert torch.allclose(f[0, geo.cut], torch.full_like(f[0, geo.cut], 20.0))


def test_short_training_reduces_the_loss(geo):
    d = load_set("std6_2s_sym_train")
    data = pb.make_data([d], n_poses=16)
    run = pb.train(geo, data, variant="inverse", w_phys=1.0, iters=30, batch=8, seed=0, log_every=0)
    assert run["loss"][-1] < run["loss"][0]
