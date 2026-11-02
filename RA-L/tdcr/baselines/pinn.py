"""
PINN baseline for the composition question: a literature-style physics-informed neural network, trained on the same
SOFA 2-module data as the identified model and tested on the same held-out sets.

Forward PINN (as Feizi et al., ICRA 2026, and Danesh et al., 2026, apply PINNs to continuum robots). A network maps
(arc position s, the 3N tendon commands, the gravity direction) to the strain field; the shape is the exact kinematic
integral of that field (as PINPE, Zhan et al., RA-L 2025). Loss = data (SOFA backbone positions at the 5 mm stations)
+ w_phys x physics residuals, evaluated on the data poses and on collocation poses that cover every mounting (physics
needs no labels):
  equilibrium   strain(s) = C(s) [tendon load + load of the weight above](s)   (Euler-Bernoulli, CAD compliance C)
  pull-only     Fischer-Burmeister(lambda_i, path shortening_i - command_i) = 0, tensions from a second head
Inverse PINN: the same, plus three learnable log-scales on the compliance of the notched cells (axial, bending x,
bending z), i.e. the PINN also identifies part of the law (the structure our identification found).

Strain parameterisation: strain = C_CAD(s) * m_theta(s, ...), the network giving a smooth effective load (an output
transform, a standard PINN device; it spares the network the notch spikes and so favours the baseline).

A network for N modules takes 3N commands, so a 2-module PINN cannot be evaluated on 4 modules: for 4 modules it is
retrained on physics alone (no data), and its accuracy and training time are reported next to the exact solution of
the same physics.

    .venv/bin/python -m tdcr.baselines.pinn --out results/pinn_baseline.json      # full grid (~1 h on CPU)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from tdcr.compliance import E_FEM, HOLES6, MASS_STD6, MODULE_L, load_set
from tdcr.eb_prior import EBPrior
from tdcr.params import STD6

torch.set_default_dtype(torch.float64)
R_D = STD6.tendon_radius_mm                             # mapping radius of the command sampler (lab mapping, mm)
S_M, S_N = 1.0, 0.1                                    # load scales: moment / E (mm^3), force / E (mm^2)
S_LAM, S_SLACK = 0.05, 0.2                             # tension / E (mm^2) and slack (mm) scales of the pull-only term
TRAIN_2S = ("std6_2s_sym_train_g0", "std6_2s_sym_train")
TESTS_2S = ("std6_2s_sym_test_g0", "std6_2s_sym_test", "std6_2s_sym_test_ghang", "std6_2s_sym_test_gx",
            "std6_2s_sym_test_gy")
TEST_4S = "std6_4s_sym_test_g0"
G_MAG = STD6.gravity_mm_s2


# ------------------------------------------------------------------ geometry (nominal plane-section beam, CAD)
class Geometry:
    """Plane-section compliance of an n-module robot on cells of `cell` mm, as torch tensors."""

    def __init__(self, n, cell=5.0):
        tag = {1: "std6_1s_sym_test_g0", 2: TRAIN_2S[0], 4: TEST_4S}[n]
        az = load_set(tag)["az"]                                         # geometry metadata, not labels
        self.n, self.cell, self.az = n, float(cell), np.asarray(az, float)
        self.eb = EBPrior(n, cell=cell, routing="paired60", tendon_az_deg=self.az, holes=HOLES6)
        eb = self.eb
        self.nc, self.nt = len(eb.yc), 3 * n
        per = self.nc // n
        cut1 = eb.cA_cad[:per] > 1.0001 * eb.cA_cad[:per].min()             # cells where a notch removes material
        self.cut = torch.tensor(np.tile(cut1, n))
        self.cA, self.cX, self.cZ = (torch.tensor(v) for v in (eb.cA, eb.cX, eb.cZ))
        self.T = torch.tensor(eb.through.astype(float))                      # (nt, nc)
        self.Dx, self.Dz = torch.tensor(eb.D[:, 0]), torch.tensor(eb.D[:, 1])
        self.yc = torch.tensor(eb.yc)
        self.length = MODULE_L * n
        self.stations = np.arange(0.0, self.length + 1e-9, 5.0)
        self.station_idx = torch.tensor([int(np.flatnonzero(np.abs(eb.edges - y) < 1e-6)[0]) for y in self.stations])
        area = eb.cell / eb.cA_cad
        self.area_frac = torch.tensor(np.concatenate([area[j * per:(j + 1) * per] / area[j * per:(j + 1) * per].sum()
                                                      for j in range(n)]))

    def mass_cells(self, mass):
        return mass * self.area_frac


def compliance_factors(geo, log_scales):
    """(3, nc) factors on the axial, bending-x, bending-z compliance: exp(scale) on notched cells, 1 elsewhere."""
    return torch.exp(log_scales[:, None] * geo.cut[None].to(log_scales.dtype))


# ------------------------------------------------------------------ differentiable shape integration
def _skew(k):
    z = torch.zeros_like(k[..., 0])
    return torch.stack([torch.stack([z, -k[..., 2], k[..., 1]], -1),
                        torch.stack([k[..., 2], z, -k[..., 0]], -1),
                        torch.stack([-k[..., 1], k[..., 0], z], -1)], -2)


def rodrigues(w):
    """Rotation vectors (..., 3) -> matrices (..., 3, 3), as tdcr.eb_prior._rodrigues."""
    th = w.norm(dim=-1, keepdim=True)
    k = w / th.clamp_min(1e-15)
    K = _skew(k)
    s, c = torch.sin(th)[..., None], (1 - torch.cos(th))[..., None]
    return torch.eye(3, dtype=w.dtype) + s * K + c * (K @ K)


def march(w, sh, cell):
    """As tdcr.eb_prior._march without shear or joint gaps: w (B, nc, 3), sh (B, nc) -> cell-centre positions
    (B, nc, 3), half-step frames (B, nc, 3, 3), edge positions (B, nc + 1, 3)."""
    B, nc = sh.shape
    Qh, Qf = rodrigues(0.5 * w), rodrigues(w)
    R = torch.eye(3, dtype=w.dtype).expand(B, 3, 3)
    p = torch.zeros(B, 3, dtype=w.dtype)
    pcs, Rhs, pes = [], [], [p]
    for c in range(nc):
        Rh = R @ Qh[:, c]
        step = Rh[:, :, 1] * (cell - sh[:, c])[:, None]
        pcs.append(p + 0.5 * step)
        p = p + step
        pes.append(p)
        Rhs.append(Rh)
        R = R @ Qf[:, c]
    return torch.stack(pcs, 1), torch.stack(Rhs, 1), torch.stack(pes, 1)


# ------------------------------------------------------------------ physics
def strains_from_loads(geo, loads):
    """Effective loads (B, nc, 3) = (m_x, m_z, n) -> rotation vectors (B, nc, 3), shortening (B, nc), nominal C."""
    kx, kz = geo.cX * loads[..., 0], geo.cZ * loads[..., 1]
    w = torch.stack([kz, torch.zeros_like(kz), -kx], -1)                 # as EBPrior.strains
    return w, geo.cA * loads[..., 2], kx, kz


def loads_from_strains(geo, w, sh):
    """Inverse of strains_from_loads (nominal compliance)."""
    return torch.stack([-w[..., 2] / geo.cX, w[..., 0] / geo.cZ, sh / geo.cA], -1)


def residuals(geo, loads, lam, dl, g, mass, log_scales):
    """Physics residuals of a candidate solution (effective loads of every cell, tendon tensions / E).
    equilibrium (B, nc, 3): network loads - physical loads (tendons + weight above, in the cell frame), with the
    physical loads scaled by the compliance factors; complementarity (B, nt): Fischer-Burmeister of (lambda, slack).
    Also returns the edge positions of the candidate shape."""
    B = loads.shape[0]
    g = g.expand(B, 3) if g.dim() == 1 else g
    w, sh, kx, kz = strains_from_loads(geo, loads)
    pc, Rc, pe = march(w, sh, geo.cell)
    # tendon loads per cell
    Tl = lam[:, :, None] * geo.T[None]                                   # (B, nt, nc)
    Mx = (Tl * geo.Dx[None, :, None]).sum(1)
    Mz = (Tl * geo.Dz[None, :, None]).sum(1)
    F = Tl.sum(1)
    # weight above each cell, its moment about the cell centre, in the cell frame (as predict_gravity)
    Fg = geo.mass_cells(mass)[None, :, None] * g[:, None, :] / E_FEM      # (B, nc, 3)
    Fsum = torch.flip(torch.cumsum(torch.flip(Fg, [1]), 1), [1])
    Fdist = torch.cat([Fsum[:, 1:], torch.zeros_like(Fsum[:, :1])], 1)
    mom = torch.flip(torch.cumsum(torch.flip(torch.cross(pc, Fg, dim=-1), [1]), 1), [1])
    mom_dist = torch.cat([mom[:, 1:], torch.zeros_like(mom[:, :1])], 1)
    Mw = mom_dist - torch.cross(pc, Fdist, dim=-1)
    Ml = torch.einsum("bcji,bcj->bci", Rc, Mw)
    Fl = torch.einsum("bcji,bcj->bci", Rc, Fdist)
    phys = torch.stack([Mx - Ml[..., 2], Mz + Ml[..., 0], F - Fl[..., 1]], -1)
    f = compliance_factors(geo, log_scales).T[None]                     # (1, nc, 3): axial, x, z -> reorder
    f = f[..., [1, 2, 0]]
    eq = (loads - f * phys) / torch.tensor([S_M, S_M, S_N])
    # pull-only compatibility: path shortening of every tendon from the candidate strains
    path = torch.einsum("ic,bc->bi", geo.T, sh) + torch.einsum("ic,bc,i->bi", geo.T, kx, geo.Dx) \
        + torch.einsum("ic,bc,i->bi", geo.T, kz, geo.Dz)
    a, b = lam / S_LAM, (path - dl) / S_SLACK
    cc = a + b - torch.sqrt(a * a + b * b + 1e-18)
    return dict(equilibrium=eq, complementarity=cc, pe=pe)


# ------------------------------------------------------------------ network
def cell_features(geo, n_freq=24):
    """Per-cell inputs: position along the robot and within the module, module index, the notched-cell flag (known
    geometry), and Fourier features of the in-module position fine enough for the notch pitch (period down to
    160/24 = 6.7 mm, the A-B notch spacing is 7.4 mm)."""
    y = geo.yc
    local = torch.remainder(y, MODULE_L) / MODULE_L
    k = 2 * math.pi * torch.arange(1, n_freq + 1, dtype=torch.float64)
    return torch.cat([(y / geo.length)[:, None], local[:, None], (torch.div(y, MODULE_L, rounding_mode="floor")
                      / geo.n)[:, None], geo.cut[:, None].to(torch.float64), torch.sin(local[:, None] * k),
                      torch.cos(local[:, None] * k)], 1)


class PINN(nn.Module):
    def __init__(self, nt, n_cell_features, width=128, depth=4):
        super().__init__()
        self.nt = nt
        cond = nt + 3
        layers, d_in = [], n_cell_features + cond
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        self.field = nn.Sequential(*layers, nn.Linear(width, 3))
        self.tension = nn.Sequential(nn.Linear(cond, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                     nn.Linear(width, nt))
        self.register_buffer("out_scale", torch.tensor([S_M, S_M, S_N]))

    def forward(self, feats, dl, g):
        """feats (nc, F) cell features, dl (B, nt) mm, g (B, 3) mm/s^2 -> loads (B, nc, 3), lam (B, nt)."""
        B, nc = dl.shape[0], feats.shape[0]
        cond = torch.cat([dl / 10.0, g / G_MAG], 1)                        # (B, nt + 3)
        x = torch.cat([feats[None].expand(B, nc, -1), cond[:, None].expand(B, nc, -1)], -1)
        loads = self.field(x) * self.out_scale
        lam = nn.functional.softplus(self.tension(cond)) * S_LAM
        return loads, lam


# ------------------------------------------------------------------ data, sampling, training
def make_data(sets, n_poses=None, seed=0):
    dl, g, C = [], [], []
    for d in sets:
        P = len(d["dl"])
        gv = np.zeros(3) if d["g"] is None else np.asarray(d["g"], float)
        dl.append(d["dl"]); C.append(d["C"]); g.append(np.repeat(gv[None], P, 0))
    dl, g, C = (torch.tensor(np.concatenate(v)) for v in (dl, g, C))
    if n_poses is not None:
        idx = torch.tensor(np.random.default_rng(seed).choice(len(dl), n_poses, replace=False))
        dl, g, C = dl[idx], g[idx], C[idx]
    return dict(dl=dl, g=g, C=C, mass=float(sets[0]["mass"]))


def sample_commands(geo, P, theta_max_deg, rng):
    """Commands of the lab mapping for random bends (uniform over the disk of radius theta_max per section)."""
    n = geo.n
    th = np.radians(theta_max_deg * np.sqrt(rng.uniform(0, 1, (P, n))))
    al = rng.uniform(0, 2 * np.pi, (P, n))
    psi = np.radians(geo.az)                                            # (n, 3)
    dl = np.zeros((P, 3 * n))
    for s in range(n):
        for k in range(3):
            dl[:, 3 * s + k] = sum(R_D * th[:, j] * np.cos(psi[s, k] - al[:, j]) for j in range(s + 1))
    return torch.tensor(dl)


def sample_gravity(P, rng, directions="all"):
    if directions == "none":
        return torch.zeros(P, 3)
    axes = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    k = rng.integers(0, len(axes) + 1, P)                               # the last index draws a random direction
    rnd = rng.normal(size=(P, 3)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    g = np.where((k == len(axes))[:, None], rnd, axes[np.minimum(k, len(axes) - 1)])
    return torch.tensor(g * G_MAG)


def train(geo, data, variant="forward", w_phys=1.0, iters=3000, batch=32, seed=0, lr=2e-3, theta_max=60.0,
          gravity="all", mass=None, log_scales=None, log_every=200, lbfgs_rounds=0, lbfgs_iters=500, n_colloc=200):
    """data None: physics only. variant 'inverse' learns the three notch log-scales (unless log_scales is given and
    fixed). Adam for `iters` steps, then `lbfgs_rounds` x `lbfgs_iters` full-batch L-BFGS iterations.
    -> dict(model, log_scales, loss, seconds)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feats = cell_features(geo)
    model = PINN(geo.nt, feats.shape[1])
    fixed = log_scales is not None
    s = torch.zeros(3) if log_scales is None else torch.as_tensor(log_scales, dtype=torch.float64).clone()
    s.requires_grad_(variant == "inverse" and not fixed)
    params = list(model.parameters()) + ([s] if s.requires_grad else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr / 20)
    mass = data["mass"] if data is not None else mass
    losses, t0 = [], time.time()
    for it in range(iters):
        dl_c = sample_commands(geo, batch, theta_max, rng)
        g_c = sample_gravity(batch, rng, gravity)
        if data is not None:
            idx = torch.tensor(rng.choice(len(data["dl"]), min(batch, len(data["dl"])), replace=False))
            dl_b, g_b = torch.cat([data["dl"][idx], dl_c]), torch.cat([data["g"][idx], g_c])
        else:
            dl_b, g_b = dl_c, g_c
        loads, lam = model(feats, dl_b, g_b)
        r = residuals(geo, loads, lam, dl_b, g_b, mass, s)
        l_phys = (r["equilibrium"] ** 2).mean() + (r["complementarity"] ** 2).mean()
        loss = w_phys * l_phys
        if data is not None:
            pe = r["pe"][:len(idx)][:, geo.station_idx]
            loss = loss + ((pe - data["C"][idx]) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  it {it:5d}  loss {float(loss):.4g}  phys {float(l_phys.detach()):.3g}  scales "
                  f"{np.round(np.exp(s.detach().numpy()), 2).tolist()}  {time.time() - t0:.0f}s", flush=True)
    if lbfgs_rounds:
        # second stage, standard for PINNs: full-batch L-BFGS on all data poses and a fixed collocation set (the
        # tip sits at the end of a long integration chain; first-order steps alone fit even 8 poses only to ~5 mm)
        dl_c, g_c = sample_commands(geo, n_colloc, theta_max, rng), sample_gravity(n_colloc, rng, gravity)
        dl_f = dl_c if data is None else torch.cat([data["dl"], dl_c])
        g_f = g_c if data is None else torch.cat([data["g"], g_c])
        nd = 0 if data is None else len(data["dl"])
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            loads, lam = model(feats, dl_f, g_f)
            r = residuals(geo, loads, lam, dl_f, g_f, mass, s)
            total = w_phys * ((r["equilibrium"] ** 2).mean() + (r["complementarity"] ** 2).mean())
            if nd:
                total = total + ((r["pe"][:nd][:, geo.station_idx] - data["C"]) ** 2).sum(-1).mean()
            total.backward()
            return total

        for k in range(lbfgs_rounds):
            val = float(lb.step(closure))
            losses.append(val)
            if log_every:
                print(f"  L-BFGS round {k}  loss {val:.4g}  scales {np.round(np.exp(s.detach().numpy()), 2).tolist()}"
                      f"  {time.time() - t0:.0f}s", flush=True)
    return dict(model=model, log_scales=s.detach(), loss=losses, seconds=time.time() - t0, variant=variant,
                w_phys=w_phys)


@torch.no_grad()
def predict(run, geo, dl, g, chunk=64):
    """-> station positions (P, K, 3) of the PINN's shape (exact kinematic integral of its strain field)."""
    out = []
    for i in range(0, len(dl), chunk):
        loads, _ = run["model"](cell_features(geo), dl[i:i + chunk], g[i:i + chunk])
        w, sh, _, _ = strains_from_loads(geo, loads)
        out.append(march(w, sh, geo.cell)[2][:, geo.station_idx])
    return torch.cat(out).numpy()


def errors(C_pred, C_true):
    tip = np.linalg.norm(C_pred[:, -1] - C_true[:, -1], axis=1)
    along = np.linalg.norm(C_pred - C_true, axis=2).mean(1)
    return dict(tip=float(tip.mean()), tip95=float(np.percentile(tip, 95)), along=float(along.mean()), poses=len(tip))


def evaluate(run, geo, tags):
    out = {}
    for t in tags:
        d = load_set(t)
        dl = torch.tensor(d["dl"])
        g = torch.zeros(len(dl), 3) if d["g"] is None else torch.tensor(np.repeat(np.asarray(d["g"], float)[None],
                                                                                  len(dl), 0))
        out[t] = errors(predict(run, geo, dl, g), d["C"])
    return out


def exact_same_physics(n, tags, log_scales):
    """The PINN's physics solved exactly by our solver (plane-section CAD x the notch factors), for reference."""
    from tdcr.eb_prior import predict_gravity
    geo = Geometry(n, cell=0.5)
    eb = geo.eb
    f = compliance_factors(geo, torch.as_tensor(log_scales, dtype=torch.float64)).numpy()
    eb.cA, eb.cX, eb.cZ = eb.cA * f[0], eb.cX * f[1], eb.cZ * f[2]
    eb.assemble()
    out = {}
    for t in tags:
        d = load_set(t)
        if d["g"] is None:
            C = eb.predict(d["dl"], geo.stations)[0]
        else:
            C = predict_gravity(eb, d["dl"], geo.stations, g_vec=d["g"], E=E_FEM, mass=d["mass"], tol=1e-8,
                                iters=400)[0]
        out[t] = errors(C, d["C"])
    return out


def main(argv=None):                                                    # pragma: no cover (long runs)
    """Stage 2s: train one PINN on the 2-module SOFA data (85 % train, 15 % validation for choosing w_phys) and test it
    on every 2-module set. Stage 4s (--four-from <stage-2s json>): retrain on 4-module physics only, with that run's
    notch factors, and test on the 4-module set; also solve the same physics exactly for reference."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("forward", "inverse"), default="forward")
    ap.add_argument("--w-phys", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2000, help="Adam steps")
    ap.add_argument("--lbfgs-rounds", type=int, default=3)
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--four-from", help="stage 4s: a stage-2s result json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    kw = dict(seed=a.seed, lbfgs_rounds=a.lbfgs_rounds, lbfgs_iters=a.lbfgs_iters)
    if a.four_from:
        src = json.load(open(a.four_from))
        ls = np.log(np.asarray(src["notch_factors"]))
        geo4 = Geometry(4)
        print(f"[4 modules, physics only, notch factors {np.round(src['notch_factors'], 2).tolist()}]", flush=True)
        run4 = train(geo4, None, "forward", 1.0, a.iters, theta_max=45.0, gravity="none", mass=MASS_STD6,
                     log_scales=ls, **kw)
        res = dict(source=a.four_from, variant=src["variant"], w_phys=src["w_phys"], train_seconds_4s=run4["seconds"],
                   notch_factors=src["notch_factors"], tests_4s=evaluate(run4, geo4, [TEST_4S]),
                   exact_same_physics=exact_same_physics(4, [TEST_4S], ls))
    else:
        data = make_data([load_set(t) for t in TRAIN_2S])
        P = len(data["dl"])
        perm = np.random.default_rng(a.seed).permutation(P)
        nv = int(round(a.val_frac * P))
        val = {k: v[torch.tensor(perm[:nv])] for k, v in data.items() if k != "mass"}
        tr = {k: v[torch.tensor(perm[nv:])] for k, v in data.items() if k != "mass"}
        tr["mass"] = val["mass"] = data["mass"]
        geo2 = Geometry(2)
        print(f"[{a.variant} w_phys={a.w_phys}] 2 modules: {len(tr['dl'])} training, {nv} validation poses", flush=True)
        run = train(geo2, tr, a.variant, a.w_phys, a.iters, **kw)
        ls = run["log_scales"].numpy()
        res = dict(variant=a.variant, w_phys=a.w_phys, seed=a.seed, iters=a.iters, lbfgs=[a.lbfgs_rounds, a.lbfgs_iters],
                   train_seconds_2s=run["seconds"], notch_factors=np.exp(ls).tolist(),
                   validation=errors(predict(run, geo2, val["dl"], val["g"]), val["C"].numpy()),
                   tests_2s=evaluate(run, geo2, TESTS_2S), exact_same_physics=exact_same_physics(2, TESTS_2S, ls))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
