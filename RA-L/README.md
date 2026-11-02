# tdcr-twin

Sensor-free digital twin of a modular, notched tendon-driven continuum robot (TDCR): tendon displacements plus the
robot's mounting give the whole backbone shape in a few milliseconds. The code and data behind the RA-L paper in
`paper/`.

The model, in layers:

- **L0, physics prior** (`tdcr/eb_prior.py`): a variable-section Euler-Bernoulli rod whose cross-section follows the
  CAD of the notched module, pull-only tendons (a convex QP, or an LCP with capstan friction), coupling between
  sections (Camarillo), gravity in any direction by a fixed point, optional Timoshenko shear, joint gaps and point
  masses.
- **L1, identified module** (`tdcr/compliance.py`): a compliance profile along the module (three log-scales on the
  notched cells and three smooth fields), identified once from 1- and 2-module SOFA FEM data and reused unchanged
  for any number of modules (`results/mc_std6.json`).
- **L2, hardware calibration** (`tdcr/calibrate.py`): five physical numbers from about 15 motion-capture poses.
- **L3, learned residual** (`scripts/n_transfer.py`): a GP on the local segment increments, to test what learning
  adds and where it stops transferring.
- **Twin** (`tdcr/twin.py`): the real-time interface, `Twin(n_modules, mounting).shape(dl)`.

The ground truth is SOFA FEM of the real robot ("std6": six channels, 160 mm modules; parameters in
`tdcr/params.py`, the only place they are defined).

## Setup

```bash
python3.12 -m venv .venv                      # 3.12, like SOFA 25.06's SofaPython3
.venv/bin/pip install -e ".[dev]"             # add ,pinn for the PINN baseline (torch, CPU wheel:
                                              #   --extra-index-url https://download.pytorch.org/whl/cpu)
.venv/bin/python -m pytest -q
```

SOFA is needed only to re-run the FEM ground truth: see `sim/README.md`.

## Using the twin

```python
from tdcr.twin import Twin
tw = Twin(4, "hanging")                 # identified module (results/mc_std6.json), no hardware calibration
s = tw.shape([3.0, -1.5, -1.5] + [0.0] * 9)    # mm per tendon, + wound in, lab tendon order
s.positions[-1], s.converged, s.solve_ms
```

## Reproducing the paper's simulation results

One command regenerates every simulation-derived number and figure of `paper/main.tex` and checks each against the
printed value:

```bash
.venv/bin/python scripts/reproduce.py          # fast mode, ~5 min on a laptop: everything but the multi-hour jobs
.venv/bin/python scripts/reproduce.py --full   # also n_transfer (~2 h, 8 threads) and the PINN baseline (hours)
.venv/bin/python scripts/reproduce.py --list   # the steps, their commands and their outputs
```

It runs the producers below one at a time (logs in `results/logs/`), writes their result files in place under
`results/` (so `git diff results/` shows what moved), draws Fig. 1 and 2 into `results/figures/` and compares them
pixel by pixel with `paper/figures/`, and finally writes **`results/CLAIMS.md`** (readable) and `results/claims.json`:
for every number in the paper, its line, the printed value, the regenerated value, the command and a status
(reproduced / differs / stored, not re-run / not reproducible here / hardware pending), with the physics audit's
corrections as notes. No SOFA, hardware or network is needed; Table II's first and last rows need the `pinn` extra.

| Paper item | Step (`scripts/...`) | Writes |
|---|---|---|
| Table I (identified physics), x20.2 / 5.9 / 5.3 | `module_compliance.py` (~1 min) | `results/mc_std6.json` |
| Table I (learned column), Fig. 2 points, residual variants, coverage | `n_transfer.py` (~2 h, long) | `results/n_transfer.json` |
| Table I re-evaluated (comment under the table); at a tight tolerance | `mc_reeval.py`, `tol_reeval.py` | `results/mc_std6_reeval.json`, `results/tol_reeval.json` |
| Table II PINN rows | `pinn_stage1.sh`, `pinn_stage2.sh` (hours, long) | `results/pinn/2s_*.json`, `4s_*.json` |
| Table II exact-solver rows, x20.2 / 5.8 / 5.3 in 80 s | `exact_global3.py` | `results/pinn/exact_global3_2s.json`, `pinn_laws_exact.json` |
| Prop. 1 example, Prop. 3, tendon stretch, Remark 1 hysteresis, sag, mounting error | `physics_props.py` | `results/physics_props.json` |
| Prop. 1 (4.5 mm), Prop. 2 Jacobian, run-time medians and p95 | `study_guide_figs.py` | `results/study_guide/numbers.json` |
| Slowest call, Cosserat time per pose | `runtime.py` | `results/runtime.json` |
| D30 vs D40 sensitivity | `geometry_sensitivity.py` | `results/geometry_sensitivity.json` |
| Calibration on synthetic data (0.003, 0.244, mass drift) | `calib_synthetic.py` | `results/calib_synthetic.json` |
| Ablation footnote (older robot) | `mc_ablation.py` | `results/mc_ablation.json` |
| FEM nodal mass vs the part ("2 %") | `fem_mass.py` (needs the std6 meshes, `data/README.md`) | `results/fem_mass.json` |
| FEM mesh error (0.6 mm) | `mesh_error.py` (stub: needs SOFA; collects the stored old-robot values) | `results/mesh_error.json` |
| Fig. 1, Fig. 2 | `fig_profile.py`, `fig_transfer.py` (`--out-dir`; default `paper/figures/`) | `results/figures/` |
| every number of the paper | `claims.py` | `results/claims.json`, `results/CLAIMS.md` |

Timings depend on the machine and its load; run on an otherwise idle computer. `module_compliance` runs with
`OMP_NUM_THREADS=1`, which makes `results/mc_std6.json` bit-reproducible.

`PACKAGING_NOTES.md` maps every file of the old research repository to its place here and lists the open
questions for the physics audit.

## Layout

| Folder | What |
|---|---|
| `tdcr/` | the library: `params`, `paths`, `eb_prior`, `compliance`, `backbones` (FEM data, metrics, residual GP, backbone extraction), `twin`, `calibrate`, `doe` (command designs), `hw_eval`; `baselines/` (Cosserat rod, PINN); `hardware/` (bench tools: servos, homing, pose lists, motion capture, live loop); constant-curvature and Cosserat kinematics (`cc_kinematics`, `multisection`, `tendon_mapping`, `cosserat`) |
| `scripts/` | the paper pipeline (producers and figures) |
| `sim/` | SOFA scenes of the ground truth |
| `paper/` | the RA-L manuscript and its figures |
| `data/` | CAD parameters, cable paths, contact surfaces, bench data; `data/README.md` says where the large inputs live |
| `results/` | the FEM backbones and result files the paper uses |
| `tests/` | `pytest` |
