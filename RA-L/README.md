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

| Paper item | Command | Writes |
|---|---|---|
| Table I (identified physics) | `.venv/bin/python scripts/module_compliance.py --train std6_1s_sym_train_g0 std6_2s_sym_train_g0 std6_1s_sym_train std6_2s_sym_train --out results/mc_std6.json` (~2 min) | `results/mc_std6.json` |
| Table I (learned column), Fig. 2 points | `OMP_NUM_THREADS=8 .venv/bin/python scripts/n_transfer.py --mc results/mc_std6.json` | `results/n_transfer.json` |
| Fig. 2 | `.venv/bin/python scripts/fig_transfer.py` | `paper/figures/fig_transfer.pdf` |
| Fig. 1 | `.venv/bin/python scripts/fig_profile.py` | `paper/figures/fig_profile.pdf` |
| Prop. 1, Prop. 2, run times | `.venv/bin/python scripts/study_guide_figs.py` | `results/study_guide/numbers.json` |
| Table II (PINN) | `scripts/pinn_stage1.sh`, then `scripts/pinn_stage2.sh` (needs the `pinn` extra; hours on CPU) | `results/pinn/` |

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
