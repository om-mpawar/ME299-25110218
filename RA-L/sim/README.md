# SOFA ground truth

The FEM scenes that produced the std6 ground truth of the paper. They need SOFA 25.06 with SofaPython3,
SoftRobots and STLIB (not on PyPI) and the std6 meshes in `data/meshes/` (see `data/README.md`).

```bash
export SOFA_ROOT=<your SOFA 25.06 install>
source sim/sofa_env.sh
N_WORKERS=6 sim/std6_sweep.sh 2s train g0     # -> results/sweeps/std6_2s_sym_train_g0/pose_*.npz
.venv/bin/python scripts/extract_backbones.py --sections 2 --dir results/sweeps/std6_2s_sym_train_g0 \
    --tag std6_2s_sym_train_g0               # -> results/backbone_std6_2s_sym_train_g0.npz
```

Never run two SOFA sweeps on one machine at the same time: they stall each other.

| File | Role on the std6 path |
|---|---|
| `std6_sweep.sh` | the std6 preset of the old `cloud/session/sweeps.sh` + `start.sh`: mesh, cables, command design, gravity direction, solver switches |
| `shape_sweep.py` | runs one DOE row: reset, ramp the tendon pulls, settle, store every node (`--pose-dir`) |
| `multi_section_scene.py` | `createSceneRoutedN`: the routed n-module scene (also used for one module when a cable file is given) |
| `two_section_scene.py` | material, gravity, contact and solver settings and helpers shared by the scenes |
| `tendon.py` | pull-only cable actuator |
| `mass.py` | uniform or volume-distributed mass |
| `contact_gap.py` | notch gap check that decides whether a pose is redone with contact |
| `cylinder_scene.py` | imported by `shape_sweep.py` (single-section scene without a cable file; not used for std6) |

Units are mm, kg, s (force in mN, E in kPa, gravity in mm/s^2). The robot parameters (E 56000 kPa, nu 0.48,
33.7 g per module, gravity 9810, 160 mm modules, channels at 9.5 mm) are the scene defaults, read from
`tdcr/params.py`; the `TDCR_*` environment variables still override them (`TDCR_GRAVITY=0` for the gravity-free
sets, `TDCR_GRAVITY_DIR` for the mounting). The other `TDCR_*` switches in `two_section_scene.py` and
`multi_section_scene.py` (stops, struts, solver choices, legacy two-body scene) are experiment options that the std6
runs left at their defaults, except those `std6_sweep.sh` sets.
