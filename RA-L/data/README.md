# Data

What is here, and where the large inputs that are not here live.

## In the repository

| Path | What |
|---|---|
| `cad/std_tdcr_L150.json` | Parameters of the standard notched module (the CAD the prior integrates; `tdcr.eb_prior.CAD_JSON`). std6 uses the same geometry, with its six channel holes cut in the model (`holes=`). |
| `cad/std6_tdcr_L150.json` | The real robot's (std6) CAD parameters: same module, six channels. |
| `cad/std6_tdcr_L150_D40.json` | The 40 mm disc variant (the D30 vs D40 comparison, `tests/test_design_profiles.py`, study-guide figure `disc`). |
| `cables/std6_{one,two,four}_section_sym.cables.json` | Tendon paths of the simulated std6 robot (lab tendon numbering). The single source of the routing: `tdcr.eb_prior.geometry_from_cables`, the SOFA scenes, `tdcr.doe`. |
| `contacts/std6_*_sym.contact.npz` | Facing notch surfaces for disc contact in the SOFA scenes (`sim/two_section_scene.add_contact_pipeline`, `sim/contact_gap.py`). |
| `hardware/measurements.json` | Bench measurements of the printed robot (masses, joint parts). |
| `hardware/poses/` | Command lists for the first hardware sessions (`python -m tdcr.hardware.pose_lists`). |

FEM results the paper uses are in `results/` (see the top-level README).

## Not in the repository

| What | Size | Where |
|---|---|---|
| std6 tetrahedral meshes `std6_{one,two,four}_section_sym.vtk`, needed to re-run the SOFA ground truth (`sim/std6_sweep.sh`), to re-extract backbones (`scripts/extract_backbones.py`) by `tests/test_scene_mass.py` (skipped without them) and by `scripts/fem_mass.py` (the paper's "2 %" FEM mass number; reported as not reproducible without the two-module mesh) | 28 MB | the old repository, `TDCR_updated/data/meshes/` (git-tracked there). Copy them to `data/meshes/`. |
| Raw std6 SOFA sweeps (every mesh node of every pose, `results/sweeps/std6_*/pose_*.npz`), from which `results/backbone_std6_*.npz` were extracted | ~GBs | the lab PC that ran the std6 queue (`results/sweeps/std6_*` there). NOT on this laptop, and NOT in a GitHub release: the release `sim-data-2026-10-11` named in the work plan does not exist on `om-mpawar/virtual-sensor` (checked 2 Nov 2026; only `sim-data-2026-10-03` and `sim-data-2026-10-04` exist). |
| Raw sweeps of the older 3-channel robot (`std_*`, `*_cad_sym*`) | ~1.8 GB | GitHub releases `sim-data-2026-10-03` and `sim-data-2026-10-04` of `om-mpawar/virtual-sensor`, and the old repository's `results/sweeps/`. |
| Older meshes, CAD (STEP/STL), print files, cable files of earlier robots | 115 MB+ | the old repository (`data/meshes`, `data/cables`, `cad/`). |
| Hardware sessions (servo logs, motion capture, `dataset.npz`) | | `data/hw/<session>/` once recorded (ignored by git); none exist yet. |
