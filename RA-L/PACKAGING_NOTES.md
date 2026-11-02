# Packaging notes (W1)

How `tdcr-twin` was made from the research repository, for the verification chats (W2 reproducibility, W3 physics,
W4 code review) and the owner.

- Source: `/home/pawar/Documents/SOFArelated/TDCR_updated`, branch `twin-impl`, commit `d363e90` (the working tree
  had no uncommitted changes to tracked files; untracked files were not used). The old repository was only read.
- First commit of this repository = the kept files **verbatim** at their new paths; every later commit is a change on
  top of that, so `git diff <first commit>..` shows everything the packaging changed.
- Rule: behaviour preserved. No physics or bug was fixed; what looked wrong is listed under "Issues seen".

## 1. Old path -> new path

### Library (`tdcr/`, installable: `pip install -e .`)

| Old | New | Notes |
|---|---|---|
| `tdcr/eb_prior.py` | `tdcr/eb_prior.py` | constants read from `tdcr.params`; CAD and cache paths from `tdcr.paths` |
| `tdcr/twin.py`, `tdcr/calibrate.py` | same | import `tdcr.compliance` instead of `analysis.module_compliance` |
| `analysis/module_compliance.py` | `tdcr/compliance.py` (constants, `lab_to_scene`, `gravity_of`, `load_set`, `ModuleModel`, `fit`, `TESTS`, `TESTS_STD6`) + `scripts/module_compliance.py` (`report`, CLI) | the library no longer imports `analysis/` |
| `analysis/model_study.py` (`load`, `mirror_perm`, `mirror`, `to_intrinsic`, `from_intrinsic`, `evaluate`, `gp_fit_predict`, `modes_fit_predict`) | `tdcr/backbones.py` | the rest (priors, variants, CLI) left out |
| `analysis/n_transfer.py` `ModesGP` | `tdcr/backbones.py` | imported by a test |
| `analysis/shape_analysis.py` (`stations_for`, `node_sections`, `kabsch`, `backbone`) | `tdcr/backbones.py` | module length from `tdcr.params` (was `cc_kinematics.BACKBONE_L`, env default 220) |
| `tools/make_cmd_doe.py` | `tdcr/doe.py` (`python -m tdcr.doe`) | |
| `analysis/hw_eval.py` | `tdcr/hw_eval.py` (`python -m tdcr.hw_eval`) | |
| `analysis/cosserat_baseline.py` | `tdcr/baselines/cosserat_rod.py` | |
| `analysis/pinn_baseline.py` | `tdcr/baselines/pinn.py` | needs the `pinn` extra (torch) |
| `hardware/{pose_lists,mocap,run_poses,homing,hysteresis_probe}.py` | `tdcr/hardware/` | `python -m tdcr.hardware.<module>` |
| `examples/live_twin.py` | `tdcr/hardware/live_twin.py` | |
| `tdcr/cc_kinematics.py`, `multisection.py`, `tendon_mapping.py`, `cosserat.py`, `make_cables_json.py`, `paths.py`, `__init__.py` | same | `cosserat.py`: the `--sweep` CLI (needed `tdcr/digitaltuning.py`) removed, `--self-test` kept; `paths.py` rewritten (new locations, `sofa_root()`) |
| (new) | `tdcr/params.py` | the one robot-parameter source (section 2) |

### Paper pipeline (`scripts/`)

| Old | New |
|---|---|
| `analysis/module_compliance.py` (CLI) | `scripts/module_compliance.py` |
| `analysis/n_transfer.py` | `scripts/n_transfer.py` |
| `paper_ral/figures/make_figs.py` | `scripts/fig_transfer.py` (Fig. 2; `--mc`, `--nt`, `--out-dir`, default `paper/figures/`) |
| `docs/eb_prior/make_figs.py` (second figure, `profile.pdf`, copied to the paper by hand) | `scripts/fig_profile.py` (Fig. 1, writes `paper/figures/fig_profile.pdf` directly; the first figure, `sections.pdf`, belongs to the derivation notes and was left out) |
| `docs/study_guide/make_figs.py` | `scripts/study_guide_figs.py` (writes `results/study_guide/`, not tracked) |
| `analysis/extract_backbones.py` | `scripts/extract_backbones.py` |
| `results/pinn/run_stage1.sh`, `run_stage2.sh` | `scripts/pinn_stage1.sh`, `pinn_stage2.sh` |

### SOFA (`sim/`) and data

| Old | New |
|---|---|
| `sim/{shape_sweep,multi_section_scene,two_section_scene,tendon,mass,contact_gap,cylinder_scene}.py` | same |
| `cloud/session/sweeps.sh` (std6 preset) + `start.sh` + `std6_queue.sh` (job list) | `sim/std6_sweep.sh` (one sweep per call) + `sim/sofa_env.sh` |
| `paper_ral/main.tex`, `refs.bib`, `figures/fig_profile.pdf`, `figures/fig_transfer.pdf` | `paper/` |
| `cad/std_tdcr_L150.json`, `cad/std6_tdcr_L150.json`, `cad/std6_tdcr_L150_D40.json` | `data/cad/` |
| `data/cables/std6_*_sym.cables.json`, `data/contacts/std6_*_sym.contact.npz` | same |
| `hardware/poses/`, `hardware/measurements.json` | `data/hardware/` (servo zeros go to `data/hardware/zeros/`) |
| `results/backbone_std6_*.npz` (17), `mc_std6.json`, `mc_std6_reeval.json`, `n_transfer.json`, `mc_ablation.json`, `mc_grav.json`, `pinn/*.json`, `configs_std6_*.csv` + `.meta.json` | `results/` (the stored files, unchanged) |
| `results/backbone_std_{1s_sym,2s_sym,1s_sym_g0,2s_sym_g0,1s_sym_test,2s_sym_test,1s_sym_test_g0,2s_sym_test_g0,4s_sym_stabletest_g0}.npz` | `results/` (older robot: `mc_grav.json`, which `tdcr.doe` and the pose lists screen with, and the `mc_ablation` footnote) |
| `tests/test_*.py` (19 of 21) | `tests/` |
| `tests/conftest.py` | rewritten: `chdir` to the root only; `sys.path` comes from `pythonpath = ["."]` in `pyproject.toml` |

### Left out

Legacy models and their scripts (`tdcr/digitaltuning.py`, `explicit_model.py`, `mechanics_prior.py`,
`check_cables.py`; most of `analysis/`: `benchmark_twin`, `calib_eval`, `compare_*`, `eb_eval`, `eb_report`,
`explicit_shape`, `final_eval`, `fit_*_prior`, `fresh_test_*`, `gap_check`, `learning_curve`, `make_figures`,
`make_overview`, `modular_transfer`, `module_compliance_figs`, `night_checks`, `plot_*`, `prior_figs`,
`stability_classifier`, `std6_sanity`, `test_eval`, `twin_reach`), `sim/mor/` and its test
(`test_mor_snapshots.py`), the other `sim/` tools (`converge_trace`, `diverge_trace`, `factor_check`,
`mesh_isotropy_check`, `merge_poses`, `parallel_sweep`, `profile_step`, `run_*.sh`), `tools/` (mesh, CAD, cable,
DOE generators other than `make_cmd_doe`; cloud sync), `render/`, `imgui/`, `poster/`, `paper/` (old paper),
`paper_ral` build files and the untracked Overleaf copies, `docs/`, `cloud/`, `logs/`, `requirements.txt` (now
`pyproject.toml`), meshes (`data/meshes`, 115 MB; the std6 ones are 28 MB), raw sweeps, CAD STEP/STL and print
files, older cable and contact files, `results/` files the paper does not use, and `ral-paper.zip`. Where the big
inputs live: `data/README.md`.

## 2. One parameter source: the choices

All std6 values now come from `tdcr/params.py` (`STD6`), plus `MASS_STD = 0.0346` for the older module. The stale
environment-variable defaults are gone from the library: `tdcr/cc_kinematics.py` no longer reads `TDCR_TENDON_R`
(default 9.0) or `TDCR_SECTION_L` (default 220). Where values conflicted, the one kept is the one the paper's numbers
were produced with. **W3: please confirm each line.**

| Quantity | Kept | Conflicting copies (old repo) | Why this value |
|---|---|---|---|
| Module length | 160 mm | 220: `cc_kinematics` env default (also via `multisection.SECTION_LENGTHS`, `shape_analysis.L`, `multi_section_scene` when no length is given) | Every std6 run exported `TDCR_SECTION_L=160` (`sweeps.sh` std6 preset); the backbone files' stations end at 160 n; `twin`, `module_compliance`, `hw_eval`, the baselines all used 160. |
| Tendon (channel) radius | 9.5 mm | 9.0: `cc_kinematics` env default | std6 runs exported `TDCR_TENDON_R=9.5`; CAD channel radius; `hw_eval`, `cosserat_baseline`, `pose_lists`, `pinn_baseline`, `make_cmd_doe --r-d`, `eb_prior.section_profile(hole_r)` all 9.5. |
| Tendon bearing radius (moment arm in the EB prior) | 10.55 mm | (a different quantity, not a conflict) | `eb_prior.R_BEAR`; the std6 cable files put every cable point at radius 10.55 (`bearing_radius_mm`, channel wall). Kept as a separate parameter `bearing_radius_mm`. **W3: confirm 10.55 rather than 9.5 is the right moment arm.** |
| E | 56000 kPa | 50000: `cosserat.py` default, `two_section_scene` default, `cylinder_scene` | std6 SOFA ran with `TDCR_YOUNG=56000` (`sweeps.sh`); `E_FEM`, `predict_gravity`, `cosserat_baseline`. |
| nu | 0.48 | 0.45: `cosserat.py`, `two_section_scene` default, `cylinder_scene` | `TDCR_POISSON=0.48` (`sweeps.sh`); `timoshenko_factor`, `cosserat_baseline`. |
| Outer radius | 12.5 mm (tube, ID 14) | 11.5: `cosserat.py` (solid rod), `make_cad_vtk`, `make_cylinder_vtk` | CAD OD 25; `eb_prior`, `cosserat_baseline`. The 11.5 copies belong to the old cylinder robot. |
| Module mass | 0.0337 kg | 0.0346: `MASS_STD`, `predict_gravity` default; 0.063: `sweeps.sh` default | `sweeps.sh` std6 preset exports `TDCR_SECTION_MASS=0.0337` after the 0.063 default; `load_set` gives std6 sets 0.0337; the twin's default. **0.0346 is kept as `predict_gravity`'s default mass** (behaviour): see issue 1. **W3: confirm from the sweep `.meta.json`/pose files that the std6 FEM ran with 0.0337, not 0.063.** |
| Gravity | 9810 mm/s^2 | 9.81: `two_section_scene` default, `cylinder_scene` | Units mm, kg, s; every gravity sweep exported 9810. |

What changed in practice:

- `tdcr/cosserat.py` defaults: E 50000 -> 56000, nu 0.45 -> 0.48, solid rod radius 11.5 -> 12.5. The paper's Cosserat
  baseline passes E, nu, EA, EI explicitly, so its numbers do not change; only calls that relied on the defaults
  (the module's self-test) see the new values.
- `sim/two_section_scene.py` defaults: E, nu, mass per section 0.015 -> 0.0337, gravity 9.81 -> 9810. The std6 sweeps
  set all four through the environment, so the std6 ground truth is unchanged; the `TDCR_*` overrides still work
  (`TDCR_GRAVITY=0` for the gravity-free sets). `TDCR_TENDON_R` / `TDCR_SECTION_L` are no longer read anywhere.
- `sim/cylinder_scene.py` keeps its own 50000 / 0.45 / -9.81 (the old single-section cylinder; not on the std6 path,
  imported by `shape_sweep.py` only). Candidate for removal (W4).
- Numbers that are not robot parameters stayed where they were (relaxation, tolerances, knot spacing, priors' ridge,
  hardware limits).

## 3. Reproduction (old code vs packaged code)

Both run with the same venv (Python 3.12.3, the old repo's pinned versions: numpy 2.5.2, scipy 1.18.1,
scikit-learn 1.9.0, numba 0.68.0, threadpoolctl 3.6.0), the old code from a scratch copy of `d363e90`.

| Item | Packaged vs old code | Packaged vs stored file in the old repo |
|---|---|---|
| `results/mc_std6.json` (Table I), `scripts/module_compliance.py`, `OMP_NUM_THREADS=1`, 78 s | **bit-identical** JSON; identical log | identical up to 4e-10 in theta and 1e-10 mm in the errors (BLAS thread count; the stored run used the default threads) |
| `results/n_transfer.json` (Table I learned column, Fig. 2 points), `OMP_NUM_THREADS=8`, ~2 h | **identical** JSON (apart from the `mc` path string); identical log | `L0+L1` identical (3e-11 mm); the Fig. 2 series `+L3 local (output)` within 0.0004 mm; **differs in the "inside" coupling on the 2-module horizontal sets**: `std6_2s_sym_test_gx` `+L3 local (inside)` tip 5.642 vs 5.417 mm (95 % 19.6 vs 15.9, max 25.2 vs 21.8), `local+N (inside)` 5.386 vs 5.374; `std6_2s_sym_test_gy` `local (inside)` 6.878 vs 6.653. Everything else within 0.005 mm. The old code at `d363e90` gives the same new values, so the stored file is stale (issue 2), not the packaging. The paper quotes none of the changed entries (it quotes the inside variants on four modules and hanging, unchanged). |
| Fig. 2 (`scripts/fig_transfer.py`) | pixel-identical (150 dpi) from the stored JSONs and from the fresh ones | pixel-identical to `paper/figures/fig_transfer.pdf` |
| Fig. 1 (`scripts/fig_profile.py`) | pixel-identical | pixel-identical to `paper/figures/fig_profile.pdf` |
| Study-guide numbers (`scripts/study_guide_figs.py`): G, Prop. 1 (tip moves 4.53 mm), fixed point, pretension, Prop. 2 (singular values 297.3 ... 0.1536) | identical | identical to `docs/study_guide/figs/numbers.json` |
| Run times (same script, idle laptop) | median 0.152 / 0.291 / 0.591 ms without gravity, 0.876 / 1.678 / 3.618 ms hanging (1 / 2 / 4 modules), p95 4 modules hanging 15.3 ms; old code on the same run 0.150 / 0.287 / 0.594, 0.881 / 1.674 / 3.647, 15.4 | stored 0.151 / 0.291 / 0.595, 0.872 / 1.729 / 3.667, 15.8 (paper: 0.87 / 1.73 / 3.7 ms, p95 16 ms). Timing noise only. |
| Section-profile caches `eb_profile_std_<digest>.npz` | same digests, bit-identical arrays | |

Not re-run: the PINN baseline (Table II, hours on CPU; its 4 unit tests pass), the SOFA ground truth (not allowed
here), the hardware sections (pending in the paper). Every number that did not reproduce is the stale
`n_transfer.json` entries above.

## 4. Issues seen (not fixed)

Ranked roughly by how much they could matter. File references are to this repository.

1. **Several std6 FEM test sets are partial sweeps** (poses in the backbone file / rows in the command design):
   `std6_2s_sym_rand_g0` 2/100, `std6_2s_sym_test_gy` 22/100, `std6_2s_sym_test` 69/100, `std6_2s_sym_test_gx`
   92/100, `std6_2s_sym_train` 175/200 (all other std6 sets complete; no 4-module set with gravity exists). Table I
   and Fig. 2 report "2 horiz. y" on 22 poses and Table I's `2s rand` on 2. If the missing rows are poses that
   failed or fell rather than poses never run, the sets are biased towards easy poses. W3/W5: check the sweep
   directories on the lab PC (`failed_*.txt`, `skipped_*.txt`) and the sample sizes quoted in the paper. Reproduce:
   compare `len(np.load("results/backbone_<tag>.npz")["C"])` with the rows of `results/configs_<...>.csv`.
2. **`results/n_transfer.json` predates the gravity fixed-point convergence fix** (stored 21 Oct, commit 54abed7; the
   fix is f464bd2 / ff9fa02 of 29 Oct). `mc_std6.json` was regenerated after the fix (0a471f2), `n_transfer.json` was
   not. A fresh run (old or packaged code, identical) changes only the "inside" residual variants on the 2-module
   horizontal sets, by up to 0.23 mm mean tip (section 3); the stored file is kept unchanged here so the repository
   matches the paper. W2: regenerate it with the rest.
3. **Command designs screened with the old robot's physics.** `tdcr/doe.py:screen` (old `tools/make_cmd_doe.py`) and
   `tdcr/hardware/pose_lists.py:screen_model` build the screening model from `results/mc_grav.json` (identified on the
   older 3-channel `std_*` sets), and the gravity check calls `predict_gravity` with its default mass 0.0346 kg
   (`tdcr/doe.py:85`), not std6's 0.0337. This only decides which commands enter `configs_std6_*.csv` (no reported
   error depends on it directly), but the screen's bend / load / buckling limits are those of another identification.
4. **Fig. 1 is the old 3-channel profile.** `scripts/fig_profile.py` (old `docs/eb_prior/make_figs.py`) plots
   `section_profile()` with `holes=None` on `std_tdcr_L150.json`; the std6 model cuts six channel holes
   (`holes=HOLES6`). Known to W5 (caption vs code).
5. **std6 uses the std CAD file.** `tdcr/eb_prior.py:CAD_JSON` is `data/cad/std_tdcr_L150.json` for every std6
   result; `std6_tdcr_L150.json` has the same geometry (it differs in section count, tendon list, volume and mass
   metadata), and the holes are cut via `holes=`. Harmless today, but the 6-channel CAD file is not the one used.
6. **Moment arm 10.55 vs channel radius 9.5** (section 2): the EB prior and the SOFA cable points use 10.55 mm (the
   channel wall), the Cosserat baseline, the constant-curvature model in `hw_eval` and the lab mapping use 9.5 mm.
   W3: which is physical for the moment arm, and does the FEM cable sit at 10.55?
7. **Cached prior objects are shared and mutated.** `ModuleModel.prior` (`tdcr/compliance.py:117-125`) caches one
   `EBPrior` per geometry and `build` overwrites its compliances, tendon stretch and gap in place: two `build` calls on
   the same `ModuleModel` with different `hw` or `gap` return the same object, the first silently changed. `Twin`
   makes a fresh model per instance, so the twin is safe; scripts that build twice and keep both are not. The cache
   key also ignores `cad_json`. (W4)
8. **`predict_gravity` with `iters=0`** raises `NameError` (`it` undefined, `tdcr/eb_prior.py:469`). Never called that
   way today.
9. **Shadowed name in `ModuleModel.build`**: the field array `f` is rebound to the file handle of `cad_json`
   (`tdcr/compliance.py:145`); harmless because `f` is not used afterwards. (W4)
10. **Import side effect removed**: the old `analysis/module_compliance.py` called `warnings.simplefilter("ignore")` at
    import, so any program that imported the twin silenced every warning. `tdcr/compliance.py` does not (the scripts
    and `tdcr/doe.py` still do). Only warning output changes.
11. **Not on the std6 path but still in `sim/`**: `cylinder_scene.py` (old cylinder robot; 50000 kPa, 0.45, gravity
    -9.81 mm/s^2, i.e. 1000x too weak) is imported by `shape_sweep.py` at module level; the legacy two-body scene
    (`TDCR_LEGACY_2S`) and many `TDCR_*` experiment switches remain in `two_section_scene.py`. Kept so the scenes stay
    as they ran; W4 can prune once someone can run SOFA.
12. **`tdcr/hw_eval.py` reads predictions of `sim/mor/run_reduced.py` and `sim/calibrate_sofa.py`**, which were not
    ported (MOR is not in the paper); those models are skipped with a note when their files are missing, as before.
13. **The release named in the work plan does not exist**: `sim-data-2026-10-11` is not a release of
    `om-mpawar/virtual-sensor` (only `sim-data-2026-10-03` and `-10-04`, older robot). The raw std6 sweeps seem to be
    only on the lab PC (`data/README.md`). Worth backing up before anyone needs to re-extract.
14. **`tdcr/compliance.py:fit` forks `min(20, n_params)` processes** regardless of the machine; fine on the 22-core
    laptop.

## 5. Tests

`.venv/bin/python -m pytest -q`: **94 passed** (with the `pinn` extra installed and the std6 one-module mesh copied
to `data/meshes/`, both done in this checkout's `.venv`; without them 88 pass and 3 skip). The old suite at
`d363e90`, run the same way, has 97 tests: these 94 plus the 3 of `test_mor_snapshots.py` (MOR not ported).
Test bodies are unchanged except their imports, the CAD paths (`data/cad/`), and the mesh-skip in
`test_scene_mass.py`.
