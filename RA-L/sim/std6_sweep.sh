#!/usr/bin/env bash
# One std6 FEM sweep (the paper's ground truth), as the old cloud/session/sweeps.sh + start.sh ran it with
# TDCR_MODEL=std6, in N_WORKERS parallel SOFA processes sharing one queue of DOE rows (resumable).
#
#   sim/std6_sweep.sh SWEEP DOE_TAG [GRAVITY]
#     SWEEP    1s | 2s | 4s
#     DOE_TAG  train | test | rand          (commands: results/configs_std6_<SWEEP>_<DOE_TAG>.csv, tdcr/doe.py)
#     GRAVITY  g0 (gravity-free) | up (default) | hang | x | y | mx | my     (lab frame, z along the backbone)
#   e.g.  N_WORKERS=6 sim/std6_sweep.sh 2s train g0      -> results/sweeps/std6_2s_sym_train_g0/pose_*.npz
#   then  .venv/bin/python scripts/extract_backbones.py --sections 2 --dir results/sweeps/std6_2s_sym_train_g0 \
#             --tag std6_2s_sym_train_g0
#
# Needs SOFA_ROOT and the std6 meshes in data/meshes (data/README.md). Never run two SOFA sweeps on one machine at
# the same time: they stall each other. The robot parameters (E, nu, mass, gravity, module length, tendon radius)
# are the scene defaults from tdcr/params.py; the run switches below are those of the std6 preset.
set -euo pipefail
cd "$(dirname "$0")/.."
SWEEP=${1:?SWEEP: 1s, 2s or 4s}; DOE_TAG=${2:?DOE_TAG: train, test or rand}; GRAV=${3:-up}
source sim/sofa_env.sh
case "$SWEEP" in
    1s) SECTIONS=1; NAME=one_section ;;
    2s) SECTIONS=2; NAME=two_section ;;
    4s) SECTIONS=4; NAME=four_section ;;
    *)  echo "unknown SWEEP '$SWEEP'"; exit 1 ;;
esac
MESH=data/meshes/std6_${NAME}_sym.vtk
CABLES=data/cables/std6_${NAME}_sym.cables.json
DOE=results/configs_std6_${SWEEP}_${DOE_TAG}.csv
POSES=results/sweeps/std6_${SWEEP}_sym_${DOE_TAG}
case "$GRAV" in
    g0)   export TDCR_GRAVITY=0; POSES=${POSES}_g0 ;;
    up)   export TDCR_GRAVITY_DIR=-z ;;
    hang) export TDCR_GRAVITY_DIR=+z; POSES=${POSES}_ghang ;;
    x|y)  export TDCR_GRAVITY_DIR=+$GRAV; POSES=${POSES}_g$GRAV ;;
    mx|my) export TDCR_GRAVITY_DIR=-${GRAV#m}; POSES=${POSES}_g$GRAV ;;
    *)    echo "unknown GRAVITY '$GRAV'"; exit 1 ;;
esac
# Final pipeline (2026-10-01): mirror-symmetric mesh + polar corotational FEM, implicit Euler with dt = 1.0 and early
# stopping, disc contact on the narrow-gap surfaces with a 40-step ramp; a pose is first settled WITHOUT contact
# (10-step ramp) and redone with contact only if a notch gap ends below 0.08 mm, or straight away when a section bends
# more than 50 deg. Settling is checked over an EVEN number of steps (contact states can alternate every step).
# The robot is tensioned by hand to taut-and-straight: zero pretension. Contact files without the glued joint faces.
export TDCR_FEM_METHOD=polar TDCR_CONTACT=${TDCR_CONTACT:-1} TDCR_SERIAL_FEM=1 TDCR_GAP_MODE=rotated
SWEEP_ARGS="--dt 1.0 --tol 1e-3 --check-steps 6 --ramp-steps 40 --ramp-steps-nocontact 10 --contact-auto 0.08 --contact-above 50"
SETTLE=${SETTLE:-400}
ROWS=$(($(wc -l < "$DOE") - 1))
N_WORKERS=${N_WORKERS:-1}
mkdir -p "$POSES/.claims" "$POSES/logs"
# Shared queue: each worker walks the rows in its own random order and claims a row with an atomic mkdir.
for ((j = 0; j < N_WORKERS; j++)); do
    (for r in $(seq 0 $((ROWS - 1)) | shuf); do
        f=$(printf '%s/pose_%04d.npz' "$POSES" "$r"); c=$(printf '%s/.claims/row_%04d' "$POSES" "$r")
        [ -f "$f" ] && continue; mkdir "$c" 2>/dev/null || continue
        .venv/bin/python sim/shape_sweep.py --sections $SECTIONS --doe "$DOE" --rows $r:$((r + 1)) \
            --settle-steps $SETTLE --pretension 0 $SWEEP_ARGS --mesh "$MESH" --cables "$CABLES" --pose-dir "$POSES"
    done) > "$POSES/logs/worker_$j.log" 2>&1 &
done
wait
echo "[std6_sweep] $SWEEP $DOE_TAG $GRAV: $(ls "$POSES"/pose_*.npz 2>/dev/null | wc -l) of $ROWS poses in $POSES"
