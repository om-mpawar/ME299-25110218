"""
Run time (paper Sec. VI-D): the twin's solve time per pose and the Cosserat baseline's.

Twin: the study guide's protocol (scripts/study_guide_figs.py, fig_time): 21 FEM test commands, 10 interpolation
steps between consecutive ones (200 poses), warm-started along the sequence, 1/2/4 modules, gravity-free and hanging;
repeated `--reps` times with a fresh twin. The paper's medians and 95th percentile come from the study guide's single
pass (results/study_guide/numbers.json); its "slowest call 25 ms" is the largest single call over three such passes
(old repository session of 28 Oct 2026, which printed it and stored nothing), recorded here as max_ms.

Cosserat: tdcr.baselines.cosserat_rod.predict (uniform rod, pull-only tendons, no gravity) on the first 20 commands of
the 2-module gravity-free test set; mean ms per pose (paper: about 280 ms; old repository 3d8b56b printed mean 277.7,
median 377.2 ms).

Timings depend on the machine and its load: run on an otherwise idle computer. The paper's numbers are from an
Intel Core Ultra 9 185H laptop.

    .venv/bin/python scripts/runtime.py                  # ~1 min -> results/runtime.json
"""
import argparse
import json
import os
import platform
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.compliance import load_set
from tdcr.paths import RESULTS, ROOT
from tdcr.twin import Twin


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def twin_times(reps):
    rows = []
    for n, tag in ((1, "std6_1s_sym_test_g0"), (2, "std6_2s_sym_test_g0"), (4, "std6_4s_sym_test_g0")):
        dl = load_set(tag)["dl"][:21]
        seq = np.concatenate([np.linspace(dl[i], dl[i + 1], 10) for i in range(20)])
        for mount in ((0.0, 0.0, 0.0), "hanging"):
            per = []
            for _ in range(reps):
                tw = Twin(n, mount)
                tw.shape(seq[0])
                t = [tw.shape(x).solve_ms for x in seq]
                per.append(dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), max_ms=float(max(t))))
            row = dict(n=n, mounting="no gravity" if mount != "hanging" else "hanging", reps=per,
                       median_ms=float(np.mean([r["median_ms"] for r in per])),
                       p95_ms=float(np.mean([r["p95_ms"] for r in per])), max_ms=float(max(r["max_ms"] for r in per)))
            rows.append(row)
            print(f"twin {n} modules {row['mounting']:10s}: median {row['median_ms']:.2f} ms, p95 {row['p95_ms']:.1f} ms, "
                  f"max {row['max_ms']:.1f} ms ({reps} passes)", flush=True)
    return rows


def cosserat_times(poses):
    from tdcr.baselines.cosserat_rod import predict
    dl = load_set("std6_2s_sym_test_g0")["dl"][:poses]
    _, _, ms = predict(dl, (0.0, 0.0, 0.0))
    print(f"Cosserat 2 modules: mean {ms.mean():.1f} ms, median {np.median(ms):.1f} ms per pose ({poses} poses)")
    return dict(poses=poses, mean_ms=float(ms.mean()), median_ms=float(np.median(ms)), max_ms=float(ms.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cosserat-poses", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(RESULTS, "runtime.json"))
    a = ap.parse_args()
    import numba
    res = dict(machine=dict(cpu=cpu_name(), cores=os.cpu_count(), platform=platform.platform(),
                            python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                            OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")),
               twin=twin_times(a.reps), cosserat_2s=cosserat_times(a.cosserat_poses))
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
