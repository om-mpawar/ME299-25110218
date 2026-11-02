"""Regression tests for the real-robot (std6) data loader and mirror augmentation (fixed 19 Oct 2026): before the
fix, std6 sets got the shared routing, no channel holes and upright gravity for every mounting (257 mm instead of
0.9 mm on four modules), and the mirror swapped the wrong tendons in sections 3-4."""
import json

from tdcr.backbones import evaluate, mirror_perm
from tdcr.compliance import ModuleModel, gravity_of, load_set


def test_std6_4s_routing_and_accuracy():
    d = load_set("std6_4s_sym_test_g0")
    assert d["routing"] == "paired60" and d["holes"] is not None and d["g"] is None
    sub = {k: (v[:8] if k in ("C", "R", "dl", "theta", "alpha") else v) for k, v in d.items()}
    m, th = ModuleModel.from_result(json.load(open("results/mc_std6.json")))
    C, R, _ = m.predict(th, sub)
    assert evaluate(C, R, sub["C"], sub["R"])["tip"].mean() < 1.5      # was 257 mm with the old loader


def test_gravity_direction_from_tag():
    assert gravity_of("std6_2s_sym_test_ghang") == (0.0, 9810.0, -0.0)
    assert gravity_of("std6_2s_sym_test_gy") == (0.0, 0.0, -9810.0)
    assert gravity_of("std6_2s_sym_test_g0") is None


def test_mirror_perm_paired60():
    az = load_set("std6_4s_sym_test_g0")["az"]
    assert mirror_perm(4, az).tolist() == [0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9]
