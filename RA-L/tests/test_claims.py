"""scripts/claims.py: every claim points at a line of paper/main.tex, and the comparison rule is what CLAIMS.md says.
(When the paper is edited and a snippet no longer matches, update the claim in scripts/claims.py.)"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def claims():
    spec = importlib.util.spec_from_file_location("claims", os.path.join(ROOT, "scripts", "claims.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CLAIMS.clear()
    mod.define()
    return mod


def test_every_claim_is_found_in_the_paper(claims):
    missing = [c["where"] for c in claims.CLAIMS if claims.find_line(c["tex"], c["occurrence"]) is None]
    assert not missing


def test_printed_value_is_on_its_line(claims):
    for c in claims.CLAIMS:
        if c["printed"] in ("figure", "not identifiable", "hundreds", "X", "X / Y", "0", "about the mesh error"):
            continue
        line = claims.find_line(c["tex"], c["occurrence"])
        text = " ".join(claims.LINES[line - 1:line - 1 + c["tex"].count("\n") + 1])
        assert c["printed"] in text, (c["where"], c["printed"], text)


def test_rounding_rule(claims):
    def status(printed, value, **kw):
        c = dict(where="t", tex="\\begin{document}", printed=printed, value=lambda: value, step=None, source="",
                 tol=None, rel=None, status=None, note=None, occurrence=0, text=None)
        c.update(kw)
        return claims.evaluate(c, {})["status"]
    assert status("0.21", 0.2074) == "reproduced"
    assert status("0.21", 0.2151) == "differs"
    assert status("1.1--1.2", (1.079, 1.198)) == "reproduced"
    assert status("0.3--0.7", (3.39, 4.17)) == "differs"
    assert status("80", 37.0, rel=0.25) == "differs"
    assert status("25", 22.6, rel=0.3) == "reproduced"
