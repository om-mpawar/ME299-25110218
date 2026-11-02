"""Design changes must reach the model and invalidate geometry caches."""
import json

import numpy as np
import pytest

from tdcr import eb_prior
from tdcr.twin import Twin


def test_cache_tracks_cad_contents_and_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(eb_prior, "CACHE", str(tmp_path / "profile.npz"))
    g = json.load(open(eb_prior.CAD_JSON))
    path = tmp_path / "design.json"
    path.write_text(json.dumps(g))
    kw = dict(cad_json=path, dy=0.5, dc=0.5)
    a = eb_prior.section_profile(**kw)
    g["inner_dia"] = 18.0
    path.write_text(json.dumps(g))
    b = eb_prior.section_profile(**kw)
    assert not np.allclose(a[1], b[1])
    assert np.mean(b[1]) < np.mean(a[1])
    cached = eb_prior.section_profile(**kw)
    np.testing.assert_array_equal(cached[1], b[1])
    c = eb_prior.section_profile(**{**kw, "dy": 0.25})
    assert len(c[0]) == 2 * len(b[0])
    assert len(list(tmp_path.glob("profile_*.npz"))) == 3


def test_explicit_design_changes_twin_compliance():
    old = Twin(1, (0, 0, 0))
    new = Twin(1, (0, 0, 0), cad_json="data/cad/std6_tdcr_L150_D40.json")
    assert not np.allclose(old.eb.cX, new.eb.cX)
    assert not np.allclose(old.eb.cA, new.eb.cA)
    assert np.all(np.isfinite(new.shape([1., 0., 0.]).positions))


def test_mismatched_length_rejected():
    with pytest.raises(ValueError, match="CAD module height"):
        eb_prior.EBPrior(1, L=120.)
