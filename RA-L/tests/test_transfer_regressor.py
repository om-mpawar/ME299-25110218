"""A reused module regressor must match the previous fit-and-predict path."""
import numpy as np

from analysis.model_study import modes_fit_predict
from analysis.n_transfer import ModesGP


def test_reused_modes_gp_matches_previous_predictions():
    rng = np.random.default_rng(42)
    F = rng.normal(size=(24, 3))
    Y = np.sin(F @ rng.normal(size=(3, 12)))
    query = rng.normal(size=(7, 3))
    old = modes_fit_predict(Y, F, query, 4, restarts=0, per_output=False)
    model = ModesGP(Y, F, 4, restarts=0)
    reused = np.concatenate([model.predict(query[:3]), model.predict(query[3:])])
    np.testing.assert_allclose(reused, old, atol=1e-10)
