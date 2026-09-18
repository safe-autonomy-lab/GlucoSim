"""Suite coverage, unit delivery, and conservative handling of failed candidates."""
from argparse import Namespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from examples import run_scenarios as runner


def test_combined_action_delivers_meal_grams_and_bolus_units_once():
    action = runner.get_meal_bolus_scenario(60., 4.)
    delivered = np.sum([np.asarray(action(t)) for t in range(120)], axis=0)
    np.testing.assert_allclose(delivered, [60., 4., 0.])
    assert float(action(29.)[1]) == 0.
    assert float(action(30.)[1]) == 4.
    assert float(action(31.)[1]) == 0.


def test_metrics_weight_time_and_reject_truncated_or_nonfinite_runs():
    params = Namespace(Vg=2.)
    times = np.array([0., 1., 3.])
    states = np.zeros((3, 18))
    states[:, 3] = 2. * np.array([100., 200., 200.])
    result = runner.response_metrics(times, states, params, 3.)
    assert result['eligible']
    assert result['hyper_auc_mgdl_min'] == 50.
    assert result['tir_70_180_pct'] == pytest.approx(100. / 6.)
    assert not runner.response_metrics(times, states, params, 4.)['eligible']
    states[-1, 3] = 2. * 69.
    assert not runner.response_metrics(times, states, params, 3.)['eligible']
    states[-1, 5] = np.nan
    assert not runner.response_metrics(times, states, params, 3.)['eligible']
    assert not runner.response_metrics(times[:1], states[:1], params, 3.)['eligible']


def test_selection_rejects_unsafe_shortcuts_and_breaks_ties_by_dose():
    rows = [dict(eligible=False, hyper_auc_mgdl_min=0., bolus_u=10.),
            dict(eligible=True, hyper_auc_mgdl_min=20., bolus_u=2.),
            dict(eligible=True, hyper_auc_mgdl_min=20., bolus_u=4.)]
    assert runner.select_bolus(rows) is rows[1]
    assert runner.select_bolus(rows[:1]) is None


def test_suite_covers_cross_product_and_preserves_prior_outputs(tmp_path, monkeypatch):
    def fake_simulate(horizon, dt, x0, action_fn, params, cfg, key):
        assert not cfg.enable
        dose = float(action_fn(30.)[1])
        glucose = np.array([110., 230. - 20. * dose, 120.])
        states = np.tile(np.asarray(x0), (3, 1))
        states[:, 3] = glucose * params.Vg
        return np.array([0., 30., horizon]), states
    monkeypatch.setattr(runner, 'simulate', fake_simulate)
    args = Namespace(meals=[30., 60.], boluses=[2., 4., 100.], hours=6.,
                     output_dir=str(tmp_path), name=None, seed=42, gif=False)
    previous = tmp_path / 'existing.csv'
    previous.write_text('preserve me')
    output = runner.run_suite(args)
    metrics = pd.read_csv(output + '/candidate_metrics.csv')
    recommendations = pd.read_csv(output + '/recommendations.csv')
    assert len(metrics) == 3 * (1 + 2 * 3)  # basal + meals x [0,2,4], 100 exceeds max
    assert len(recommendations) == 6
    assert set(metrics.diabetes_type) == {'t1d', 't2d', 't2d_no_pump'}
    assert set(metrics.bolus_u) == {0., 2., 4.}
    assert recommendations.recommended_bolus_u.eq(4.).all()
    assert previous.read_text() == 'preserve me'
    assert pd.read_csv(output + '/glucose_traces.csv').shape[0] == 3 * len(metrics)


@pytest.mark.parametrize('meals,doses,hours', [([0.], [0.], 6.), ([30.], [-1.], 6.),
                                                ([30.], [0.], .5)])
def test_invalid_suite_grid_rejected_before_writing(tmp_path, meals, doses, hours):
    args = Namespace(meals=meals, boluses=doses, hours=hours, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        runner.run_suite(args)
    assert not list(tmp_path.iterdir())
