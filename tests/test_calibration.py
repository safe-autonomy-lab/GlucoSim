"""Keep the legacy factory calibration stages and public imports stable."""
import dataclasses
from typing import get_type_hints

import pytest

from glucosim.simglucose.core import params
from glucosim.simglucose.physiology import calibration


@pytest.mark.parametrize('name', [
    'autobalance_basal_t1d', 'autobalance_basal_t2d', '_steady_state_insulin_from_Sb',
])
def test_legacy_calibration_imports(monkeypatch, name):
    sentinel = object()
    received = []

    def delegated(*args, **kwargs):
        received.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(calibration, name, delegated)
    helper = getattr(params, name)
    assert params.PatientParams in get_type_hints(helper).values()
    if name == 'autobalance_basal_t1d':
        assert helper(sentinel, basal_scale=0.8, hepatic_scale=1.2) is sentinel
        assert received == [((sentinel, 0.8, 1.2), {})]
    else:
        assert helper(sentinel) is sentinel
        assert received == [((sentinel,), {})]


@pytest.mark.parametrize('kind,expected', [
    ('t1d', ['t1d']),
    ('t2d', ['pump_rate', 'glucose', 'insulin', 'insulin']),
    ('t2d_no_pump', ['pump_rate', 'glucose', 'insulin', 'insulin', 'insulin', 'glucose']),
])
def test_factory_keeps_calibration_order(monkeypatch, kind, expected):
    calls = []
    for name, label in [
        ('autobalance_basal_t1d', 't1d'),
        ('autobalance_basal_t2d', 'glucose'),
        ('_steady_state_insulin_from_Sb', 'insulin'),
        ('pump_basal_rate_t2d', 'pump_rate'),
    ]:
        original = getattr(calibration, name)

        def tracked(*args, _original=original, _label=label, **kwargs):
            calls.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(calibration, name, tracked)
    params.create_patient_params('adolescent#001', diabetes_type=kind)
    assert calls == expected


@pytest.mark.parametrize('kind', ['pump', 'no_pump'])
def test_each_pass_receives_previous_result(monkeypatch, kind):
    base = params.create_patient_params('adolescent#001', diabetes_type='t1d')
    calls = []

    def balance(p):
        calls.append(('glucose', p.Vm0, p.Ipb, p.Ilb))
        return dataclasses.replace(p, Vm0=p.Vm0 + 1.)

    def insulin(p):
        calls.append(('insulin', p.Vm0, p.Ipb, p.Ilb))
        return p.Ipb + 2., p.Ilb + 3.

    monkeypatch.setattr(calibration, 'autobalance_basal_t2d', balance)
    monkeypatch.setattr(calibration, '_steady_state_insulin_from_Sb', insulin)
    result = getattr(calibration, 'calibrate_t2d_' + kind)(base)
    if kind == 'pump':
        assert calls == [('glucose', base.Vm0, base.Ipb, base.Ilb),
                         ('insulin', base.Vm0 + 1., base.Ipb, base.Ilb),
                         ('insulin', base.Vm0 + 1., base.Ipb + 2., base.Ilb + 3.)]
        assert result.Ipb == base.Ipb + 4.
        assert result.Ilb == base.Ilb + 6.
    else:
        assert calls == [('insulin', base.Vm0, base.Ipb, base.Ilb),
                         ('glucose', base.Vm0, base.Ipb + 2., base.Ilb + 3.)]
        assert result.Ipb == base.Ipb + 2.
        assert result.Ilb == base.Ilb + 3.
    assert result.Vm0 == base.Vm0 + 1.
