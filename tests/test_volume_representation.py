"""Authoritative per-kg volumes and the intentionally changed representation.

The checked-in historical fixture was produced by the real factory at the last
production checkpoint. Tests need neither Git history nor an old implementation.
"""
import base64
import dataclasses
from decimal import Decimal
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glucosim.simglucose.core import conversion, params, patient_loader


KINDS = ('t1d', 't2d', 't2d_no_pump')
REMOVED = ('V_G', 'V_I', 'V_G_L', 'V_I_L')
PRE_CHANGE = '1130a478de832cee01452942d85dd1b375030b32'


@pytest.fixture(scope='module')
def patient():
    return params.create_patient_params('adolescent#001', diabetes_type='t1d',
                                        BW=100.0, Vg=2.0, Vi=0.05)


@pytest.fixture(scope='module')
def historical_fixture():
    fixture_path = Path(__file__).with_name('fixtures') / 'patient_params_before_volumes.json'
    fixture = json.loads(fixture_path.read_text())
    assert fixture['provenance']['source_revision'] == PRE_CHANGE
    assert fixture['provenance']['factory'] == (
        'glucosim.simglucose.core.params.create_patient_params')
    assert fixture['provenance']['kwargs'] == {
        'diabetes_type': 't1d', 'BW': 100.0, 'Vg': 2.0, 'Vi': 0.05}
    assert fixture['provenance']['source_sha256'] == (
        '82c60a17d8054f477a09c82081883d7cf743d2ccead0e140162f3586bba9d1c7')
    return fixture


@pytest.mark.parametrize('kind', KINDS)
def test_factory_volume_arithmetic_uses_effective_inputs(kind):
    p = params.create_patient_params('adolescent#001', diabetes_type=kind,
                                     BW=100.0, Vg=2.0, Vi=0.05)
    assert (p.BW, p.Vg, p.Vi) == (100.0, 2.0, 0.05)
    # 200 dL = 20 L; 0.05 L/kg * 100 kg = 5 L.
    assert p.V_G_L == 20.0
    assert p.V_I_L == 5.0


@pytest.mark.parametrize('kind', KINDS)
def test_direct_adapter_volume_arithmetic_without_csv_reload(monkeypatch, patient, kind):
    def unexpected_load(*args, **kwargs):
        pytest.fail('A direct adapter reloaded CSV inputs')
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', unexpected_load)
    kwargs = {} if kind == 't1d' else {
        'config': dict(BW_factor=1.0, Ib_factor=1.0, Ipb_factor=1.0, HEb_factor=1.0)}
    adapted = getattr(params, 'adapt_params_for_' + kind)(patient, **kwargs)
    assert (adapted.BW, adapted.Vg, adapted.Vi) == (100.0, 2.0, 0.05)
    assert adapted.V_G_L == 20.0
    assert adapted.V_I_L == 5.0


@pytest.mark.parametrize('replacement,expected', [
    ({'BW': 120.0}, (24.0, 6.0)),
    ({'Vg': 3.0}, (30.0, 5.0)),
    ({'Vi': 0.1}, (20.0, 10.0)),
    ({'BW': 80.0, 'Vg': 3.0, 'Vi': 0.1}, (24.0, 8.0)),
])
def test_authoritative_replacement_updates_totals_immediately(patient, replacement, expected):
    replaced = dataclasses.replace(patient, **replacement)
    assert (replaced.V_G_L, replaced.V_I_L) == expected
    assert (patient.V_G_L, patient.V_I_L) == (20.0, 5.0)


def test_no_duplicate_storage_and_properties_are_read_only(patient):
    fields = {field.name for field in dataclasses.fields(patient)}
    assert fields.isdisjoint(REMOVED)
    assert vars(patient).keys().isdisjoint(REMOVED)
    assert dataclasses.asdict(patient).keys().isdisjoint(REMOVED)
    for name in ('V_G', 'V_I'):
        assert not hasattr(patient, name)
        assert name not in patient.UNITS
        assert name not in patient.DESCRIPTIONS
        assert name not in patient.CATEGORIES
    for name in ('V_G_L', 'V_I_L'):
        descriptor = getattr(type(patient), name)
        assert isinstance(descriptor, property) and descriptor.fset is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(patient, name, 999.0)


@pytest.mark.parametrize('name', REMOVED)
def test_constructor_and_replace_reject_derived_names(patient, name):
    with pytest.raises(TypeError, match=name):
        params.PatientParams(**dict(dataclasses.asdict(patient), **{name: 99.0}))
    with pytest.raises(TypeError, match=name):
        dataclasses.replace(patient, **{name: 99.0})


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('factory', [params.create_patient_params, params.create_env_params])
@pytest.mark.parametrize('name', REMOVED)
def test_factory_rejects_derived_names_before_loading(monkeypatch, factory, kind, name):
    def unexpected_load(*args, **kwargs):
        pytest.fail('Derived-volume rejection must precede loading/calibration')
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', unexpected_load)
    with pytest.raises(ValueError, match=name) as caught:
        factory('adolescent#001', diabetes_type=kind, **{name: 99.0})
    message = str(caught.value)
    assert 'BW' in message and 'Vg' in message and 'Vi' in message


def test_pytree_removes_exactly_four_leaves_and_roundtrips(patient, historical_fixture):
    old_fields = historical_fixture['dataclass_field_order']
    new_fields = [f.name for f in dataclasses.fields(patient)]
    assert new_fields == [name for name in old_fields if name not in REMOVED]
    leaves, tree = jax.tree_util.tree_flatten(patient)
    assert len(leaves) == historical_fixture['pytree_leaf_count'] - 4
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored == patient
    assert (restored.V_G_L, restored.V_I_L) == (20.0, 5.0)


@pytest.mark.parametrize('array', [np.asarray, jnp.asarray])
def test_arithmetic_helpers_accept_scalar_and_vector_arrays(array):
    assert conversion._as_liters_from_Vg(array(2.0), array(100.0)) == 20.0
    assert conversion._as_liters_from_Vi(array(0.05), array(100.0)) == 5.0
    np.testing.assert_array_equal(
        conversion._as_liters_from_Vg(array([2.0, 3.0]), array([100.0, 80.0])),
        [20.0, 24.0])
    np.testing.assert_array_equal(
        conversion._as_liters_from_Vi(array([0.05, 0.1]), array([100.0, 80.0])),
        [5.0, 8.0])


def test_helpers_and_properties_are_jittable(patient):
    compute = jax.jit(lambda p: (p.V_G_L, p.V_I_L))
    for p, expected in ((patient, (20.0, 5.0)),
                        (dataclasses.replace(patient, BW=80.0, Vg=3.0, Vi=0.1),
                         (24.0, 8.0))):
        np.testing.assert_array_equal(np.asarray(compute(p)), expected)
    helper = jax.jit(lambda weight, glucose, insulin: (
        conversion._as_liters_from_Vg(glucose, weight),
        conversion._as_liters_from_Vi(insulin, weight)))
    np.testing.assert_array_equal(np.asarray(helper(100.0, 2.0, 0.05)), [20.0, 5.0])


@pytest.mark.parametrize('markdown', [False, True])
def test_tables_expose_only_two_computed_totals(patient, markdown):
    table = patient.to_table(markdown=markdown)
    rows = [line.strip('| ').split('|')[0].strip().split()[0]
            for line in table.splitlines() if line.strip('| ')]
    for name, value in (('V_G_L', '20.0'), ('V_I_L', '5.0')):
        assert rows.count(name) == 1
        line = next(line for line in table.splitlines() if line.strip('| ').startswith(name))
        assert value in line
        assert patient.UNITS[name] == 'L'
    assert 'V_G' not in rows and 'V_I' not in rows


@pytest.mark.parametrize('protocol', [4, pickle.HIGHEST_PROTOCOL])
def test_new_pickle_roundtrip_preserves_inputs_without_duplicate_storage(patient, protocol):
    restored = pickle.loads(pickle.dumps(patient, protocol=protocol))
    assert restored == patient
    assert vars(restored).keys().isdisjoint(REMOVED)
    replaced = dataclasses.replace(restored, BW=120.0)
    assert (replaced.V_G_L, replaced.V_I_L) == (24.0, 6.0)


@pytest.mark.parametrize('protocol', [4, 5])
def test_actual_prechange_pickle_is_explicitly_rejected(historical_fixture, protocol):
    record = historical_fixture['pickles'][str(protocol)]
    payload = base64.b64decode(record['base64'], validate=True)
    assert hashlib.sha256(payload).hexdigest() == record['sha256']
    # Bytes were emitted by the actual pre-change factory/class, not synthesized.
    with pytest.raises(ValueError, match='(?i)(obsolete|volume)'):
        pickle.loads(payload)


def test_direct_conversion_preserves_decimal_input_accepted_before_cleanup(patient):
    # Insulin-volume arithmetic formerly returned float before unit validation.
    # Decimal is intentionally a direct-input case, outside factory validation.
    supplied = dataclasses.replace(patient, Vi=Decimal('0.05'))
    converted = conversion.patient_to_t2d_params(supplied)
    assert converted.Vi is supplied.Vi
    assert converted.Vi == Decimal('0.05')
    assert converted.BW == 100.0
