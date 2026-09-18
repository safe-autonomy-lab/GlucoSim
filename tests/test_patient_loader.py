"""Preserve the CSV loader's public import and parsing/error contracts."""
from pathlib import Path

import pandas as pd
import pytest

from glucosim.simglucose.core import params
from glucosim.simglucose.core.patient_loader import load_patient_parameters_from_csv


CSV = Path(params.__file__).parent.parent / 'params' / 'vpatient_params.csv'


def test_legacy_import_and_bundled_patients():
    assert params.load_patient_parameters_from_csv is load_patient_parameters_from_csv
    patients = load_patient_parameters_from_csv(CSV)
    assert set(patients) == {f'{cohort}#{i:03d}' for cohort in ('adolescent', 'adult', 'child')
                             for i in range(1, 11)}
    assert all(isinstance(value, float) for patient in patients.values() for value in patient.values())


def test_missing_file_preserves_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='Patient parameters CSV file not found'):
        load_patient_parameters_from_csv(tmp_path / 'absent.csv')


@pytest.mark.parametrize('column', ['BW', 'kmax'])
def test_missing_columns_preserve_parse_error(tmp_path, column):
    frame = pd.read_csv(CSV).iloc[:1].drop(columns=[column])
    path = tmp_path / 'patient.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Failed to parse patient parameters CSV'):
        load_patient_parameters_from_csv(path)


def test_optional_field_and_empty_names(tmp_path):
    frame = pd.read_csv(CSV).iloc[:2].drop(columns=['Fsnc'])
    frame.loc[frame.index[1], 'Name'] = ''
    path = tmp_path / 'patient.csv'
    frame.to_csv(path, index=False)
    patients = load_patient_parameters_from_csv(path)
    assert list(patients) == ['adolescent#001']
    assert patients['adolescent#001']['Fsnc'] == 1.0


def test_invalid_number_preserves_parse_error(tmp_path):
    frame = pd.read_csv(CSV).iloc[:1].astype({'BW': object})
    frame.loc[frame.index[0], 'BW'] = 'invalid'
    path = tmp_path / 'patient.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Failed to parse patient parameters CSV'):
        load_patient_parameters_from_csv(path)
