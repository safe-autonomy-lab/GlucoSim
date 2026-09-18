"""CSV decoding for virtual patients; adaptation and calibration live elsewhere."""
import logging
import os
from typing import Dict

import pandas as pd

# Retain the legacy logging channel and its caller-configured level.
logger = logging.getLogger('glucosim.simglucose.core.params')


def load_patient_parameters_from_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load patient parameters from CSV file and return as dictionary.
    
    Args:
        csv_path: Path to the vpatient_params.csv file
        
    Returns:
        Dictionary mapping patient names to their parameter dictionaries
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Patient parameters CSV file not found: {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded patient data for {len(df)} patients from {csv_path}")
        
        # Validate required columns
        required_columns = ['Name', 'BW', 'EGPb', 'Gb', 'Ib', 'u2ss', 'Vg', 'Vi', 'Ipb', 'Ilb', 'Gpb', 'Gtb']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Convert to dictionary format
        patient_data = {}
        for _, row in df.iterrows():
            patient_name = row['Name']
            if pd.isna(patient_name) or patient_name == '':
                continue  # Skip empty rows
                
            # Extract parameters from CSV row
            params = {
                # Core Physiological
                'BW': float(row['BW']),
                'EGPb': float(row['EGPb']), # mg/kg/min
                'Gb': float(row['Gb']), # mg/dL
                'Ib': float(row['Ib']),
                'u2ss': float(row['u2ss']),
                'Vg': float(row['Vg']), # dL/kg
                'Vi': float(row['Vi']), # L/kg
                'Ipb': float(row['Ipb']),
                'Ilb': float(row['Ilb']),
                'Gpb': float(row['Gpb']),
                'Gtb': float(row['Gtb']),
                'Fsnc': float(row.get('Fsnc', 1.0)),  # mg/kg/min
                
                # Meal Absorption
                'kmax': float(row['kmax']),
                'kmin': float(row['kmin']),
                'kabs': float(row['kabs']),
                'b': float(row['b']),
                'd': float(row['d']),
                'f': float(row['f']),
                
                # Insulin Kinetics
                'ka1': float(row['ka1']),
                'ka2': float(row['ka2']),
                'kd': float(row['kd']),
                'ksc': float(row['ksc']),
                'm1': float(row['m1']),
                'm2': float(row['m2']),
                'm30': float(row['m30']),
                'm4': float(row['m4']),
                'm5': float(row['m5']),
                'CL': float(row['CL']),
                'HEb': float(row['HEb']),
                
                # Insulin Action
                'Vmx': float(row['Vmx']),
                'Vm0': float(row['Vm0']),
                'Km0': float(row['Km0']),
                'p2u': float(row['p2u']),
                'ki': float(row['ki']),
                
                # Glucose Kinetics
                'kp1': float(row['kp1']),
                'kp2': float(row['kp2']),
                'kp3': float(row['kp3']),
                'k1': float(row['k1']),
                'k2': float(row['k2']),
                'ke1': float(row['ke1']),
                'ke2': float(row['ke2']),
                'Rdb': float(row['Rdb']),
                'PCRb': float(row['PCRb']),
            }
            
            patient_data[patient_name] = params
            logger.debug(f"Loaded parameters for patient: {patient_name}")
            
        return patient_data
        
    except Exception as e:
        logger.error(f"Error loading patient parameters from CSV: {e}")
        raise ValueError(f"Failed to parse patient parameters CSV: {e}")
