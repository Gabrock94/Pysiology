# -*- coding: utf-8 -*-
""" This module is used to load sample data from the package data folder. """
import pickle
from pathlib import Path

# Identify the directory where this script (e.g., sample_loader.py) is located
# and point to the 'data' subdirectory within the package.
DATA_DIR = Path(__file__).parent / 'data'

def _load_pickle_file(filename):
    """ Internal helper to locate and load pickle files. """
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Sample data file not found at {file_path}. "
            "Ensure the 'data' folder is included in your pip installation."
        )
        
    with open(file_path, "rb") as f:
        return pickle.load(f)

def loadsampleECG():
    """ Loads the sample ECG signal. """
    return _load_pickle_file('convertedECG.pkl')
 
def loadsampleEMG():
    """ Loads the sample EMG signal. """
    return _load_pickle_file('convertedEMG.pkl')
    
def loadsampleEDA():
    """ Loads the sample EDA (GSR) signal. """
    return _load_pickle_file('convertedEDA.pkl')