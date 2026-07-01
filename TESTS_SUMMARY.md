# Pysiology Test Suite - Summary

## Overview
A comprehensive unit test suite for the Pysiology package using pytest framework.

## Test Statistics

### Test Files Created
- `tests/conftest.py` - Shared fixtures and test data generators
- `tests/test_electromyography.py` - 16 test classes, ~60+ test cases
- `tests/test_electrocardiography.py` - 4 test classes, ~20+ test cases
- `tests/test_electrodermalactivity.py` - 5 test classes, ~15+ test cases
- `tests/test_utils.py` - 3 test classes, ~13 test cases
- `tests/README.md` - Comprehensive test documentation
- `tests/__init__.py` - Test package initialization

### Total Coverage
- **4 modules tested**: electromyography, electrocardiography, electrodermalactivity, utils
- **100+ individual test cases** across the entire package
- **Configuration files**: pytest.ini for test discovery and execution

## Modules Tested

### electromyography.py
- ✓ Time-domain features (IEMG, MAV, MAV1, MAV2, SSI, VAR, TM, RMS, LOG, WL, AAC, DASDV)
- ✓ Frequency-domain features (PSD, MNF, MDF, peak frequency)
- ✓ Zero-crossing and threshold features (ZC, MYOP, WAMP, SSC)
- ✓ Butterworth filters (lowpass, highpass)
- ✓ Main EMG analysis pipeline

### electrocardiography.py
- ✓ Heart rate variability (IBI, BPM, SDNN, SDSD, RMSSD, pNN50, pNN20)
- ✓ Frequency analysis (PSD, LF/HF/VLF bands)
- ✓ Butterworth filters
- ✓ Main ECG analysis pipeline

### electrodermalactivity.py
- ✓ Phasic and tonic filtering
- ✓ Peak detection (onset/offset)
- ✓ SCR features extraction
- ✓ Butterworth filters
- ✓ Main GSR analysis pipeline

### utils.py
- ✓ Leandri correlation analysis (now with fixed time array mismatch)
- ✓ Dictionary to dataframe conversion
- ✓ Input validation

## Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=pysiology

# Run specific module tests
pytest tests/test_electromyography.py -v
```

## Test Fixtures

All tests use parameterized synthetic signals:
- **EMG signal**: 50Hz + 100Hz components + noise
- **ECG signal**: Heart rate simulation (1 Hz fundamental)
- **GSR signal**: Baseline + phasic activity + noise
- **Correlation signals**: 10 correlated signals with common/unique components
- **Sample peaks**: Synthetic peak indices for HRV analysis

## Integration

Tests are designed for CI/CD integration:
- Compatible with GitHub Actions, GitLab CI, Jenkins, etc.
- Coverage report generation (XML/HTML)
- Exit codes for pass/fail status

## Quality Assurance

✓ All test files compile without syntax errors
✓ Tests follow pytest conventions
✓ Comprehensive docstrings for all test cases
✓ Input validation and edge case coverage
✓ Output type and range checking
