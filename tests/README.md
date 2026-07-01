# Pysiology Test Suite

This directory contains comprehensive unit tests for the Pysiology package.

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage report
```bash
pytest --cov=pysiology --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_electromyography.py
```

### Run specific test class
```bash
pytest tests/test_electromyography.py::TestTimeDomainFeatures
```

### Run specific test function
```bash
pytest tests/test_electromyography.py::TestTimeDomainFeatures::test_getIEMG
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests and stop on first failure
```bash
pytest -x
```

## Test Structure

### conftest.py
Contains shared fixtures for generating synthetic test signals and peaks:
- `sample_emg_signal`: EMG signal with 50Hz and 100Hz components
- `sample_ecg_signal`: ECG signal with heart rate simulation
- `sample_gsr_signal`: GSR signal with baseline and phasic activity
- `sample_signals_for_correlation`: Multiple signals for correlation analysis
- `sample_peaks`: Synthetic peak indices for ECG analysis

### test_electromyography.py
Tests for the electromyography module:
- **TestTimeDomainFeatures**: IEMG, MAV, MAV1, MAV2, SSI, VAR, RMS, LOG, TM, WL, AAC, DASDV
- **TestFrequencyDomainFeatures**: PSD, MNF, MDF, peak frequency
- **TestZeroCrossingAndThreshold**: ZC, MYOP, WAMP, SSC
- **TestFiltering**: Butterworth lowpass/highpass filters
- **TestAnalyzeEMG**: Main EMG analysis function

### test_electrocardiography.py
Tests for the electrocardiography module:
- **TestHeartRateVariabilityFeatures**: IBI, BPM, SDNN, SDSD, RMSSD, pNN50, pNN20
- **TestFrequencyAnalysis**: PSD, frequency bands (LF, HF, VLF)
- **TestFiltering**: Butterworth filters
- **TestAnalyzeECG**: Main ECG analysis function

### test_electrodermalactivity.py
Tests for the electrodermalactivity (GSR) module:
- **TestGSRFiltering**: Phasic/tonic filtering, decomposition
- **TestPeakDetection**: Peak onset and offset detection
- **TestSCRFeatures**: SCR features extraction
- **TestFiltering**: Butterworth filters
- **TestAnalyzeGSR**: Main GSR analysis function

### test_utils.py
Tests for the utils module:
- **TestLeandriCC**: Correlation analysis function
- **TestDictToDataFrame**: Dictionary to dataframe conversion
- **TestInputValidation**: Input validation and edge cases

## Test Coverage

Run coverage analysis:
```bash
pytest --cov=pysiology --cov-report=html --cov-report=term-missing
```

View HTML report:
```bash
open htmlcov/index.html
```

## Writing New Tests

When adding new tests:

1. Use descriptive test names starting with `test_`
2. Use fixtures for common test data
3. Include docstrings explaining what is being tested
4. Use assertions with meaningful messages
5. Group related tests in classes

Example:
```python
def test_feature_calculation(self, sample_emg_signal):
    """Test that feature calculation returns expected type and range."""
    signal, samplerate = sample_emg_signal
    result = emg.someFeature(signal)
    
    assert isinstance(result, (int, float, np.number))
    assert result >= 0, "Feature should be non-negative"
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines (GitHub Actions, GitLab CI, etc.) by running:
```bash
pytest --cov=pysiology --cov-report=xml
```
