# Testing Guide for Pysiology

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=pysiology --cov-report=html
```

## Test Organization

```
tests/
├── __init__.py              # Test package
├── conftest.py              # Shared fixtures
├── test_electromyography.py # EMG tests (249 lines)
├── test_electrocardiography.py # ECG tests (183 lines)
├── test_electrodermalactivity.py # GSR tests (124 lines)
├── test_utils.py            # Utility tests (140 lines)
└── README.md                # Detailed test documentation
```

## Test Breakdown

| Module | Test Classes | Test Cases | Coverage |
|--------|--------------|-----------|----------|
| electromyography | 6 | ~60+ | Time/Freq domain, filtering |
| electrocardiography | 4 | ~20+ | HRV, frequency analysis |
| electrodermalactivity | 5 | ~15+ | Filtering, peak detection, SCR |
| utils | 3 | ~13 | Correlation analysis, validation |
| **Total** | **18** | **100+** | **All modules** |

## Common Commands

### Run specific module tests
```bash
pytest tests/test_electromyography.py -v
```

### Run specific test class
```bash
pytest tests/test_electromyography.py::TestTimeDomainFeatures -v
```

### Run specific test
```bash
pytest tests/test_electromyography.py::TestTimeDomainFeatures::test_getIEMG -v
```

### Run with detailed failure info
```bash
pytest -vv --tb=long
```

### Stop on first failure
```bash
pytest -x
```

### Run last 5 failed tests
```bash
pytest --lf --maxfail=5
```

### Generate HTML coverage report
```bash
pytest --cov=pysiology --cov-report=html
open htmlcov/index.html
```

## Test Features

### Fixtures (in conftest.py)
- `sample_emg_signal`: Synthetic EMG with 50Hz + 100Hz components
- `sample_ecg_signal`: ECG simulation with heart rate
- `sample_gsr_signal`: GSR with baseline and phasic activity
- `sample_signals_for_correlation`: Multiple correlated signals
- `sample_peaks`: Synthetic peak indices

### Test Coverage
- ✓ Feature calculation correctness
- ✓ Output type validation
- ✓ Output range validation
- ✓ Input error handling
- ✓ Edge cases
- ✓ Signal processing pipeline integrity

## GitHub Actions CI/CD

Tests run automatically on:
- Push to `master` or `develop` branches
- Pull requests to `master` or `develop`
- Multiple Python versions: 3.8, 3.9, 3.10, 3.11
- Coverage reports uploaded to Codecov

## Writing New Tests

Template:
```python
def test_my_feature(self, sample_emg_signal):
    """Test description."""
    signal, samplerate = sample_emg_signal
    result = emg.myFunction(signal)
    
    assert isinstance(result, (int, float, np.number))
    assert result >= 0, "Should be non-negative"
```

Guidelines:
- Use descriptive names starting with `test_`
- Group related tests in classes
- Use fixtures for test data
- Include meaningful assertions
- Document expected behavior

## Troubleshooting

### Import errors
```bash
pip install -r requirements.txt
```

### pytest not found
```bash
pip install pytest pytest-cov
```

### Coverage not generating
```bash
pytest --cov=pysiology --cov-report=html --cov-report=term-missing
```

## See Also
- `tests/README.md` - Detailed test documentation
- `TESTS_SUMMARY.md` - Test suite overview
- `.github/workflows/tests.yml` - CI/CD configuration
