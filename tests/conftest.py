"""Pytest configuration and fixtures for Pysiology tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_emg_signal():
    """Generate a synthetic EMG signal for testing."""
    samplerate = 1000
    duration = 1
    samples = samplerate * duration
    
    # Create a signal with some frequency content
    t = np.linspace(0, duration, samples)
    signal = (
        np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
        0.5 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz component
        0.1 * np.random.normal(0, 1, samples)  # noise
    )
    return signal, samplerate


@pytest.fixture
def sample_ecg_signal():
    """Generate a synthetic ECG signal for testing."""
    samplerate = 1000
    duration = 5
    samples = samplerate * duration
    
    # Create a signal with heart rate around 60 bpm (1 beat/sec)
    t = np.linspace(0, duration, samples)
    signal = (
        np.sin(2 * np.pi * 1 * t) +  # 1 Hz fundamental (60 bpm)
        0.5 * np.sin(2 * np.pi * 3 * t) +  # harmonics
        0.1 * np.random.normal(0, 1, samples)
    )
    return signal, samplerate


@pytest.fixture
def sample_gsr_signal():
    """Generate a synthetic GSR signal for testing."""
    samplerate = 100
    duration = 10
    samples = samplerate * duration
    
    # GSR signal: baseline with some phasic activity
    t = np.linspace(0, duration, samples)
    baseline = 2 + 0.1 * np.sin(2 * np.pi * 0.1 * t)  # slow baseline drift
    phasic = 0.5 * np.exp(-t / 2) * np.sin(2 * np.pi * 0.5 * t)  # phasic activity
    noise = 0.05 * np.random.normal(0, 1, samples)
    signal = baseline + phasic + noise
    
    return signal, samplerate


@pytest.fixture
def sample_signals_for_correlation():
    """Generate multiple signals for correlation analysis."""
    samplerate = 1000
    duration = 1
    samples = samplerate * duration
    n_signals = 10
    
    t = np.linspace(0, duration, samples)
    signals = []
    
    for i in range(n_signals):
        # Create signals with common and unique components
        common = np.sin(2 * np.pi * 10 * t)
        unique = 0.5 * np.sin(2 * np.pi * (20 + i * 5) * t)
        noise = 0.1 * np.random.normal(0, 1, samples)
        signal = common + unique + noise
        signals.append(signal)
    
    return signals, samplerate


@pytest.fixture
def sample_peaks():
    """Generate sample peak indices for testing."""
    # Peaks at 100, 200, 300, 400, 500 samples (simulating heartbeats)
    return np.array([100, 200, 300, 400, 500])
