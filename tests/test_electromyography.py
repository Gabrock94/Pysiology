"""Unit tests for electromyography module."""

import pytest
import numpy as np
from pysiology import electromyography as emg


class TestTimeDomainFeatures:
    """Test time-domain EMG features."""
    
    def test_getIEMG(self, sample_emg_signal):
        """Test IEMG (Integrated EMG) calculation."""
        signal, _ = sample_emg_signal
        result = emg.getIEMG(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result > 0, "IEMG should be positive"
        
        # Test with known input
        test_signal = np.array([1, -2, 3, -4, 5])
        expected = 1 + 2 + 3 + 4 + 5  # sum of absolute values
        assert emg.getIEMG(test_signal) == expected
    
    def test_getMAV(self, sample_emg_signal):
        """Test MAV (Mean Absolute Value) calculation."""
        signal, _ = sample_emg_signal
        result = emg.getMAV(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result > 0, "MAV should be positive"
        
        # MAV should be less than or equal to max absolute value
        assert result <= np.max(np.abs(signal))
    
    def test_getMAV1(self, sample_emg_signal):
        """Test MAV1 (Modified Mean Absolute Value 1)."""
        signal, _ = sample_emg_signal
        result = emg.getMAV1(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "MAV1 should be non-negative"
    
    def test_getMAV2(self, sample_emg_signal):
        """Test MAV2 (Modified Mean Absolute Value 2)."""
        signal, _ = sample_emg_signal
        result = emg.getMAV2(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "MAV2 should be non-negative"
    
    def test_getSSI(self, sample_emg_signal):
        """Test SSI (Sum of Squared Inputs)."""
        signal, _ = sample_emg_signal
        result = emg.getSSI(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "SSI should be non-negative"
    
    def test_getVAR(self, sample_emg_signal):
        """Test VAR (Variance)."""
        signal, _ = sample_emg_signal
        result = emg.getVAR(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "Variance should be non-negative"
    
    def test_getRMS(self, sample_emg_signal):
        """Test RMS (Root Mean Square)."""
        signal, _ = sample_emg_signal
        result = emg.getRMS(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "RMS should be non-negative"
    
    def test_getLOG(self, sample_emg_signal):
        """Test LOG (Log Detector)."""
        signal, _ = sample_emg_signal
        result = emg.getLOG(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "LOG should be non-negative"
    
    def test_getTM(self, sample_emg_signal):
        """Test TM (Temporal Moment)."""
        signal, _ = sample_emg_signal
        result = emg.getTM(signal, order=3)
        
        assert isinstance(result, (int, float, np.number))
    
    def test_getWL(self, sample_emg_signal):
        """Test WL (Waveform Length)."""
        signal, _ = sample_emg_signal
        result = emg.getWL(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "WL should be non-negative"
    
    def test_getAAC(self, sample_emg_signal):
        """Test AAC (Average Amplitude Change)."""
        signal, _ = sample_emg_signal
        result = emg.getAAC(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "AAC should be non-negative"
    
    def test_getDASDV(self, sample_emg_signal):
        """Test DASDV (Difference Absolute Standard Deviation Value)."""
        signal, _ = sample_emg_signal
        result = emg.getDASDV(signal)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "DASDV should be non-negative"


class TestFrequencyDomainFeatures:
    """Test frequency-domain EMG features."""
    
    def test_getPSD(self, sample_emg_signal):
        """Test PSD (Power Spectral Density) calculation."""
        signal, samplerate = sample_emg_signal
        psd, frequencies = emg.getPSD(signal, samplerate)
        
        assert isinstance(psd, np.ndarray)
        assert isinstance(frequencies, np.ndarray)
        assert len(psd) == len(frequencies)
        assert len(psd) > 0, "PSD should have values"
        assert np.all(psd >= 0), "PSD values should be non-negative"
    
    def test_getMNF(self, sample_emg_signal):
        """Test MNF (Mean Frequency)."""
        signal, samplerate = sample_emg_signal
        psd, frequencies = emg.getPSD(signal, samplerate)
        result = emg.getMNF(psd, frequencies)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "MNF should be non-negative"
        assert result <= frequencies[-1], "MNF should be within frequency range"
    
    def test_getMDF(self, sample_emg_signal):
        """Test MDF (Median Frequency)."""
        signal, samplerate = sample_emg_signal
        psd, frequencies = emg.getPSD(signal, samplerate)
        result = emg.getMDF(psd, frequencies)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "MDF should be non-negative"
        assert result <= frequencies[-1], "MDF should be within frequency range"
    
    def test_getPeakFrequency(self, sample_emg_signal):
        """Test peak frequency detection."""
        signal, samplerate = sample_emg_signal
        psd, frequencies = emg.getPSD(signal, samplerate)
        result = emg.getPeakFrequency(psd, frequencies)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "Peak frequency should be non-negative"


class TestZeroCrossingAndThreshold:
    """Test zero-crossing and threshold-based features."""
    
    def test_getZC(self, sample_emg_signal):
        """Test ZC (Zero Crossing)."""
        signal, _ = sample_emg_signal
        result = emg.getZC(signal, threshold=0.01)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "ZC should be non-negative"
    
    def test_getMYOP(self, sample_emg_signal):
        """Test MYOP (Myopulse Percentage Rate)."""
        signal, _ = sample_emg_signal
        result = emg.getMYOP(signal, threshold=0.01)
        
        assert isinstance(result, (int, float, np.number))
        assert 0 <= result <= 1, "MYOP should be between 0 and 1"
    
    def test_getWAMP(self, sample_emg_signal):
        """Test WAMP (Willison Amplitude)."""
        signal, _ = sample_emg_signal
        result = emg.getWAMP(signal, threshold=0.01)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "WAMP should be non-negative"
    
    def test_getSSC(self, sample_emg_signal):
        """Test SSC (Slope Sign Changes)."""
        signal, _ = sample_emg_signal
        result = emg.getSSC(signal, threshold=0.01)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "SSC should be non-negative"


class TestFiltering:
    """Test filter functions."""
    
    def test_butter_lowpass_filter(self, sample_emg_signal):
        """Test lowpass Butterworth filter."""
        signal, samplerate = sample_emg_signal
        cutoff = 50
        filtered = emg.butter_lowpass_filter(signal, cutoff, samplerate, order=5)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)
        assert not np.all(filtered == signal), "Filtered signal should differ from original"
    
    def test_butter_highpass_filter(self, sample_emg_signal):
        """Test highpass Butterworth filter."""
        signal, samplerate = sample_emg_signal
        cutoff = 20
        filtered = emg.butter_highpass_filter(signal, cutoff, samplerate, order=5)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)


class TestAnalyzeEMG:
    """Test the main EMG analysis function."""
    
    def test_analyzeEMG_default_parameters(self, sample_emg_signal):
        """Test analyzeEMG with default parameters."""
        signal, samplerate = sample_emg_signal
        result = emg.analyzeEMG(signal, samplerate)
        
        assert isinstance(result, dict)
        # Check that some expected keys are present
        expected_keys = ['TimeDomain', 'FrequencyDomain']
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in results"
    
    def test_analyzeEMG_no_preprocessing(self, sample_emg_signal):
        """Test analyzeEMG without preprocessing."""
        signal, samplerate = sample_emg_signal
        result = emg.analyzeEMG(signal, samplerate, preprocessing=False)
        
        assert isinstance(result, dict)
        assert len(result) > 0, "Result should not be empty"
    
    def test_analyzeEMG_selective_analysis(self, sample_emg_signal):
        """Test analyzeEMG with selective feature analysis."""
        signal, samplerate = sample_emg_signal
        result = emg.analyzeEMG(
            signal, samplerate, 
            preprocessing=True,
            nseg=2
        )
        
        assert isinstance(result, dict)
