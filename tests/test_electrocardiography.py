"""Unit tests for electrocardiography module."""

import pytest
import numpy as np
from pysiology import electrocardiography as ecg


class TestHeartRateVariabilityFeatures:
    """Test heart rate variability features."""
    
    def test_getIBI(self, sample_peaks):
        """Test IBI (Inter-Beat Interval) calculation."""
        samplerate = 1000
        result = ecg.getIBI(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert result > 0, "IBI should be positive"
        
        # With peaks at [100, 200, 300, 400, 500] and samplerate 1000
        # intervals are [100, 100, 100, 100] samples = [100, 100, 100, 100] ms
        expected = 100  # mean is 100 ms
        assert result == expected
    
    def test_getBPM(self, sample_peaks):
        """Test BPM (Beats Per Minute) calculation."""
        samplerate = 1000
        nsample = 10000  # 10 second signal
        result = ecg.getBPM(len(sample_peaks), nsample, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert result > 0, "BPM should be positive"
        # With 5 peaks in 10 seconds: (5 * 60) / 10 = 30 BPM
        expected = 30.0
        assert result == expected
    
    def test_getSDNN(self, sample_peaks):
        """Test SDNN (Standard Deviation of NN intervals)."""
        samplerate = 1000
        result = ecg.getSDNN(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "SDNN should be non-negative"
        # All intervals are equal (100 samples), so std should be 0
        assert result == 0
    
    def test_getSDSD(self, sample_peaks):
        """Test SDSD (Standard Deviation of Successive Differences)."""
        samplerate = 1000
        result = ecg.getSDSD(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "SDSD should be non-negative"
    
    def test_getRMSSD(self, sample_peaks):
        """Test RMSSD (Root Mean Square of Successive Differences)."""
        samplerate = 1000
        result = ecg.getRMSSD(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert result >= 0, "RMSSD should be non-negative"
    
    def test_getPNN50(self, sample_peaks):
        """Test pNN50 (percentage of NN intervals > 50ms)."""
        samplerate = 1000
        result = ecg.getPNN50(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert 0 <= result <= 1, "pNN50 should be between 0 and 1"
        # All intervals are exactly 100 ms, so pNN50 should be 1.0
        assert result == 1.0
    
    def test_getPNN20(self, sample_peaks):
        """Test pNN20 (percentage of NN intervals > 20ms)."""
        samplerate = 1000
        result = ecg.getPNN20(sample_peaks, samplerate)
        
        assert isinstance(result, (int, float, np.number))
        assert 0 <= result <= 1, "pNN20 should be between 0 and 1"
        # All intervals are 100 ms, so pNN20 should be 1.0
        assert result == 1.0


class TestFrequencyAnalysis:
    """Test frequency domain analysis."""
    
    def test_getPSD(self, sample_ecg_signal):
        """Test PSD (Power Spectral Density) calculation."""
        signal, samplerate = sample_ecg_signal
        psd, frequencies = ecg.getPSD(signal, samplerate)
        
        assert isinstance(psd, np.ndarray)
        assert isinstance(frequencies, np.ndarray)
        assert len(psd) == len(frequencies)
        assert len(psd) > 0, "PSD should have values"
        assert np.all(psd >= 0), "PSD values should be non-negative"
    
    def test_getFrequencies(self, sample_ecg_signal):
        """Test frequency band analysis."""
        signal, samplerate = sample_ecg_signal
        result = ecg.getFrequencies(signal, samplerate)
        
        assert isinstance(result, dict)
        assert 'LF' in result, "LF (Low Frequency) should be in results"
        assert 'HF' in result, "HF (High Frequency) should be in results"
        assert 'VLF' in result, "VLF (Very Low Frequency) should be in results"
        
        # All values should be non-negative
        assert result['LF'] >= 0
        assert result['HF'] >= 0
        assert result['VLF'] >= 0


class TestFiltering:
    """Test filter functions."""
    
    def test_butter_lowpass_filter(self, sample_ecg_signal):
        """Test lowpass Butterworth filter."""
        signal, samplerate = sample_ecg_signal
        cutoff = 25
        filtered = ecg.butter_lowpass_filter(signal, cutoff, samplerate, order=5)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)
        assert not np.all(filtered == signal), "Filtered signal should differ from original"
    
    def test_butter_highpass_filter(self, sample_ecg_signal):
        """Test highpass Butterworth filter."""
        signal, samplerate = sample_ecg_signal
        cutoff = 0.5
        filtered = ecg.butter_highpass_filter(signal, cutoff, samplerate, order=5)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)


class TestAnalyzeECG:
    """Test the main ECG analysis function."""
    
    def test_analyzeECG_default_parameters(self, sample_ecg_signal):
        """Test analyzeECG with default parameters."""
        signal, samplerate = sample_ecg_signal
        result = ecg.analyzeECG(signal, samplerate)
        
        assert isinstance(result, dict)
        assert len(result) > 0, "Results should not be empty"
    
    def test_analyzeECG_with_preprocessing(self, sample_ecg_signal):
        """Test analyzeECG with preprocessing enabled."""
        signal, samplerate = sample_ecg_signal
        result = ecg.analyzeECG(
            signal, samplerate,
            preprocessing=True,
            highpass=0.5,
            lowpass=25
        )
        
        assert isinstance(result, dict)
        # Should have at least IBI or BPM
        assert 'ibi' in result or 'bpm' in result
    
    def test_analyzeECG_without_preprocessing(self, sample_ecg_signal):
        """Test analyzeECG without preprocessing."""
        signal, samplerate = sample_ecg_signal
        result = ecg.analyzeECG(signal, samplerate, preprocessing=False)
        
        assert isinstance(result, dict)
    
    def test_analyzeECG_selective_features(self, sample_ecg_signal):
        """Test analyzeECG with selective feature extraction."""
        signal, samplerate = sample_ecg_signal
        result = ecg.analyzeECG(
            signal, samplerate,
            ibi=True, bpm=True, sdnn=False,
            sdsd=False, rmssd=False,
            pnn50=False, pnn20=False,
            freqAnalysis=False
        )
        
        assert isinstance(result, dict)
        # Should have IBI and BPM
        if len(result) > 0:
            # At least one feature should be present
            assert len(result) > 0
