"""Unit tests for electrodermalactivity module."""

import pytest
import numpy as np
from pysiology import electrodermalactivity as gsr


class TestGSRFiltering:
    """Test GSR filtering functions."""
    
    def test_phasicGSRFilter(self, sample_gsr_signal):
        """Test phasic GSR filter."""
        signal, samplerate = sample_gsr_signal
        result = gsr.phasicGSRFilter(signal, samplerate, seconds=4)
        
        assert isinstance(result, list)
        assert len(result) == len(signal), "Filtered signal should have same length as input"
    
    def test_tonicGSRFilter(self, sample_gsr_signal):
        """Test tonic GSR filter."""
        signal, samplerate = sample_gsr_signal
        result = gsr.tonicGSRFilter(signal, samplerate, seconds=4)
        
        assert isinstance(result, list)
        assert len(result) == len(signal), "Filtered signal should have same length as input"
    
    def test_getPhasicAndTonic(self, sample_gsr_signal):
        """Test phasic and tonic decomposition."""
        signal, samplerate = sample_gsr_signal
        phasic, tonic = gsr.getPhasicAndTonic(signal, samplerate, seconds=4)
        
        assert isinstance(phasic, list)
        assert isinstance(tonic, list)
        assert len(phasic) == len(signal)
        assert len(tonic) == len(signal)
        
        # Phasic + tonic should approximately equal original signal
        reconstructed = np.array(phasic) + np.array(tonic)
        np.testing.assert_allclose(reconstructed, signal, rtol=1e-10)


class TestPeakDetection:
    """Test peak detection functions."""
    
    def test_findPeakOnsetAndOffset(self, sample_gsr_signal):
        """Test peak onset and offset detection."""
        signal, samplerate = sample_gsr_signal
        peaks = gsr.findPeakOnsetAndOffset(signal, onset=0.01, offset=0)
        
        assert isinstance(peaks, list)
        # Each peak should have 3 indices: [onset, max, offset]
        for peak in peaks:
            assert len(peak) == 3
            assert peak[0] <= peak[1] <= peak[2], "Onset should be <= max <= offset"


class TestSCRFeatures:
    """Test SCR (Skin Conductance Response) features."""
    
    def test_GSRSCRFeaturesExtraction(self, sample_gsr_signal):
        """Test SCR features extraction."""
        signal, samplerate = sample_gsr_signal
        peaks = gsr.findPeakOnsetAndOffset(signal, onset=0.01, offset=0)
        
        if len(peaks) > 0:
            peak = peaks[0]
            result = gsr.GSRSCRFeaturesExtraction(signal, samplerate, peak)
            
            assert isinstance(result, dict)
            # Should contain SCR-related features
            assert 'riseTime' in result or 'amplitude' in result


class TestFiltering:
    """Test filter functions."""
    
    def test_butter_lowpass_filter(self, sample_gsr_signal):
        """Test lowpass Butterworth filter."""
        signal, samplerate = sample_gsr_signal
        cutoff = 1
        filtered = gsr.butter_lowpass_filter(signal, cutoff, samplerate, order=2)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)
    
    def test_butter_highpass_filter(self, sample_gsr_signal):
        """Test highpass Butterworth filter."""
        signal, samplerate = sample_gsr_signal
        cutoff = 0.05
        filtered = gsr.butter_highpass_filter(signal, cutoff, samplerate, order=2)
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)


class TestAnalyzeGSR:
    """Test the main GSR analysis function."""
    
    def test_analyzeGSR_default_parameters(self, sample_gsr_signal):
        """Test analyzeGSR with default parameters."""
        signal, samplerate = sample_gsr_signal
        result = gsr.analyzeGSR(signal, samplerate)
        
        assert isinstance(result, dict)
    
    def test_analyzeGSR_without_preprocessing(self, sample_gsr_signal):
        """Test analyzeGSR without preprocessing."""
        signal, samplerate = sample_gsr_signal
        result = gsr.analyzeGSR(signal, samplerate, preprocessing=False)
        
        assert isinstance(result, dict)
    
    def test_analyzeGSR_with_preprocessing(self, sample_gsr_signal):
        """Test analyzeGSR with preprocessing."""
        signal, samplerate = sample_gsr_signal
        result = gsr.analyzeGSR(
            signal, samplerate,
            preprocessing=True,
            lowpass=1,
            highpass=0.05,
            phasic_seconds=10
        )
        
        assert isinstance(result, dict)
