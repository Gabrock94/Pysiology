"""Unit tests for utils module."""

import pytest
import numpy as np
from pysiology import utils


class TestLeandriCC:
    """Test Leandri correlation analysis function."""
    
    def test_leandriCC_basic(self, sample_signals_for_correlation):
        """Test basic leandriCC functionality."""
        signals, samplerate = sample_signals_for_correlation
        
        times, avg, median = utils.leandriCC(signals, samplerate, 0.050, tmin=0, tmax=0.9)
        
        assert isinstance(times, np.ndarray)
        assert isinstance(avg, np.ndarray)
        assert isinstance(median, np.ndarray)
        
        # All should have matching lengths
        assert len(times) == len(avg) == len(median), "Times, avg, and median should have same length"
    
    def test_leandriCC_with_nAverages(self, sample_signals_for_correlation):
        """Test leandriCC with signal averaging."""
        signals, samplerate = sample_signals_for_correlation
        
        times, avg, median = utils.leandriCC(
            signals, samplerate, 0.050, 
            nAverages=5, tmin=0, tmax=1.0
        )
        
        assert len(times) == len(avg) == len(median)
    
    def test_leandriCC_with_time_window(self, sample_signals_for_correlation):
        """Test leandriCC with specific time window."""
        signals, samplerate = sample_signals_for_correlation
        
        times, avg, median = utils.leandriCC(
            signals, samplerate, 0.050,
            tmin=0.1, tmax=0.9
        )
        
        assert len(times) == len(avg) == len(median)
        assert times[0] >= 0.1, "Start time should match tmin"
    
    def test_leandriCC_mismatched_signals(self):
        """Test that leandriCC raises error for mismatched signal lengths."""
        signals = [
            np.random.normal(0, 1, 1000),
            np.random.normal(0, 1, 500),  # Different length
        ]
        
        with pytest.raises(Exception):
            utils.leandriCC(signals, 1000, 0.050)
    
    def test_leandriCC_invalid_nAverages(self, sample_signals_for_correlation):
        """Test that leandriCC raises error for invalid nAverages."""
        signals, samplerate = sample_signals_for_correlation
        
        with pytest.raises(Exception):
            utils.leandriCC(signals, samplerate, 0.050, nAverages=1)
    
    def test_leandriCC_invalid_time_range(self, sample_signals_for_correlation):
        """Test that leandriCC validates time ranges."""
        signals, samplerate = sample_signals_for_correlation
        
        # tmax < tmin should raise error
        with pytest.raises(Exception):
            utils.leandriCC(signals, samplerate, 0.050, tmin=0.5, tmax=0.2)
    
    def test_leandriCC_tmax_exceeds_signal(self, sample_signals_for_correlation):
        """Test that leandriCC validates tmax against signal length."""
        signals, samplerate = sample_signals_for_correlation
        signal_duration = len(signals[0]) / samplerate  # in seconds
        
        # tmax beyond signal should raise error
        with pytest.raises(Exception):
            utils.leandriCC(signals, samplerate, 0.050, tmax=signal_duration + 1)
    
    def test_leandriCC_output_ranges(self, sample_signals_for_correlation):
        """Test that output values are within expected ranges."""
        signals, samplerate = sample_signals_for_correlation
        
        times, avg, median = utils.leandriCC(signals, samplerate, 0.050)
        
        # Correlation values should be between -1 and 1
        assert np.all(avg >= -1) and np.all(avg <= 1)
        assert np.all(median >= -1) and np.all(median <= 1)


class TestDictToDataFrame:
    """Test dictionary to dataframe conversion."""
    
    def test_dict_to_dataframe_simple(self):
        """Test conversion of simple nested dict."""
        test_dict = {
            'TimeDomain': {'IEMG': 100, 'MAV': 50},
            'FrequencyDomain': {'MNF': 150, 'MDF': 160}
        }
        
        df = utils.dict_to_dataframe(test_dict)
        
        assert df is not None
        # Check that nested keys are flattened
        assert 'TimeDomain_IEMG' in df.columns or 'TimeDomain_IEMG' in df.values
    
    def test_dict_to_dataframe_custom_separator(self):
        """Test conversion with custom separator."""
        test_dict = {
            'Feature1': {'Sub1': 100},
            'Feature2': {'Sub2': 200}
        }
        
        df = utils.dict_to_dataframe(test_dict, sep='.')
        
        assert df is not None


class TestInputValidation:
    """Test input validation and edge cases."""
    
    def test_leandriCC_negative_tmin(self, sample_signals_for_correlation):
        """Test that negative tmin raises error."""
        signals, samplerate = sample_signals_for_correlation
        
        with pytest.raises(Exception):
            utils.leandriCC(signals, samplerate, 0.050, tmin=-0.1)
    
    def test_leandriCC_single_window(self, sample_signals_for_correlation):
        """Test leandriCC with window size larger than signal."""
        signals, samplerate = sample_signals_for_correlation
        signal_duration = len(signals[0]) / samplerate
        
        # Window larger than signal should still work
        times, avg, median = utils.leandriCC(
            signals, samplerate, signal_duration + 0.1
        )
        
        assert len(times) == len(avg) == len(median)
