import pytest
from unittest.mock import patch, MagicMock
from llm_compression.action.safety import SafetyMonitor, SafetyViolation

@pytest.fixture
def safety_monitor():
    return SafetyMonitor(screen_size=(1920, 1080))

def test_initialization(safety_monitor):
    assert safety_monitor.screen_size == (1920, 1080)
    assert safety_monitor.min_interval == 0.1

def test_coordinate_bounds(safety_monitor):
    # Valid coordinates
    # We must patch time to avoid rate limit hitting first if tests run fast?
    # Actually rate limit is checked first.
    # Initial last_action_time is 0. So first call with current time (mocked) > 0.1 should pass.
    
    with patch('time.time', return_value=100.0):
        assert safety_monitor.validate("move", {"x": 100, "y": 100}) is True
        
    with patch('time.time', return_value=101.0):
        # Different time, safe
        assert safety_monitor.validate("click", {"x": 0, "y": 0}) is True

    # Invalid coordinates
    with patch('time.time', return_value=102.0):
        with pytest.raises(SafetyViolation, match="out of bounds"):
            safety_monitor.validate("move", {"x": 1920, "y": 100}) 
    
    with patch('time.time', return_value=103.0):
        with pytest.raises(SafetyViolation, match="out of bounds"):
            safety_monitor.validate("move", {"x": 100, "y": 1080}) 

    with patch('time.time', return_value=104.0):
        with pytest.raises(SafetyViolation, match="out of bounds"):
            safety_monitor.validate("move", {"x": -1, "y": 100}) 

def test_banned_text(safety_monitor):
    with patch('time.time', return_value=200.0):
        assert safety_monitor.validate("type", {"text": "Hello World"}) is True
    
    with patch('time.time', return_value=201.0):
        with pytest.raises(SafetyViolation, match="dangerous text"):
            safety_monitor.validate("type", {"text": "rm -rf /"})
    
    with patch('time.time', return_value=202.0):
        with pytest.raises(SafetyViolation, match="dangerous text"):
            safety_monitor.validate("type", {"text": "format c:"})

    with patch('time.time', return_value=203.0):
        with pytest.raises(SafetyViolation, match="dangerous text"):
            safety_monitor.validate("type", {"text": "del /s *.*"})

def test_rate_limit(safety_monitor):
    # Init time
    start_time = 1000.0
    
    with patch('time.time', return_value=start_time):
        # First action: OK (diff > 0.1 since last_action_time=0)
        assert safety_monitor.validate("move", {"x": 10, "y": 10}) is True
    
    # Second action immediately: FAIL
    # time is still mocked as same or small increment? 
    # Must use side_effect or specific return values
    
    with patch('time.time', return_value=start_time + 0.05):
        # Diff is 0.05 < 0.1
        with pytest.raises(SafetyViolation, match="Rate limit exceeded"):
            safety_monitor.validate("move", {"x": 20, "y": 20})
    
    # Wait enough: OK
    with patch('time.time', return_value=start_time + 0.2):
        # Diff from last SUCCESSFUL action (at 1000.0) is 0.2 > 0.1
        assert safety_monitor.validate("move", {"x": 30, "y": 30}) is True
