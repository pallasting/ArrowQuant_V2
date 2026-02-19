import sys
from unittest.mock import MagicMock

# CRITICAL: Mock dependencies BEFORE importing the module under test

# Mock pyautogui
mock_pyautogui = MagicMock()
mock_pyautogui.FAILSAFE = True
mock_pyautogui.PAUSE = 0.5
mock_pyautogui.size.return_value = (1920, 1080)
sys.modules["pyautogui"] = mock_pyautogui

# Mock cv2
mock_cv2 = MagicMock()
sys.modules["cv2"] = mock_cv2

# Mock numpy
# We must ensure numpy acts like a package to support sub-imports if any
mock_numpy = MagicMock()
mock_numpy.__path__ = [] # This makes it a package
sys.modules["numpy"] = mock_numpy

# Also mock numpy.core if something strictly imports it 
# (though usually MagicMock handles attributes, imports are different)
sys.modules["numpy.core"] = MagicMock()

# Now import the module
from llm_compression.action.manager import ActionManager

import pytest
import json

@pytest.fixture
def action_manager_instance(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    manager = ActionManager(str(workspace))
    
    # Mock safety
    if manager.safety:
        manager.safety.validate = MagicMock(return_value=True)
    
    return manager

def test_initialization(action_manager_instance):
    assert action_manager_instance.screen_size == (1920, 1080)
    from llm_compression.action.manager import PYAUTOGUI_AVAILABLE
    assert PYAUTOGUI_AVAILABLE is True

def test_execute_move(action_manager_instance):
    action_manager_instance.execute("move", x=100, y=200)
    mock_pyautogui.moveTo.assert_called()
    args, _ = mock_pyautogui.moveTo.call_args
    assert args[0] == 100
    assert args[1] == 200

def test_execute_click(action_manager_instance):
    action_manager_instance.execute("click", x=50, y=50, button="right")
    mock_pyautogui.click.assert_called_with(50, 50, clicks=1, button="right")

def test_execute_type(action_manager_instance):
    action_manager_instance.execute("type", text="hello")
    mock_pyautogui.write.assert_called_with("hello", interval=0.05)

def test_safety_check_fails(action_manager_instance):
    action_manager_instance.safety.validate.side_effect = Exception("Unsafe!")
    result = action_manager_instance.execute("move", x=0, y=0)
    assert result is False

def test_find_element_found(action_manager_instance):
    # Setup mocks for vision
    mock_pyautogui.screenshot.return_value = MagicMock()
    mock_numpy.array.return_value = MagicMock()
    mock_cv2.cvtColor.return_value = "screen_bgr"
    
    mock_template = MagicMock()
    mock_template.shape = (50, 100, 3) 
    mock_cv2.imread.return_value = mock_template
    
    # match > 0.8
    mock_cv2.minMaxLoc.return_value = (0, 0.9, (0,0), (200, 200))
    
    from llm_compression.action.manager import VISION_AVAILABLE
    assert VISION_AVAILABLE is True
    
    result = action_manager_instance.find_element("template.png")
    
    assert result is not None
    # Center = 200 + 100//2, 200 + 50//2 = 250, 225
    assert result == (250, 225)

def test_find_element_not_found(action_manager_instance):
    mock_pyautogui.screenshot.return_value = MagicMock()
    mock_cv2.imread.return_value = MagicMock()
    mock_cv2.minMaxLoc.return_value = (0, 0.5, (0,0), (0,0))
    
    result = action_manager_instance.find_element("template.png")
    assert result is None

def test_log_experience(action_manager_instance):
    action_manager_instance.execute("move", x=10, y=10)
    log_file = action_manager_instance.log_path
    assert log_file.exists()
    
    with open(log_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)
        assert data["action"] == "move"
