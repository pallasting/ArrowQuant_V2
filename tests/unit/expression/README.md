# Expression Layer Test Infrastructure

This directory contains the test infrastructure for the Expression & Presentation Layer, including fixtures, mock backends, and utilities for testing expression components.

## Directory Structure

```
tests/unit/expression/
├── __init__.py              # Package exports
├── README.md                # This file
├── fixtures.py              # Test fixtures and sample data
├── mock_backends.py         # Mock implementations of backends
└── test_expression_types.py # Tests for core data structures
```

## Test Fixtures

The `fixtures.py` module provides sample data for testing:

### Sample Texts

```python
from tests.unit.expression import SAMPLE_TEXTS

# Available text samples
SAMPLE_TEXTS["short"]      # "Hello, how are you?"
SAMPLE_TEXTS["medium"]     # Medium-length text
SAMPLE_TEXTS["long"]       # Long-form text
SAMPLE_TEXTS["technical"]  # Technical content
SAMPLE_TEXTS["casual"]     # Casual style
SAMPLE_TEXTS["formal"]     # Formal style
SAMPLE_TEXTS["empathetic"] # Empathetic tone
SAMPLE_TEXTS["error"]      # Error message
```

### Sample Conversations

```python
from tests.unit.expression import SAMPLE_CONVERSATIONS

# Available conversation histories
SAMPLE_CONVERSATIONS["empty"]       # []
SAMPLE_CONVERSATIONS["single_turn"] # One exchange
SAMPLE_CONVERSATIONS["multi_turn"]  # Multiple exchanges
SAMPLE_CONVERSATIONS["technical"]   # Technical discussion
```

### Creating Test Contexts

```python
from tests.unit.expression import create_sample_context

# Create a basic context
ctx = create_sample_context(
    user_id="test_user",
    emotion="joy",
    formality=0.7,
    language="en"
)

# Create context with custom conversation history
ctx = create_sample_context(
    user_id="test_user",
    conversation_history=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
)
```

### Creating Test Response Plans

```python
from tests.unit.expression import create_sample_plan
from llm_compression.expression.expression_types import OutputModality, ExpressionStyle

# Create a basic plan
plan = create_sample_plan(
    modalities=[OutputModality.TEXT, OutputModality.SPEECH],
    style=ExpressionStyle.CASUAL,
    emotion="joy"
)
```

### Sample Voice Configurations

```python
from tests.unit.expression import SAMPLE_VOICES

# Available voice configurations
voice = SAMPLE_VOICES["en_default"]    # Default English voice
voice = SAMPLE_VOICES["en_fast"]       # Fast English voice
voice = SAMPLE_VOICES["zh_default"]    # Default Chinese voice
voice = SAMPLE_VOICES["emotional_joy"] # Joyful voice
```

### Generating Sample Audio

```python
from tests.unit.expression import generate_sample_audio

# Generate 1 second of audio at 22050 Hz
audio = generate_sample_audio(duration_seconds=1.0, sample_rate=22050)
```

### Test Cases

Pre-defined test cases for common scenarios:

```python
from tests.unit.expression import (
    EMOTION_TEST_CASES,
    LANGUAGE_TEST_CASES,
    STYLE_TEST_CASES
)

# Emotion test cases with expected voice parameters
for case in EMOTION_TEST_CASES:
    emotion = case["emotion"]
    expected_speed = case["expected_speed"]
    expected_pitch = case["expected_pitch"]
    # Test emotion mapping...

# Language detection test cases
for case in LANGUAGE_TEST_CASES:
    text = case["text"]
    expected_lang = case["expected_language"]
    # Test language detection...

# Style selection test cases
for case in STYLE_TEST_CASES:
    intent = case["intent"]
    formality = case["formality"]
    expected_style = case["expected_style"]
    # Test style selection...
```

## Mock Backends

The `mock_backends.py` module provides mock implementations for testing without real API calls or model files.

### MockNLGBackend

Mock Natural Language Generation backend:

```python
from tests.unit.expression import create_mock_nlg_backend

# Create mock backend
backend = create_mock_nlg_backend()

# Generate streaming response
for chunk in backend.generate_streaming("Hello", "You are helpful"):
    print(chunk)

# Generate complete response
response = backend.generate_complete("Hello", "You are helpful")

# Check call count
assert backend.call_count == 2

# Simulate failure
backend.should_fail = True
backend.failure_message = "API timeout"

# Reset state
backend.reset()
```

### MockTTSBackend

Mock Text-to-Speech backend:

```python
from tests.unit.expression import create_mock_tts_backend, SAMPLE_VOICES

# Create mock backend
backend = create_mock_tts_backend(sample_rate=22050)

# Synthesize speech
audio = backend.synthesize("Hello", SAMPLE_VOICES["en_default"])
assert audio.shape == (22050,)  # 1 second of audio

# Synthesize streaming
for chunk in backend.synthesize_streaming("Hello. How are you?", SAMPLE_VOICES["en_default"]):
    print(f"Chunk shape: {chunk.shape}")

# Check what was called
assert backend.last_text == "Hello. How are you?"
assert backend.call_count == 2

# Simulate failure
backend.should_fail = True

# Reset state
backend.reset()
```

### MockTemplateEngine

Mock template-based text generation:

```python
from tests.unit.expression import create_mock_template_engine
from llm_compression.expression.expression_types import ExpressionStyle

# Create mock engine
engine = create_mock_template_engine()

# Generate from template
response = engine.generate("Hello", ExpressionStyle.CASUAL)

# Get specific template
template = engine.get_template("greeting_template")
assert template == "Hello! How can I help you today?"

# Reset state
engine.reset()
```

### MockTTSCache

Mock TTS cache for testing caching behavior:

```python
from tests.unit.expression import create_mock_tts_cache, SAMPLE_VOICES
import numpy as np

# Create mock cache
cache = create_mock_tts_cache(max_size_mb=100)

# Cache audio
audio = np.zeros(22050, dtype=np.float32)
cache.put("Hello", SAMPLE_VOICES["en_default"], audio)

# Retrieve from cache
cached = cache.get("Hello", SAMPLE_VOICES["en_default"])
assert cached is not None
assert cache.hit_count == 1

# Check cache miss
cached = cache.get("Goodbye", SAMPLE_VOICES["en_default"])
assert cached is None
assert cache.miss_count == 1

# Check hit rate
assert cache.hit_rate == 0.5  # 1 hit, 1 miss

# Clear cache
cache.clear()
cache.reset_stats()
```

### MockLanguageDetector

Mock language detection:

```python
from tests.unit.expression import create_mock_language_detector

# Create mock detector
detector = create_mock_language_detector()

# Detect language
lang = detector.detect_language("Hello, how are you?")
assert lang == "en"

lang = detector.detect_language("你好")
assert lang == "zh"

# Override detection
detector.override_language = "es"
lang = detector.detect_language("Hello")
assert lang == "es"

# Reset state
detector.reset()
```

### MockEmotionMapper

Mock emotion-to-voice parameter mapping:

```python
from tests.unit.expression import create_mock_emotion_mapper, SAMPLE_VOICES

# Create mock mapper
mapper = create_mock_emotion_mapper()

# Apply emotion
voice = SAMPLE_VOICES["en_default"]
voice.emotion = "joy"
voice.emotion_intensity = 0.7

adjusted = mapper.apply_emotion(voice)
assert adjusted.speed > 1.0  # Joy increases speed
assert adjusted.pitch > 1.0  # Joy increases pitch

# Reset state
mapper.reset()
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from tests.unit.expression import (
    create_sample_context,
    create_mock_nlg_backend,
    SAMPLE_TEXTS
)

class TestMyComponent:
    """Test MyComponent functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        context = create_sample_context()
        backend = create_mock_nlg_backend()
        
        # Act
        result = my_function(context, backend)
        
        # Assert
        assert result is not None
        assert backend.call_count == 1
```

### Testing with Fixtures

```python
import pytest
from tests.unit.expression import SAMPLE_TEXTS, EMOTION_TEST_CASES

class TestEmotionMapping:
    """Test emotion mapping."""
    
    @pytest.mark.parametrize("case", EMOTION_TEST_CASES)
    def test_emotion_parameters(self, case):
        """Test emotion parameter mapping."""
        emotion = case["emotion"]
        expected_speed = case["expected_speed"]
        expected_pitch = case["expected_pitch"]
        
        # Test emotion mapping
        result = map_emotion(emotion, intensity=1.0)
        
        assert abs(result.speed - expected_speed) < 0.01
        assert abs(result.pitch - expected_pitch) < 0.01
```

### Testing Error Handling

```python
import pytest
from tests.unit.expression import create_mock_tts_backend

class TestErrorHandling:
    """Test error handling."""
    
    def test_tts_failure_fallback(self):
        """Test fallback when TTS fails."""
        # Arrange
        backend = create_mock_tts_backend()
        backend.should_fail = True
        backend.failure_message = "TTS service unavailable"
        
        # Act & Assert
        with pytest.raises(Exception, match="TTS service unavailable"):
            backend.synthesize("Hello", voice_config)
```

## Running Tests

```bash
# Run all expression tests
pytest tests/unit/expression/

# Run specific test file
pytest tests/unit/expression/test_expression_types.py

# Run with coverage
pytest --cov=llm_compression.expression tests/unit/expression/

# Run specific test
pytest tests/unit/expression/test_expression_types.py::TestVoiceConfig::test_speed_validation

# Run tests matching pattern
pytest -k "emotion" tests/unit/expression/
```

## Best Practices

1. **Use fixtures**: Import fixtures from `tests.unit.expression` rather than creating data inline
2. **Use mock backends**: Use mock backends to avoid API calls and model loading
3. **Reset mocks**: Always reset mock state between tests or use pytest fixtures
4. **Parametrize tests**: Use `@pytest.mark.parametrize` with test case lists
5. **Test error cases**: Test both success and failure scenarios
6. **Check side effects**: Verify mock call counts and parameters
7. **Use descriptive names**: Test names should clearly describe what is being tested

## Adding New Fixtures

To add new fixtures:

1. Add sample data to `fixtures.py`
2. Export from `__init__.py`
3. Document in this README
4. Add test cases if applicable

Example:

```python
# In fixtures.py
SAMPLE_NEW_DATA = {
    "key": "value"
}

# In __init__.py
from tests.unit.expression.fixtures import SAMPLE_NEW_DATA

__all__ = [
    # ... existing exports
    "SAMPLE_NEW_DATA",
]
```

## Adding New Mock Backends

To add new mock backends:

1. Create mock class in `mock_backends.py`
2. Add factory function
3. Export from `__init__.py`
4. Document in this README

Example:

```python
# In mock_backends.py
class MockNewBackend:
    def __init__(self):
        self.call_count = 0
    
    def reset(self):
        self.call_count = 0

def create_mock_new_backend() -> MockNewBackend:
    return MockNewBackend()

# In __init__.py
from tests.unit.expression.mock_backends import (
    MockNewBackend,
    create_mock_new_backend,
)

__all__ = [
    # ... existing exports
    "MockNewBackend",
    "create_mock_new_backend",
]
```

## Requirements Validation

This test infrastructure validates **Requirement 13.1**:
- Provides unit test fixtures for all core components
- Provides mock backends to avoid external dependencies
- Enables comprehensive testing without API keys or model files
- Supports testing of error scenarios and edge cases
