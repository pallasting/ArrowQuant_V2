"""
Unit tests for VoiceManager.

Tests voice preset management, user preferences, voice selection,
and persistence functionality.

Requirements: 7.1, 7.4
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from llm_compression.expression.tts.voice_manager import (
    VoiceManager,
    VoicePreset,
    UserVoicePreferences
)
from llm_compression.expression.expression_types import TTSBackend, VoiceConfig


class TestVoicePreset:
    """Test VoicePreset dataclass."""
    
    def test_create_preset(self):
        """Test creating a voice preset."""
        preset = VoicePreset(
            id="test_voice",
            name="Test Voice",
            backend=TTSBackend.PIPER,
            voice_id="test_id",
            language="en",
            description="Test voice preset"
        )
        
        assert preset.id == "test_voice"
        assert preset.name == "Test Voice"
        assert preset.backend == TTSBackend.PIPER
        assert preset.voice_id == "test_id"
        assert preset.language == "en"
        assert preset.tags == []
    
    def test_preset_with_tags(self):
        """Test preset with tags."""
        preset = VoicePreset(
            id="test_voice",
            name="Test Voice",
            backend=TTSBackend.PIPER,
            voice_id="test_id",
            tags=["english", "neutral", "fast"]
        )
        
        assert len(preset.tags) == 3
        assert "english" in preset.tags
    
    def test_to_voice_config(self):
        """Test converting preset to VoiceConfig."""
        preset = VoicePreset(
            id="test_voice",
            name="Test Voice",
            backend=TTSBackend.PIPER,
            voice_id="test_id",
            language="en",
            accent="en-US"
        )
        
        voice_config = preset.to_voice_config()
        
        assert isinstance(voice_config, VoiceConfig)
        assert voice_config.voice_id == "test_id"
        assert voice_config.language == "en"
        assert voice_config.accent == "en-US"


class TestUserVoicePreferences:
    """Test UserVoicePreferences dataclass."""
    
    def test_create_preferences(self):
        """Test creating user preferences."""
        prefs = UserVoicePreferences(
            user_id="user123",
            default_voice_id="piper_en_us",
            preferred_language="en"
        )
        
        assert prefs.user_id == "user123"
        assert prefs.default_voice_id == "piper_en_us"
        assert prefs.preferred_language == "en"
        assert prefs.voice_history == []
    
    def test_preferences_with_history(self):
        """Test preferences with voice history."""
        prefs = UserVoicePreferences(
            user_id="user123",
            voice_history=["voice1", "voice2", "voice3"]
        )
        
        assert len(prefs.voice_history) == 3
        assert "voice1" in prefs.voice_history


class TestVoiceManager:
    """Test VoiceManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for preferences."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def voice_manager(self, temp_dir):
        """Create VoiceManager with temporary directory."""
        return VoiceManager(preferences_dir=temp_dir)
    
    def test_initialization(self, voice_manager):
        """Test VoiceManager initialization."""
        assert voice_manager is not None
        assert len(voice_manager.presets) > 0
        assert voice_manager.preferences_dir.exists()
    
    def test_default_presets_loaded(self, voice_manager):
        """Test that default presets are loaded."""
        # Should have presets for different backends
        piper_presets = [
            p for p in voice_manager.presets.values()
            if p.backend == TTSBackend.PIPER
        ]
        assert len(piper_presets) > 0
        
        # Check specific preset exists
        preset = voice_manager.get_preset("piper_en_us_lessac")
        assert preset is not None
        assert preset.language == "en"
    
    def test_add_preset(self, voice_manager):
        """Test adding a custom preset."""
        custom_preset = VoicePreset(
            id="custom_voice",
            name="Custom Voice",
            backend=TTSBackend.PIPER,
            voice_id="custom_id",
            language="fr",
            tags=["french", "custom"]
        )
        
        voice_manager.add_preset(custom_preset)
        
        retrieved = voice_manager.get_preset("custom_voice")
        assert retrieved is not None
        assert retrieved.id == "custom_voice"
        assert retrieved.language == "fr"
    
    def test_get_preset(self, voice_manager):
        """Test getting preset by ID."""
        preset = voice_manager.get_preset("piper_en_us_lessac")
        
        assert preset is not None
        assert preset.id == "piper_en_us_lessac"
        assert preset.backend == TTSBackend.PIPER
    
    def test_get_nonexistent_preset(self, voice_manager):
        """Test getting nonexistent preset."""
        preset = voice_manager.get_preset("nonexistent")
        assert preset is None
    
    def test_list_all_presets(self, voice_manager):
        """Test listing all presets."""
        presets = voice_manager.list_presets()
        assert len(presets) > 0
    
    def test_list_presets_by_language(self, voice_manager):
        """Test filtering presets by language."""
        en_presets = voice_manager.list_presets(language="en")
        assert len(en_presets) > 0
        assert all(p.language == "en" for p in en_presets)
        
        zh_presets = voice_manager.list_presets(language="zh")
        assert len(zh_presets) > 0
        assert all(p.language == "zh" for p in zh_presets)
    
    def test_list_presets_by_backend(self, voice_manager):
        """Test filtering presets by backend."""
        piper_presets = voice_manager.list_presets(backend=TTSBackend.PIPER)
        assert len(piper_presets) > 0
        assert all(p.backend == TTSBackend.PIPER for p in piper_presets)
        
        azure_presets = voice_manager.list_presets(backend=TTSBackend.AZURE)
        assert len(azure_presets) > 0
        assert all(p.backend == TTSBackend.AZURE for p in azure_presets)
    
    def test_list_presets_by_tags(self, voice_manager):
        """Test filtering presets by tags."""
        fast_presets = voice_manager.list_presets(tags=["fast"])
        assert len(fast_presets) > 0
        assert all("fast" in p.tags for p in fast_presets)
        
        emotional_presets = voice_manager.list_presets(tags=["emotional"])
        assert len(emotional_presets) > 0
        assert all("emotional" in p.tags for p in emotional_presets)
    
    def test_list_presets_multiple_filters(self, voice_manager):
        """Test filtering presets with multiple criteria."""
        presets = voice_manager.list_presets(
            language="en",
            backend=TTSBackend.PIPER,
            tags=["fast"]
        )
        
        assert len(presets) > 0
        for preset in presets:
            assert preset.language == "en"
            assert preset.backend == TTSBackend.PIPER
            assert "fast" in preset.tags
    
    def test_select_voice_default(self, voice_manager):
        """Test selecting voice with no criteria."""
        voice_config = voice_manager.select_voice()
        
        assert isinstance(voice_config, VoiceConfig)
        assert voice_config.voice_id is not None
    
    def test_select_voice_by_language(self, voice_manager):
        """Test selecting voice by language."""
        voice_config = voice_manager.select_voice(language="zh")
        
        assert voice_config.language == "zh"
    
    def test_select_voice_by_backend(self, voice_manager):
        """Test selecting voice by backend."""
        voice_config = voice_manager.select_voice(backend=TTSBackend.PIPER)
        
        # Should select a Piper voice
        # Verify by checking against known Piper voice IDs
        piper_presets = voice_manager.list_presets(backend=TTSBackend.PIPER)
        piper_voice_ids = [p.voice_id for p in piper_presets]
        assert voice_config.voice_id in piper_voice_ids
    
    def test_select_voice_by_tags(self, voice_manager):
        """Test selecting voice by tags."""
        voice_config = voice_manager.select_voice(tags=["emotional"])
        
        # Should select a voice with emotional tag
        # Find the preset to verify
        matching_presets = voice_manager.list_presets(tags=["emotional"])
        matching_voice_ids = [p.voice_id for p in matching_presets]
        assert voice_config.voice_id in matching_voice_ids
    
    def test_user_preferences_not_found(self, voice_manager):
        """Test getting preferences for nonexistent user."""
        prefs = voice_manager.get_user_preferences("nonexistent_user")
        assert prefs is None
    
    def test_set_user_preferences(self, voice_manager):
        """Test setting user preferences."""
        prefs = UserVoicePreferences(
            user_id="user123",
            default_voice_id="piper_en_us_lessac",
            preferred_language="en",
            custom_speed=1.2
        )
        
        voice_manager.set_user_preferences("user123", prefs)
        
        # Retrieve and verify
        retrieved = voice_manager.get_user_preferences("user123")
        assert retrieved is not None
        assert retrieved.user_id == "user123"
        assert retrieved.default_voice_id == "piper_en_us_lessac"
        assert retrieved.custom_speed == 1.2
    
    def test_user_preferences_persistence(self, voice_manager):
        """Test that user preferences persist to disk."""
        prefs = UserVoicePreferences(
            user_id="user456",
            default_voice_id="azure_en_us_jenny",
            preferred_language="en"
        )
        
        voice_manager.set_user_preferences("user456", prefs)
        
        # Create new manager with same directory
        new_manager = VoiceManager(preferences_dir=voice_manager.preferences_dir)
        
        # Should load from disk
        retrieved = new_manager.get_user_preferences("user456")
        assert retrieved is not None
        assert retrieved.user_id == "user456"
        assert retrieved.default_voice_id == "azure_en_us_jenny"
    
    def test_update_user_voice(self, voice_manager):
        """Test updating user's voice selection."""
        voice_manager.update_user_voice(
            user_id="user789",
            voice_id="piper_en_us_lessac",
            set_as_default=True
        )
        
        prefs = voice_manager.get_user_preferences("user789")
        assert prefs is not None
        assert prefs.default_voice_id == "piper_en_us_lessac"
        assert "piper_en_us_lessac" in prefs.voice_history
    
    def test_update_user_voice_history(self, voice_manager):
        """Test that voice history is maintained."""
        user_id = "user_history"
        
        # Update voice multiple times
        voices = ["voice1", "voice2", "voice3"]
        for voice_id in voices:
            voice_manager.update_user_voice(user_id, voice_id)
        
        prefs = voice_manager.get_user_preferences(user_id)
        assert len(prefs.voice_history) == 3
        assert all(v in prefs.voice_history for v in voices)
    
    def test_voice_history_limit(self, voice_manager):
        """Test that voice history is limited to 10 entries."""
        user_id = "user_limit"
        
        # Add 15 voices
        for i in range(15):
            voice_manager.update_user_voice(user_id, f"voice{i}")
        
        prefs = voice_manager.get_user_preferences(user_id)
        assert len(prefs.voice_history) == 10
        # Should keep the last 10
        assert "voice14" in prefs.voice_history
        assert "voice0" not in prefs.voice_history
    
    def test_select_voice_with_user_default(self, voice_manager):
        """Test selecting voice uses user's default."""
        # Set user preferences
        prefs = UserVoicePreferences(
            user_id="user_default",
            default_voice_id="azure_en_us_jenny",
            custom_speed=1.5,
            custom_pitch=1.2
        )
        voice_manager.set_user_preferences("user_default", prefs)
        
        # Select voice for user
        voice_config = voice_manager.select_voice(user_id="user_default")
        
        # Should use user's default voice
        assert voice_config.voice_id == "en-US-JennyNeural"
        # Should apply user's custom parameters
        assert voice_config.speed == 1.5
        assert voice_config.pitch == 1.2
    
    def test_select_voice_with_user_preferred_language(self, voice_manager):
        """Test selecting voice uses user's preferred language."""
        # Set user preferences with preferred language
        prefs = UserVoicePreferences(
            user_id="user_lang",
            preferred_language="zh"
        )
        voice_manager.set_user_preferences("user_lang", prefs)
        
        # Select voice without specifying language
        voice_config = voice_manager.select_voice(user_id="user_lang")
        
        # Should use user's preferred language
        assert voice_config.language == "zh"
    
    def test_get_voice_for_user(self, voice_manager):
        """Test convenience method for getting user's voice."""
        # Set user preferences
        prefs = UserVoicePreferences(
            user_id="user_convenience",
            default_voice_id="piper_en_us_lessac"
        )
        voice_manager.set_user_preferences("user_convenience", prefs)
        
        # Get voice for user
        voice_config = voice_manager.get_voice_for_user("user_convenience")
        
        assert isinstance(voice_config, VoiceConfig)
        assert voice_config.voice_id == "en_US-lessac-medium"
    
    def test_get_voice_for_user_with_language_override(self, voice_manager):
        """Test getting user's voice with language override."""
        # Set user preferences with English default
        prefs = UserVoicePreferences(
            user_id="user_override",
            preferred_language="en"
        )
        voice_manager.set_user_preferences("user_override", prefs)
        
        # Get voice with Spanish override
        voice_config = voice_manager.get_voice_for_user(
            "user_override",
            language="es"
        )
        
        # Should use Spanish voice
        assert voice_config.language == "es"
    
    def test_preferences_with_backend_enum(self, voice_manager):
        """Test preferences with backend enum serialization."""
        prefs = UserVoicePreferences(
            user_id="user_backend",
            preferred_backend=TTSBackend.AZURE
        )
        
        voice_manager.set_user_preferences("user_backend", prefs)
        
        # Retrieve and verify backend enum is preserved
        retrieved = voice_manager.get_user_preferences("user_backend")
        assert retrieved.preferred_backend == TTSBackend.AZURE
    
    def test_select_voice_with_preferred_backend(self, voice_manager):
        """Test selecting voice uses user's preferred backend."""
        prefs = UserVoicePreferences(
            user_id="user_backend_pref",
            preferred_backend=TTSBackend.OPENAI,
            preferred_language="en"
        )
        voice_manager.set_user_preferences("user_backend_pref", prefs)
        
        voice_config = voice_manager.select_voice(user_id="user_backend_pref")
        
        # Should select an OpenAI voice
        openai_presets = voice_manager.list_presets(backend=TTSBackend.OPENAI)
        openai_voice_ids = [p.voice_id for p in openai_presets]
        assert voice_config.voice_id in openai_voice_ids


class TestVoiceManagerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for preferences."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def voice_manager(self, temp_dir):
        """Create VoiceManager with temporary directory."""
        return VoiceManager(preferences_dir=temp_dir)
    
    def test_empty_preset_list(self, voice_manager):
        """Test behavior with no matching presets."""
        # Clear all presets
        voice_manager.presets.clear()
        
        # Should return empty list
        presets = voice_manager.list_presets(language="nonexistent")
        assert presets == []
        
        # Should still return a fallback voice config
        voice_config = voice_manager.select_voice()
        assert isinstance(voice_config, VoiceConfig)
    
    def test_invalid_user_preferences_file(self, voice_manager, temp_dir):
        """Test handling of corrupted preferences file."""
        # Create invalid JSON file
        prefs_file = temp_dir / "corrupt_user.json"
        with open(prefs_file, 'w') as f:
            f.write("invalid json {{{")
        
        # Should return None and not crash
        prefs = voice_manager.get_user_preferences("corrupt_user")
        assert prefs is None
    
    def test_voice_history_duplicates(self, voice_manager):
        """Test that voice history doesn't add duplicates."""
        user_id = "user_duplicates"
        
        # Add same voice multiple times
        for _ in range(3):
            voice_manager.update_user_voice(user_id, "same_voice")
        
        prefs = voice_manager.get_user_preferences(user_id)
        # Should only appear once
        assert prefs.voice_history.count("same_voice") == 1


class TestVoiceParameterAdjustment:
    """Test voice parameter adjustment functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for preferences."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def voice_manager(self, temp_dir):
        """Create VoiceManager with temporary directory."""
        return VoiceManager(preferences_dir=temp_dir)
    
    @pytest.fixture
    def base_voice_config(self):
        """Create a base voice configuration for testing."""
        return VoiceConfig(
            voice_id="test_voice",
            language="en",
            speed=1.0,
            pitch=1.0,
            volume=1.0
        )
    
    def test_adjust_speed(self, voice_manager, base_voice_config):
        """Test adjusting speech speed."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            speed=1.5
        )
        
        assert adjusted.speed == 1.5
        assert adjusted.pitch == 1.0  # Unchanged
        assert adjusted.volume == 1.0  # Unchanged
    
    def test_adjust_pitch(self, voice_manager, base_voice_config):
        """Test adjusting pitch."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            pitch=1.2
        )
        
        assert adjusted.speed == 1.0  # Unchanged
        assert adjusted.pitch == 1.2
        assert adjusted.volume == 1.0  # Unchanged
    
    def test_adjust_volume(self, voice_manager, base_voice_config):
        """Test adjusting volume."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            volume=0.8
        )
        
        assert adjusted.speed == 1.0  # Unchanged
        assert adjusted.pitch == 1.0  # Unchanged
        assert adjusted.volume == 0.8
    
    def test_adjust_multiple_parameters(self, voice_manager, base_voice_config):
        """Test adjusting multiple parameters at once."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            speed=1.3,
            pitch=1.1,
            volume=0.9
        )
        
        assert adjusted.speed == 1.3
        assert adjusted.pitch == 1.1
        assert adjusted.volume == 0.9
    
    def test_adjust_no_parameters(self, voice_manager, base_voice_config):
        """Test calling adjust with no parameters."""
        adjusted = voice_manager.adjust_voice_parameters(base_voice_config)
        
        # Should return unchanged
        assert adjusted.speed == 1.0
        assert adjusted.pitch == 1.0
        assert adjusted.volume == 1.0
    
    def test_speed_validation_min(self, voice_manager, base_voice_config):
        """Test speed validation at minimum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            speed=0.5
        )
        assert adjusted.speed == 0.5
    
    def test_speed_validation_max(self, voice_manager, base_voice_config):
        """Test speed validation at maximum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            speed=2.0
        )
        assert adjusted.speed == 2.0
    
    def test_speed_validation_below_min(self, voice_manager, base_voice_config):
        """Test speed validation rejects values below minimum."""
        with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                speed=0.3
            )
    
    def test_speed_validation_above_max(self, voice_manager, base_voice_config):
        """Test speed validation rejects values above maximum."""
        with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                speed=2.5
            )
    
    def test_pitch_validation_min(self, voice_manager, base_voice_config):
        """Test pitch validation at minimum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            pitch=0.5
        )
        assert adjusted.pitch == 0.5
    
    def test_pitch_validation_max(self, voice_manager, base_voice_config):
        """Test pitch validation at maximum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            pitch=2.0
        )
        assert adjusted.pitch == 2.0
    
    def test_pitch_validation_below_min(self, voice_manager, base_voice_config):
        """Test pitch validation rejects values below minimum."""
        with pytest.raises(ValueError, match="Pitch must be between 0.5 and 2.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                pitch=0.3
            )
    
    def test_pitch_validation_above_max(self, voice_manager, base_voice_config):
        """Test pitch validation rejects values above maximum."""
        with pytest.raises(ValueError, match="Pitch must be between 0.5 and 2.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                pitch=2.5
            )
    
    def test_volume_validation_min(self, voice_manager, base_voice_config):
        """Test volume validation at minimum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            volume=0.0
        )
        assert adjusted.volume == 0.0
    
    def test_volume_validation_max(self, voice_manager, base_voice_config):
        """Test volume validation at maximum boundary."""
        adjusted = voice_manager.adjust_voice_parameters(
            base_voice_config,
            volume=1.0
        )
        assert adjusted.volume == 1.0
    
    def test_volume_validation_below_min(self, voice_manager, base_voice_config):
        """Test volume validation rejects values below minimum."""
        with pytest.raises(ValueError, match="Volume must be between 0.0 and 1.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                volume=-0.1
            )
    
    def test_volume_validation_above_max(self, voice_manager, base_voice_config):
        """Test volume validation rejects values above maximum."""
        with pytest.raises(ValueError, match="Volume must be between 0.0 and 1.0"):
            voice_manager.adjust_voice_parameters(
                base_voice_config,
                volume=1.5
            )
    
    def test_preview_voice_parameters(self, voice_manager, base_voice_config):
        """Test previewing voice parameter changes."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            speed=1.3,
            pitch=1.1,
            volume=0.9
        )
        
        assert preview["current"]["speed"] == 1.0
        assert preview["current"]["pitch"] == 1.0
        assert preview["current"]["volume"] == 1.0
        
        assert preview["proposed"]["speed"] == 1.3
        assert preview["proposed"]["pitch"] == 1.1
        assert preview["proposed"]["volume"] == 0.9
        
        assert preview["valid"] is True
        assert len(preview["warnings"]) == 0
    
    def test_preview_with_invalid_speed(self, voice_manager, base_voice_config):
        """Test preview detects invalid speed."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            speed=3.0
        )
        
        assert preview["valid"] is False
        assert any("Speed" in w for w in preview["warnings"])
    
    def test_preview_with_invalid_pitch(self, voice_manager, base_voice_config):
        """Test preview detects invalid pitch."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            pitch=0.3
        )
        
        assert preview["valid"] is False
        assert any("Pitch" in w for w in preview["warnings"])
    
    def test_preview_with_invalid_volume(self, voice_manager, base_voice_config):
        """Test preview detects invalid volume."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            volume=1.5
        )
        
        assert preview["valid"] is False
        assert any("Volume" in w for w in preview["warnings"])
    
    def test_preview_naturalness_warning_slow_speed(
        self,
        voice_manager,
        base_voice_config
    ):
        """Test preview warns about unnaturally slow speed."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            speed=0.6
        )
        
        assert preview["valid"] is True
        assert any("unnaturally slow" in w for w in preview["warnings"])
    
    def test_preview_naturalness_warning_fast_speed(
        self,
        voice_manager,
        base_voice_config
    ):
        """Test preview warns about unnaturally fast speed."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            speed=1.6
        )
        
        assert preview["valid"] is True
        assert any("unnaturally fast" in w for w in preview["warnings"])
    
    def test_preview_naturalness_warning_low_pitch(
        self,
        voice_manager,
        base_voice_config
    ):
        """Test preview warns about unnaturally low pitch."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            pitch=0.6
        )
        
        assert preview["valid"] is True
        assert any("unnaturally low" in w for w in preview["warnings"])
    
    def test_preview_naturalness_warning_high_pitch(
        self,
        voice_manager,
        base_voice_config
    ):
        """Test preview warns about unnaturally high pitch."""
        preview = voice_manager.preview_voice_parameters(
            base_voice_config,
            pitch=1.4
        )
        
        assert preview["valid"] is True
        assert any("unnaturally high" in w for w in preview["warnings"])
    
    def test_preview_no_changes(self, voice_manager, base_voice_config):
        """Test preview with no parameter changes."""
        preview = voice_manager.preview_voice_parameters(base_voice_config)
        
        assert preview["current"] == preview["proposed"]
        assert preview["valid"] is True
    
    def test_update_user_voice_parameters(self, voice_manager):
        """Test updating user's custom voice parameters."""
        user_id = "test_user"
        
        voice_manager.update_user_voice_parameters(
            user_id,
            speed=1.3,
            pitch=1.1,
            volume=0.9
        )
        
        prefs = voice_manager.get_user_preferences(user_id)
        assert prefs.custom_speed == 1.3
        assert prefs.custom_pitch == 1.1
        assert prefs.custom_volume == 0.9
    
    def test_update_user_voice_parameters_partial(self, voice_manager):
        """Test updating only some user parameters."""
        user_id = "test_user_partial"
        
        # Set initial parameters
        voice_manager.update_user_voice_parameters(
            user_id,
            speed=1.2,
            pitch=1.0,
            volume=0.8
        )
        
        # Update only speed
        voice_manager.update_user_voice_parameters(
            user_id,
            speed=1.5
        )
        
        prefs = voice_manager.get_user_preferences(user_id)
        assert prefs.custom_speed == 1.5
        assert prefs.custom_pitch == 1.0  # Unchanged
        assert prefs.custom_volume == 0.8  # Unchanged
    
    def test_update_user_voice_parameters_invalid_speed(self, voice_manager):
        """Test updating user parameters with invalid speed."""
        with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
            voice_manager.update_user_voice_parameters(
                "test_user",
                speed=3.0
            )
    
    def test_update_user_voice_parameters_invalid_pitch(self, voice_manager):
        """Test updating user parameters with invalid pitch."""
        with pytest.raises(ValueError, match="Pitch must be between 0.5 and 2.0"):
            voice_manager.update_user_voice_parameters(
                "test_user",
                pitch=0.3
            )
    
    def test_update_user_voice_parameters_invalid_volume(self, voice_manager):
        """Test updating user parameters with invalid volume."""
        with pytest.raises(ValueError, match="Volume must be between 0.0 and 1.0"):
            voice_manager.update_user_voice_parameters(
                "test_user",
                volume=1.5
            )
    
    def test_user_parameters_persistence(self, voice_manager):
        """Test that user voice parameters persist to disk."""
        user_id = "persist_params_user"
        
        voice_manager.update_user_voice_parameters(
            user_id,
            speed=1.4,
            pitch=1.2,
            volume=0.85
        )
        
        # Create new manager with same directory
        new_manager = VoiceManager(preferences_dir=voice_manager.preferences_dir)
        
        # Should load from disk
        prefs = new_manager.get_user_preferences(user_id)
        assert prefs.custom_speed == 1.4
        assert prefs.custom_pitch == 1.2
        assert prefs.custom_volume == 0.85
    
    def test_select_voice_applies_user_parameters(self, voice_manager):
        """Test that select_voice applies user's custom parameters."""
        user_id = "custom_params_user"
        
        # Set user preferences with custom parameters
        voice_manager.update_user_voice_parameters(
            user_id,
            speed=1.3,
            pitch=1.1,
            volume=0.9
        )
        
        # Set a default voice
        prefs = voice_manager.get_user_preferences(user_id)
        prefs.default_voice_id = "piper_en_us_lessac"
        voice_manager.set_user_preferences(user_id, prefs)
        
        # Select voice for user
        voice_config = voice_manager.select_voice(user_id=user_id)
        
        # Should apply custom parameters
        assert voice_config.speed == 1.3
        assert voice_config.pitch == 1.1
        assert voice_config.volume == 0.9
