"""
Voice Manager for TTS voice presets and user preferences.

This module manages voice presets, user preferences, and voice selection
for the TTS system. It provides functionality to store predefined voices,
manage per-user settings, and select voices based on criteria.

Requirements: 7.1, 7.4
"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json

from llm_compression.expression.expression_types import VoiceConfig, TTSBackend

logger = logging.getLogger(__name__)


@dataclass
class VoicePreset:
    """
    Voice preset configuration.
    
    A preset defines a complete voice configuration that can be
    selected by ID or criteria.
    """
    id: str
    name: str
    backend: TTSBackend
    voice_id: str
    language: str = "en"
    accent: Optional[str] = None
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_voice_config(self) -> VoiceConfig:
        """Convert preset to VoiceConfig."""
        return VoiceConfig(
            voice_id=self.voice_id,
            language=self.language,
            accent=self.accent
        )


@dataclass
class UserVoicePreferences:
    """
    User-specific voice preferences.
    
    Stores per-user voice settings including default voice,
    language preferences, and custom voice parameters.
    """
    user_id: str
    default_voice_id: Optional[str] = None
    preferred_language: str = "en"
    preferred_backend: Optional[TTSBackend] = None
    custom_speed: float = 1.0
    custom_pitch: float = 1.0
    custom_volume: float = 1.0
    voice_history: List[str] = None
    
    def __post_init__(self):
        if self.voice_history is None:
            self.voice_history = []


class VoiceManager:
    """
    Manages voice presets and user preferences.
    
    Responsibilities:
    - Store and retrieve voice presets
    - Manage user voice preferences
    - Select voices by ID or criteria
    - Load/save preferences to disk
    
    Requirements: 7.1, 7.4
    """
    
    def __init__(self, preferences_dir: Optional[Path] = None):
        """
        Initialize voice manager.
        
        Args:
            preferences_dir: Directory for storing user preferences
                           (defaults to ~/.ai-os/voice_preferences)
        """
        self.preferences_dir = preferences_dir or (
            Path.home() / ".ai-os" / "voice_preferences"
        )
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        # Voice presets storage
        self.presets: Dict[str, VoicePreset] = {}
        
        # User preferences storage
        self.user_preferences: Dict[str, UserVoicePreferences] = {}
        
        # Initialize default presets
        self._init_default_presets()
        
        logger.info(
            f"Initialized VoiceManager with {len(self.presets)} presets, "
            f"preferences_dir: {self.preferences_dir}"
        )
    
    def _init_default_presets(self):
        """Initialize default voice presets for different backends."""
        default_presets = [
            # Piper voices (local, fast)
            VoicePreset(
                id="piper_en_us_lessac",
                name="Lessac (US English)",
                backend=TTSBackend.PIPER,
                voice_id="en_US-lessac-medium",
                language="en",
                accent="en-US",
                description="Clear American English voice, medium quality",
                tags=["english", "american", "neutral", "fast"]
            ),
            VoicePreset(
                id="piper_en_gb_alan",
                name="Alan (British English)",
                backend=TTSBackend.PIPER,
                voice_id="en_GB-alan-medium",
                language="en",
                accent="en-GB",
                description="British English voice, medium quality",
                tags=["english", "british", "neutral", "fast"]
            ),
            VoicePreset(
                id="piper_es_carlfm",
                name="Carlfm (Spanish)",
                backend=TTSBackend.PIPER,
                voice_id="es_ES-carlfm-x_low",
                language="es",
                accent="es-ES",
                description="Spanish voice, low quality but fast",
                tags=["spanish", "neutral", "fast"]
            ),
            
            # Azure voices (cloud, high quality)
            VoicePreset(
                id="azure_en_us_jenny",
                name="Jenny (US English)",
                backend=TTSBackend.AZURE,
                voice_id="en-US-JennyNeural",
                language="en",
                accent="en-US",
                description="Natural American English voice with emotion support",
                tags=["english", "american", "natural", "emotional"]
            ),
            VoicePreset(
                id="azure_zh_cn_xiaoxiao",
                name="Xiaoxiao (Chinese)",
                backend=TTSBackend.AZURE,
                voice_id="zh-CN-XiaoxiaoNeural",
                language="zh",
                accent="zh-CN",
                description="Natural Chinese voice with emotion support",
                tags=["chinese", "mandarin", "natural", "emotional"]
            ),
            VoicePreset(
                id="azure_ja_jp_nanami",
                name="Nanami (Japanese)",
                backend=TTSBackend.AZURE,
                voice_id="ja-JP-NanamiNeural",
                language="ja",
                accent="ja-JP",
                description="Natural Japanese voice with emotion support",
                tags=["japanese", "natural", "emotional"]
            ),
            
            # OpenAI voices (cloud, very natural)
            VoicePreset(
                id="openai_alloy",
                name="Alloy",
                backend=TTSBackend.OPENAI,
                voice_id="alloy",
                language="en",
                description="Neutral, balanced voice",
                tags=["english", "neutral", "natural"]
            ),
            VoicePreset(
                id="openai_nova",
                name="Nova",
                backend=TTSBackend.OPENAI,
                voice_id="nova",
                language="en",
                description="Warm, friendly voice",
                tags=["english", "friendly", "natural"]
            ),
        ]
        
        for preset in default_presets:
            self.presets[preset.id] = preset
        
        logger.info(f"Initialized {len(default_presets)} default voice presets")
    
    def add_preset(self, preset: VoicePreset):
        """
        Add a voice preset.
        
        Args:
            preset: Voice preset to add
        """
        self.presets[preset.id] = preset
        logger.info(f"Added voice preset: {preset.id} ({preset.name})")
    
    def get_preset(self, preset_id: str) -> Optional[VoicePreset]:
        """
        Get voice preset by ID.
        
        Args:
            preset_id: Preset identifier
            
        Returns:
            Voice preset or None if not found
        """
        preset = self.presets.get(preset_id)
        if preset:
            logger.debug(f"Retrieved preset: {preset_id}")
        else:
            logger.warning(f"Preset not found: {preset_id}")
        return preset
    
    def list_presets(
        self,
        language: Optional[str] = None,
        backend: Optional[TTSBackend] = None,
        tags: Optional[List[str]] = None
    ) -> List[VoicePreset]:
        """
        List voice presets with optional filtering.
        
        Args:
            language: Filter by language (e.g., "en", "zh", "ja")
            backend: Filter by TTS backend
            tags: Filter by tags (any match)
            
        Returns:
            List of matching voice presets
        """
        results = list(self.presets.values())
        
        # Apply filters
        if language:
            results = [p for p in results if p.language == language]
        
        if backend:
            results = [p for p in results if p.backend == backend]
        
        if tags:
            results = [
                p for p in results
                if any(tag in p.tags for tag in tags)
            ]
        
        logger.debug(
            f"Listed {len(results)} presets "
            f"(language={language}, backend={backend}, tags={tags})"
        )
        
        return results
    
    def select_voice(
        self,
        user_id: Optional[str] = None,
        language: Optional[str] = None,
        backend: Optional[TTSBackend] = None,
        tags: Optional[List[str]] = None
    ) -> VoiceConfig:
        """
        Select voice based on criteria.
        
        Selection priority:
        1. User's default voice (if user_id provided)
        2. Matching preset by criteria
        3. Default voice for language
        4. System default voice
        
        Args:
            user_id: User identifier for preference lookup
            language: Preferred language
            backend: Preferred backend
            tags: Preferred tags
            
        Returns:
            Selected voice configuration
        """
        # Try user's default voice
        if user_id:
            prefs = self.get_user_preferences(user_id)
            if prefs and prefs.default_voice_id:
                preset = self.get_preset(prefs.default_voice_id)
                if preset:
                    logger.info(
                        f"Selected user's default voice: {preset.id} "
                        f"for user {user_id}"
                    )
                    voice_config = preset.to_voice_config()
                    
                    # Apply user's custom parameters
                    voice_config.speed = prefs.custom_speed
                    voice_config.pitch = prefs.custom_pitch
                    voice_config.volume = prefs.custom_volume
                    
                    return voice_config
            
            # Use user's preferred language if not specified
            if not language and prefs:
                language = prefs.preferred_language
            
            # Use user's preferred backend if not specified
            if not backend and prefs and prefs.preferred_backend:
                backend = prefs.preferred_backend
        
        # Find matching presets
        candidates = self.list_presets(
            language=language,
            backend=backend,
            tags=tags
        )
        
        if candidates:
            # Select first matching preset
            preset = candidates[0]
            logger.info(f"Selected voice by criteria: {preset.id}")
            return preset.to_voice_config()
        
        # Fallback to default voice for language
        if language:
            default_for_lang = self.list_presets(language=language)
            if default_for_lang:
                preset = default_for_lang[0]
                logger.info(f"Selected default voice for language {language}: {preset.id}")
                return preset.to_voice_config()
        
        # System default (first preset)
        if self.presets:
            preset = next(iter(self.presets.values()))
            logger.info(f"Selected system default voice: {preset.id}")
            return preset.to_voice_config()
        
        # Ultimate fallback
        logger.warning("No presets available, using fallback voice config")
        return VoiceConfig(voice_id="default", language="en")
    
    def get_user_preferences(
        self,
        user_id: str
    ) -> Optional[UserVoicePreferences]:
        """
        Get user voice preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences or None if not found
        """
        # Check in-memory cache
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        
        # Try to load from disk
        prefs = self._load_user_preferences(user_id)
        if prefs:
            self.user_preferences[user_id] = prefs
        
        return prefs
    
    def set_user_preferences(
        self,
        user_id: str,
        preferences: UserVoicePreferences
    ):
        """
        Set user voice preferences.
        
        Args:
            user_id: User identifier
            preferences: User preferences to save
        """
        # Ensure user_id matches
        preferences.user_id = user_id
        
        # Update in-memory cache
        self.user_preferences[user_id] = preferences
        
        # Save to disk
        self._save_user_preferences(preferences)
        
        logger.info(f"Updated preferences for user: {user_id}")
    
    def update_user_voice(
        self,
        user_id: str,
        voice_id: str,
        set_as_default: bool = False
    ):
        """
        Update user's voice selection.
        
        Args:
            user_id: User identifier
            voice_id: Voice preset ID
            set_as_default: Whether to set as default voice
        """
        # Get or create preferences
        prefs = self.get_user_preferences(user_id)
        if not prefs:
            prefs = UserVoicePreferences(user_id=user_id)
        
        # Update voice history
        if voice_id not in prefs.voice_history:
            prefs.voice_history.append(voice_id)
            # Keep only last 10 voices
            if len(prefs.voice_history) > 10:
                prefs.voice_history.pop(0)
        
        # Set as default if requested
        if set_as_default:
            prefs.default_voice_id = voice_id
            logger.info(f"Set default voice for user {user_id}: {voice_id}")
        
        # Save preferences
        self.set_user_preferences(user_id, prefs)
    
    def _load_user_preferences(
        self,
        user_id: str
    ) -> Optional[UserVoicePreferences]:
        """
        Load user preferences from disk.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences or None if not found
        """
        prefs_file = self.preferences_dir / f"{user_id}.json"
        
        if not prefs_file.exists():
            return None
        
        try:
            with open(prefs_file, 'r') as f:
                data = json.load(f)
            
            # Convert backend string to enum if present
            if 'preferred_backend' in data and data['preferred_backend']:
                data['preferred_backend'] = TTSBackend(data['preferred_backend'])
            
            prefs = UserVoicePreferences(**data)
            logger.debug(f"Loaded preferences for user: {user_id}")
            return prefs
        
        except Exception as e:
            logger.error(f"Failed to load preferences for user {user_id}: {e}")
            return None
    
    def _save_user_preferences(self, preferences: UserVoicePreferences):
        """
        Save user preferences to disk.
        
        Args:
            preferences: User preferences to save
        """
        prefs_file = self.preferences_dir / f"{preferences.user_id}.json"
        
        try:
            # Convert to dict
            data = asdict(preferences)
            
            # Convert backend enum to string
            if data.get('preferred_backend'):
                data['preferred_backend'] = data['preferred_backend'].value
            
            # Save to file
            with open(prefs_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved preferences for user: {preferences.user_id}")
        
        except Exception as e:
            logger.error(
                f"Failed to save preferences for user {preferences.user_id}: {e}"
            )
    
    def get_voice_for_user(
        self,
        user_id: str,
        language: Optional[str] = None
    ) -> VoiceConfig:
        """
        Get voice configuration for user.
        
        Convenience method that combines preference lookup and voice selection.
        
        Args:
            user_id: User identifier
            language: Override language (uses user's preferred if not specified)
            
        Returns:
            Voice configuration for user
        """
        return self.select_voice(
            user_id=user_id,
            language=language
        )
    
    def adjust_voice_parameters(
        self,
        voice_config: VoiceConfig,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None
    ) -> VoiceConfig:
        """
        Adjust voice parameters with validation.
        
        Validates parameter ranges to ensure natural-sounding output:
        - Speed: 0.5-2.0 (0.5=half speed, 2.0=double speed)
        - Pitch: 0.5-2.0 (0.5=lower, 2.0=higher)
        - Volume: 0.0-1.0 (0.0=silent, 1.0=full volume)
        
        Args:
            voice_config: Voice configuration to adjust
            speed: Speech speed multiplier (0.5-2.0)
            pitch: Pitch multiplier (0.5-2.0)
            volume: Volume level (0.0-1.0)
            
        Returns:
            Adjusted voice configuration
            
        Raises:
            ValueError: If parameters are out of acceptable range
        
        Requirements: 7.3, 7.7
        """
        # Validate speed
        if speed is not None:
            if not 0.5 <= speed <= 2.0:
                raise ValueError(
                    f"Speed must be between 0.5 and 2.0, got {speed}"
                )
            voice_config.speed = speed
            logger.debug(f"Adjusted speed to {speed}")
        
        # Validate pitch
        if pitch is not None:
            if not 0.5 <= pitch <= 2.0:
                raise ValueError(
                    f"Pitch must be between 0.5 and 2.0, got {pitch}"
                )
            voice_config.pitch = pitch
            logger.debug(f"Adjusted pitch to {pitch}")
        
        # Validate volume
        if volume is not None:
            if not 0.0 <= volume <= 1.0:
                raise ValueError(
                    f"Volume must be between 0.0 and 1.0, got {volume}"
                )
            voice_config.volume = volume
            logger.debug(f"Adjusted volume to {volume}")
        
        return voice_config
    
    def preview_voice_parameters(
        self,
        voice_config: VoiceConfig,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Preview voice parameter changes without applying them.
        
        Returns a preview of what the voice would sound like with
        the specified parameters, including validation results.
        
        Args:
            voice_config: Current voice configuration
            speed: Proposed speech speed (0.5-2.0)
            pitch: Proposed pitch (0.5-2.0)
            volume: Proposed volume (0.0-1.0)
            
        Returns:
            Dictionary with preview information:
            - current: Current parameter values
            - proposed: Proposed parameter values
            - valid: Whether proposed values are valid
            - warnings: List of warnings about naturalness
        
        Requirements: 7.6, 7.7
        """
        preview = {
            "current": {
                "speed": voice_config.speed,
                "pitch": voice_config.pitch,
                "volume": voice_config.volume
            },
            "proposed": {
                "speed": speed if speed is not None else voice_config.speed,
                "pitch": pitch if pitch is not None else voice_config.pitch,
                "volume": volume if volume is not None else voice_config.volume
            },
            "valid": True,
            "warnings": []
        }
        
        # Validate proposed values
        try:
            if speed is not None and not 0.5 <= speed <= 2.0:
                preview["valid"] = False
                preview["warnings"].append(
                    f"Speed {speed} is out of range (0.5-2.0)"
                )
            
            if pitch is not None and not 0.5 <= pitch <= 2.0:
                preview["valid"] = False
                preview["warnings"].append(
                    f"Pitch {pitch} is out of range (0.5-2.0)"
                )
            
            if volume is not None and not 0.0 <= volume <= 1.0:
                preview["valid"] = False
                preview["warnings"].append(
                    f"Volume {volume} is out of range (0.0-1.0)"
                )
            
            # Add naturalness warnings for extreme values
            if speed is not None:
                if speed < 0.7:
                    preview["warnings"].append(
                        "Speed below 0.7 may sound unnaturally slow"
                    )
                elif speed > 1.5:
                    preview["warnings"].append(
                        "Speed above 1.5 may sound unnaturally fast"
                    )
            
            if pitch is not None:
                if pitch < 0.7:
                    preview["warnings"].append(
                        "Pitch below 0.7 may sound unnaturally low"
                    )
                elif pitch > 1.3:
                    preview["warnings"].append(
                        "Pitch above 1.3 may sound unnaturally high"
                    )
        
        except Exception as e:
            preview["valid"] = False
            preview["warnings"].append(f"Validation error: {e}")
        
        logger.debug(
            f"Preview: valid={preview['valid']}, "
            f"warnings={len(preview['warnings'])}"
        )
        
        return preview
    
    def update_user_voice_parameters(
        self,
        user_id: str,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None
    ):
        """
        Update user's custom voice parameters.
        
        Args:
            user_id: User identifier
            speed: Custom speech speed (0.5-2.0)
            pitch: Custom pitch (0.5-2.0)
            volume: Custom volume (0.0-1.0)
            
        Raises:
            ValueError: If parameters are out of acceptable range
        
        Requirements: 7.3, 7.4
        """
        # Get or create preferences
        prefs = self.get_user_preferences(user_id)
        if not prefs:
            prefs = UserVoicePreferences(user_id=user_id)
        
        # Validate and update parameters
        if speed is not None:
            if not 0.5 <= speed <= 2.0:
                raise ValueError(
                    f"Speed must be between 0.5 and 2.0, got {speed}"
                )
            prefs.custom_speed = speed
        
        if pitch is not None:
            if not 0.5 <= pitch <= 2.0:
                raise ValueError(
                    f"Pitch must be between 0.5 and 2.0, got {pitch}"
                )
            prefs.custom_pitch = pitch
        
        if volume is not None:
            if not 0.0 <= volume <= 1.0:
                raise ValueError(
                    f"Volume must be between 0.0 and 1.0, got {volume}"
                )
            prefs.custom_volume = volume
        
        # Save preferences
        self.set_user_preferences(user_id, prefs)
        
        logger.info(
            f"Updated voice parameters for user {user_id}: "
            f"speed={prefs.custom_speed}, pitch={prefs.custom_pitch}, "
            f"volume={prefs.custom_volume}"
        )
