"""
Voice Manager Demo

Demonstrates voice preset management, user preferences, and voice selection
functionality of the VoiceManager class.

Requirements: 7.1, 7.4
"""

import logging
from pathlib import Path

from llm_compression.expression.tts.voice_manager import (
    VoiceManager,
    VoicePreset,
    UserVoicePreferences
)
from llm_compression.expression.expression_types import TTSBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic VoiceManager usage."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Voice Manager Usage")
    print("="*60)
    
    # Create voice manager
    voice_manager = VoiceManager()
    
    # List all available presets
    print("\n1. Available Voice Presets:")
    all_presets = voice_manager.list_presets()
    for preset in all_presets[:5]:  # Show first 5
        print(f"   - {preset.id}: {preset.name} ({preset.language}, {preset.backend.value})")
    print(f"   ... and {len(all_presets) - 5} more")
    
    # Get a specific preset
    print("\n2. Get Specific Preset:")
    preset = voice_manager.get_preset("piper_en_us_lessac")
    if preset:
        print(f"   ID: {preset.id}")
        print(f"   Name: {preset.name}")
        print(f"   Backend: {preset.backend.value}")
        print(f"   Voice ID: {preset.voice_id}")
        print(f"   Language: {preset.language}")
        print(f"   Description: {preset.description}")
        print(f"   Tags: {', '.join(preset.tags)}")
    
    # Convert to voice config
    print("\n3. Convert to VoiceConfig:")
    voice_config = preset.to_voice_config()
    print(f"   Voice ID: {voice_config.voice_id}")
    print(f"   Language: {voice_config.language}")
    print(f"   Speed: {voice_config.speed}")
    print(f"   Pitch: {voice_config.pitch}")


def demo_filtering():
    """Demonstrate filtering presets by criteria."""
    print("\n" + "="*60)
    print("DEMO 2: Filtering Voice Presets")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Filter by language
    print("\n1. English Voices:")
    en_presets = voice_manager.list_presets(language="en")
    for preset in en_presets[:3]:
        print(f"   - {preset.name} ({preset.backend.value})")
    
    print("\n2. Chinese Voices:")
    zh_presets = voice_manager.list_presets(language="zh")
    for preset in zh_presets:
        print(f"   - {preset.name} ({preset.backend.value})")
    
    # Filter by backend
    print("\n3. Piper Voices (Local, Fast):")
    piper_presets = voice_manager.list_presets(backend=TTSBackend.PIPER)
    for preset in piper_presets:
        print(f"   - {preset.name} ({preset.language})")
    
    print("\n4. Azure Voices (Cloud, High Quality):")
    azure_presets = voice_manager.list_presets(backend=TTSBackend.AZURE)
    for preset in azure_presets:
        print(f"   - {preset.name} ({preset.language})")
    
    # Filter by tags
    print("\n5. Fast Voices:")
    fast_presets = voice_manager.list_presets(tags=["fast"])
    for preset in fast_presets:
        print(f"   - {preset.name} ({preset.language}, {preset.backend.value})")
    
    print("\n6. Emotional Voices:")
    emotional_presets = voice_manager.list_presets(tags=["emotional"])
    for preset in emotional_presets:
        print(f"   - {preset.name} ({preset.language}, {preset.backend.value})")
    
    # Multiple filters
    print("\n7. English + Piper + Fast:")
    filtered = voice_manager.list_presets(
        language="en",
        backend=TTSBackend.PIPER,
        tags=["fast"]
    )
    for preset in filtered:
        print(f"   - {preset.name}")


def demo_voice_selection():
    """Demonstrate voice selection by criteria."""
    print("\n" + "="*60)
    print("DEMO 3: Voice Selection")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Select default voice
    print("\n1. Default Voice Selection:")
    voice_config = voice_manager.select_voice()
    print(f"   Selected: {voice_config.voice_id} ({voice_config.language})")
    
    # Select by language
    print("\n2. Select Chinese Voice:")
    voice_config = voice_manager.select_voice(language="zh")
    print(f"   Selected: {voice_config.voice_id} ({voice_config.language})")
    
    # Select by backend
    print("\n3. Select OpenAI Voice:")
    voice_config = voice_manager.select_voice(backend=TTSBackend.OPENAI)
    print(f"   Selected: {voice_config.voice_id}")
    
    # Select by tags
    print("\n4. Select Emotional Voice:")
    voice_config = voice_manager.select_voice(tags=["emotional"])
    print(f"   Selected: {voice_config.voice_id} ({voice_config.language})")
    
    # Select with multiple criteria
    print("\n5. Select English + Fast Voice:")
    voice_config = voice_manager.select_voice(
        language="en",
        tags=["fast"]
    )
    print(f"   Selected: {voice_config.voice_id} ({voice_config.language})")


def demo_user_preferences():
    """Demonstrate user preference management."""
    print("\n" + "="*60)
    print("DEMO 4: User Preferences")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Create user preferences
    print("\n1. Create User Preferences:")
    user_id = "alice"
    prefs = UserVoicePreferences(
        user_id=user_id,
        default_voice_id="azure_en_us_jenny",
        preferred_language="en",
        custom_speed=1.2,
        custom_pitch=1.1,
        custom_volume=0.9
    )
    voice_manager.set_user_preferences(user_id, prefs)
    print(f"   Created preferences for user: {user_id}")
    print(f"   Default voice: {prefs.default_voice_id}")
    print(f"   Custom speed: {prefs.custom_speed}")
    print(f"   Custom pitch: {prefs.custom_pitch}")
    
    # Retrieve preferences
    print("\n2. Retrieve User Preferences:")
    retrieved = voice_manager.get_user_preferences(user_id)
    print(f"   User: {retrieved.user_id}")
    print(f"   Default voice: {retrieved.default_voice_id}")
    print(f"   Preferred language: {retrieved.preferred_language}")
    
    # Select voice for user
    print("\n3. Select Voice for User:")
    voice_config = voice_manager.select_voice(user_id=user_id)
    print(f"   Selected: {voice_config.voice_id}")
    print(f"   Speed: {voice_config.speed} (custom)")
    print(f"   Pitch: {voice_config.pitch} (custom)")
    print(f"   Volume: {voice_config.volume} (custom)")
    
    # Update user voice
    print("\n4. Update User Voice:")
    voice_manager.update_user_voice(
        user_id=user_id,
        voice_id="piper_en_us_lessac",
        set_as_default=False
    )
    prefs = voice_manager.get_user_preferences(user_id)
    print(f"   Voice history: {prefs.voice_history}")
    
    # Set new default
    print("\n5. Set New Default Voice:")
    voice_manager.update_user_voice(
        user_id=user_id,
        voice_id="openai_nova",
        set_as_default=True
    )
    prefs = voice_manager.get_user_preferences(user_id)
    print(f"   New default: {prefs.default_voice_id}")
    print(f"   Voice history: {prefs.voice_history}")


def demo_multi_user():
    """Demonstrate multi-user preference management."""
    print("\n" + "="*60)
    print("DEMO 5: Multi-User Preferences")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Create preferences for multiple users
    users = [
        ("alice", "en", "azure_en_us_jenny", 1.2),
        ("bob", "zh", "azure_zh_cn_xiaoxiao", 1.0),
        ("charlie", "ja", "azure_ja_jp_nanami", 0.9),
    ]
    
    print("\n1. Create Preferences for Multiple Users:")
    for user_id, lang, voice_id, speed in users:
        prefs = UserVoicePreferences(
            user_id=user_id,
            default_voice_id=voice_id,
            preferred_language=lang,
            custom_speed=speed
        )
        voice_manager.set_user_preferences(user_id, prefs)
        print(f"   {user_id}: {voice_id} ({lang}, speed={speed})")
    
    # Select voices for each user
    print("\n2. Select Voices for Each User:")
    for user_id, _, _, _ in users:
        voice_config = voice_manager.get_voice_for_user(user_id)
        print(f"   {user_id}: {voice_config.voice_id} "
              f"({voice_config.language}, speed={voice_config.speed})")
    
    # Override language for a user
    print("\n3. Override Language for User:")
    voice_config = voice_manager.get_voice_for_user("alice", language="es")
    print(f"   alice (Spanish override): {voice_config.voice_id} "
          f"({voice_config.language})")


def demo_custom_preset():
    """Demonstrate adding custom voice presets."""
    print("\n" + "="*60)
    print("DEMO 6: Custom Voice Presets")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Add custom preset
    print("\n1. Add Custom Voice Preset:")
    custom_preset = VoicePreset(
        id="custom_french",
        name="Custom French Voice",
        backend=TTSBackend.PIPER,
        voice_id="fr_FR-siwis-medium",
        language="fr",
        accent="fr-FR",
        description="Custom French voice for testing",
        tags=["french", "custom", "test"]
    )
    voice_manager.add_preset(custom_preset)
    print(f"   Added: {custom_preset.name}")
    print(f"   ID: {custom_preset.id}")
    print(f"   Language: {custom_preset.language}")
    
    # List French voices
    print("\n2. List French Voices:")
    fr_presets = voice_manager.list_presets(language="fr")
    for preset in fr_presets:
        print(f"   - {preset.name} ({preset.backend.value})")
    
    # Select custom voice
    print("\n3. Select Custom Voice:")
    voice_config = voice_manager.select_voice(language="fr")
    print(f"   Selected: {voice_config.voice_id} ({voice_config.language})")


def demo_persistence():
    """Demonstrate preference persistence."""
    print("\n" + "="*60)
    print("DEMO 7: Preference Persistence")
    print("="*60)
    
    # Create voice manager with temporary directory
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    print(f"\n1. Create VoiceManager with temp dir: {temp_dir}")
    voice_manager1 = VoiceManager(preferences_dir=temp_dir)
    
    # Set preferences
    print("\n2. Set User Preferences:")
    prefs = UserVoicePreferences(
        user_id="persistent_user",
        default_voice_id="azure_en_us_jenny",
        preferred_language="en",
        custom_speed=1.3
    )
    voice_manager1.set_user_preferences("persistent_user", prefs)
    print(f"   Set preferences for: persistent_user")
    
    # Create new manager with same directory
    print("\n3. Create New VoiceManager (same directory):")
    voice_manager2 = VoiceManager(preferences_dir=temp_dir)
    
    # Retrieve preferences
    print("\n4. Retrieve Preferences from New Manager:")
    retrieved = voice_manager2.get_user_preferences("persistent_user")
    if retrieved:
        print(f"   User: {retrieved.user_id}")
        print(f"   Default voice: {retrieved.default_voice_id}")
        print(f"   Custom speed: {retrieved.custom_speed}")
        print("   ✓ Preferences persisted successfully!")
    else:
        print("   ✗ Failed to retrieve preferences")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n5. Cleaned up temp directory")


def demo_voice_parameter_adjustment():
    """Demonstrate voice parameter adjustment functionality."""
    print("\n" + "="*60)
    print("DEMO 8: Voice Parameter Adjustment")
    print("="*60)
    
    voice_manager = VoiceManager()
    
    # Get a base voice configuration
    print("\n1. Get Base Voice Configuration:")
    voice_config = voice_manager.select_voice(language="en")
    print(f"   Voice: {voice_config.voice_id}")
    print(f"   Speed: {voice_config.speed}")
    print(f"   Pitch: {voice_config.pitch}")
    print(f"   Volume: {voice_config.volume}")
    
    # Adjust speed
    print("\n2. Adjust Speech Speed:")
    adjusted = voice_manager.adjust_voice_parameters(
        voice_config,
        speed=1.3
    )
    print(f"   New speed: {adjusted.speed}")
    
    # Adjust multiple parameters
    print("\n3. Adjust Multiple Parameters:")
    adjusted = voice_manager.adjust_voice_parameters(
        voice_config,
        speed=1.2,
        pitch=1.1,
        volume=0.9
    )
    print(f"   Speed: {adjusted.speed}")
    print(f"   Pitch: {adjusted.pitch}")
    print(f"   Volume: {adjusted.volume}")
    
    # Preview changes
    print("\n4. Preview Parameter Changes:")
    preview = voice_manager.preview_voice_parameters(
        voice_config,
        speed=1.5,
        pitch=1.2,
        volume=0.85
    )
    print(f"   Current: speed={preview['current']['speed']}, "
          f"pitch={preview['current']['pitch']}, "
          f"volume={preview['current']['volume']}")
    print(f"   Proposed: speed={preview['proposed']['speed']}, "
          f"pitch={preview['proposed']['pitch']}, "
          f"volume={preview['proposed']['volume']}")
    print(f"   Valid: {preview['valid']}")
    if preview['warnings']:
        print(f"   Warnings:")
        for warning in preview['warnings']:
            print(f"     - {warning}")
    
    # Preview with extreme values
    print("\n5. Preview with Extreme Values:")
    preview = voice_manager.preview_voice_parameters(
        voice_config,
        speed=0.6,  # Very slow
        pitch=1.4   # Very high
    )
    print(f"   Valid: {preview['valid']}")
    print(f"   Warnings:")
    for warning in preview['warnings']:
        print(f"     - {warning}")
    
    # Preview with invalid values
    print("\n6. Preview with Invalid Values:")
    preview = voice_manager.preview_voice_parameters(
        voice_config,
        speed=3.0,  # Out of range
        volume=1.5  # Out of range
    )
    print(f"   Valid: {preview['valid']}")
    print(f"   Warnings:")
    for warning in preview['warnings']:
        print(f"     - {warning}")
    
    # Update user voice parameters
    print("\n7. Update User Voice Parameters:")
    user_id = "demo_user"
    voice_manager.update_user_voice_parameters(
        user_id,
        speed=1.3,
        pitch=1.1,
        volume=0.9
    )
    print(f"   Updated parameters for user: {user_id}")
    
    # Retrieve user's voice with custom parameters
    print("\n8. Get User Voice with Custom Parameters:")
    prefs = voice_manager.get_user_preferences(user_id)
    prefs.default_voice_id = "piper_en_us_lessac"
    voice_manager.set_user_preferences(user_id, prefs)
    
    user_voice = voice_manager.select_voice(user_id=user_id)
    print(f"   Voice: {user_voice.voice_id}")
    print(f"   Speed: {user_voice.speed} (custom)")
    print(f"   Pitch: {user_voice.pitch} (custom)")
    print(f"   Volume: {user_voice.volume} (custom)")
    
    # Test validation
    print("\n9. Test Parameter Validation:")
    try:
        voice_manager.adjust_voice_parameters(
            voice_config,
            speed=3.0  # Invalid
        )
        print("   ✗ Validation failed to catch invalid speed")
    except ValueError as e:
        print(f"   ✓ Validation caught invalid speed: {e}")
    
    try:
        voice_manager.update_user_voice_parameters(
            user_id,
            pitch=0.3  # Invalid
        )
        print("   ✗ Validation failed to catch invalid pitch")
    except ValueError as e:
        print(f"   ✓ Validation caught invalid pitch: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Voice Manager Demo")
    print("="*60)
    print("\nThis demo showcases the VoiceManager functionality:")
    print("- Voice preset management")
    print("- User preference storage")
    print("- Voice selection by criteria")
    print("- Multi-user support")
    print("- Preference persistence")
    print("- Voice parameter adjustment")
    
    try:
        demo_basic_usage()
        demo_filtering()
        demo_voice_selection()
        demo_user_preferences()
        demo_multi_user()
        demo_custom_preset()
        demo_persistence()
        demo_voice_parameter_adjustment()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
