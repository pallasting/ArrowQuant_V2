import time
import os
import sounddevice as sd
import numpy as np

from llm_compression.sensors.audio import AudioModule
from llm_compression.sensors.asr_engine import ASREngine
from llm_compression.expression.tts.tts_engine import TTSEngine
from llm_compression.expression.expression_types import TTSConfig, TTSBackend
from llm_compression.action.manager import ActionManager

def tts_play(tts_engine, text):
    """Play TTS audibly."""
    try:
        chunks = list(tts_engine.synthesize(text))
        if chunks:
            arr = np.concatenate(chunks)
            sd.play(arr, samplerate=tts_engine.config.sample_rate)
    except Exception as e:
        print(f"Failed to play TTS: {e}")

def main():
    print("🧠 Loading AI-OS Cortex Modules...")

    # Load TTS (Expression Layer Extension)
    config = TTSConfig(backend=TTSBackend.PIPER, streaming=False, cache_enabled=False)
    tts = TTSEngine(config)

    # Load Action Manager with embedded TTS narration Hook
    os.makedirs("workspace", exist_ok=True)
    action = ActionManager("workspace", tts_engine=tts)

    # Load ASR (Auditory Perception)
    asr = ASREngine(model_size="tiny", device="auto")

    # The Command Evaluation loop (A prototype cognitive intent matcher)
    def on_voice_command(filepath):
        print(f"\n[ASR] Transcribing Voice Command...")
        try:
            text = asr.transcribe_sync(filepath, language="en")
            print(f"👂 [User Said]: {text}")
            
            # Simple keyword-based intent resolution (To be replaced by CognitiveLoop LLM in future)
            ltext = text.lower()
            if "move" in ltext or "right" in ltext:
                action.execute("move", x=1000, y=500)
            elif "click" in ltext:
                action.execute("click", clicks=1)
            elif "type" in ltext or "hello" in ltext:
                action.execute("type", text="hello from voice ai ")
            else:
                pass # Unrecognized
        except Exception as e:
            print(f"Error handling voice command: {e}")

    print("\n🎤 Starting Auditory Microphone Capture...")
    mic = AudioModule(output_dir="workspace", on_audio_ready=on_voice_command)
    mic.start()
    
    tts_play(tts, "Voice control initialized. Say move or click to test actions.")
    print("\n=============================================")
    print("🎙️ Microphone Live! System is listening...")
    print("Try speaking a command in English:")
    print("  - 'move right'")
    print("  - 'click'")
    print("  - 'type something'")
    print("Press Ctrl+C to stop.")
    print("=============================================\n")
    
    try:
        while True:
            time.sleep(1)
            # Sleep keeps the main thread alive while callbacks handle events
    except KeyboardInterrupt:
        mic.stop()
        print("\nShutting down AI-OS Voice Agent.")

if __name__ == "__main__":
    main()
