import time
import os
import sounddevice as sd
import numpy as np

from llm_compression.sensors.audio import AudioModule
from llm_compression.sensors.asr_engine import ASREngine
from llm_compression.expression.tts.tts_engine import TTSEngine
from llm_compression.expression.expression_types import TTSConfig, TTSBackend
from llm_compression.action.manager import ActionManager
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
            # Let Whisper auto-detect language or force target
            text = asr.transcribe_sync(filepath, language="zh") # Default to Chinese detection
            print(f"👂 [User Said]: {text}")
            
            # Bilingual keyword matching
            ltext = text.lower()
            if any(k in ltext for k in ["move", "right", "移", "右", "动"]):
                import pyautogui
                w, h = pyautogui.size()
                action.execute("move", x=w//2, y=h//2) # move to center
            elif any(k in ltext for k in ["click", "点", "击"]):
                action.execute("click", clicks=1)
            elif any(k in ltext for k in ["type", "hello", "打", "写", "字"]):
                action.execute("type", text="hello from voice ai ")
            else:
                print("🤔 [Intent] Command not recognized.")
        except Exception as e:
            print(f"Error handling voice command: {e}")

    print("\n🎤 Starting Auditory Microphone Capture...")
    mic = AudioModule(output_dir="workspace", on_audio_ready=on_voice_command)
    mic.start()
    
    tts_play(tts, "Voice control initialized. Say move or click to test actions.")
    print("\n=============================================")
    print("🎙️ Microphone Live! System is listening...")
    print("Try speaking a command (in Chinese or English):")
    print("  - 'move right' / '往右移'")
    print("  - 'click' / '点击'")
    print("  - 'type something' / '打字'")
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
