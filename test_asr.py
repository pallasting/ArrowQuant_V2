import time
import os
from llm_compression.sensors.audio import AudioModule
from llm_compression.sensors.asr_engine import ASREngine

def main():
    print("Loading Faster-Whisper ASR engine...")
    asr = ASREngine(model_size="tiny", device="auto")
    
    def on_voice(filepath):
        print(f"\n[ASR] Transcribing file: {filepath}")
        try:
            text = asr.transcribe_sync(filepath, language="en")
            print(f"[ASR] User said: {text}")
        except Exception as e:
            print(f"[ASR] Error transcribing: {e}")

    os.makedirs("test_audio_out", exist_ok=True)
    audio = AudioModule(output_dir="test_audio_out", on_audio_ready=on_voice)
    audio.start()
    
    print("\nMicrophone listening configured successfully!")
    print("\nGenerating a simulated speech file via TTS to test ASR...")
    
    from llm_compression.expression.tts.tts_engine import TTSEngine
    from llm_compression.expression.expression_types import TTSConfig, TTSBackend
    import soundfile as sf
    import numpy as np
    
    config = TTSConfig(backend=TTSBackend.PIPER, streaming=False, cache_enabled=False)
    tts = TTSEngine(config)
    
    test_text = "This is a test of the artificial intelligence operating system speech recognition."
    chunks = list(tts.synthesize(test_text))
    if chunks:
        audio_array = chunks[0]
        test_file = "test_audio_out/simulated_speech.wav"
        sf.write(test_file, audio_array, config.sample_rate)
        
        on_voice(test_file)
        
    audio.stop()

if __name__ == "__main__":
    main()
