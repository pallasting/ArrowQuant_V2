
"""
AI-OS Auditory Cortex Implementation.

Interfaces with local microphones (PyAudio) to capture auditory stimuli.
"""

import sys
import time
import logging
import threading
import array
from pathlib import Path

logger = logging.getLogger(__name__)

# Try import drivers
AUDIO_DRIVER = None
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_DRIVER = "sounddevice"
except ImportError:
    try:
        import pyaudio
        import wave
        AUDIO_DRIVER = "pyaudio"
    except ImportError:
        logger.warning("No audio driver found (sounddevice/pyaudio). Auditory sense disabled.")

CHUNK = 1024
RATE = 16000
SILENCE_THRESHOLD = 500  # Adjust based on environment

class AudioModule:
    """Manages microphone input stream."""
    
    def __init__(self, output_dir: str, device_index=None):
        if not AUDIO_DRIVER:
            raise ImportError("Audio driver missing")
            
        self.output_dir = Path(output_dir) / "audio_stream"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_index = device_index
        
        # PyAudio State
        self.p = None
        self.stream = None
        
        # SoundDevice State
        self.sd_stream = None
        
        self.running = False
        self.thread = None
        
        # Buffer
        self.recording_buffer = [] # For PyAudio (bytes) or SD (numpy)
        self.is_recording = False
        self.silence_counter = 0

    def start(self):
        """Start listening."""
        if self.running: return
        self.running = True

        if AUDIO_DRIVER == "sounddevice":
            self._start_sd()
        else:
            self._start_pyaudio()
            
    def _start_sd(self):
        """Start sounddevice stream."""
        try:
            def callback(indata, frames, time, status):
                if status: logger.warning(status)
                self._process_chunk(indata) # indata is numpy array

            self.sd_stream = sd.InputStream(
                channels=1,
                samplerate=RATE,
                callback=callback,
                device=self.device_index
            )
            self.sd_stream.start()
            logger.info("Auditory Module started (sounddevice)")
        except Exception as e:
            logger.error(f"SD start failed: {e}")
            self.running = False

    def _start_pyaudio(self):
        """Start PyAudio stream."""
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            self.thread = threading.Thread(target=self._pyaudio_loop, daemon=True)
            self.thread.start()
            logger.info("Auditory Module started (PyAudio)")
        except Exception as e:
            logger.error(f"PyAudio start failed: {e}")
            self.running = False

    def stop(self):
        """Stop listening."""
        self.running = False
        
        # Stop SD
        if self.sd_stream:
            self.sd_stream.stop()
            self.sd_stream.close()
            
        # Stop PyAudio
        if self.thread: self.thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p: self.p.terminate()
            
        logger.info("Auditory Module stopped")

    def _process_chunk(self, data):
        """Common logic for VAD."""
        if AUDIO_DRIVER == "sounddevice":
            # data is numpy float32 [-1, 1] usually
            # Convert to int16 for compatibility or just check energy
            energy = np.max(np.abs(data)) * 32767
            raw_bytes = (data * 32767).astype(np.int16).tobytes()
        else:
            # PyAudio data is bytes
            import array
            snd_data = array.array('h', data)
            if sys.byteorder == 'big': snd_data.byteswap()
            energy = max(snd_data)
            raw_bytes = data
            
        if energy > SILENCE_THRESHOLD:
            if not self.is_recording:
                self.is_recording = True
                logger.info("Voice detected...")
            self.recording_buffer.append(raw_bytes)
            self.silence_counter = 0
        else:
            if self.is_recording:
                self.silence_counter += 1
                self.recording_buffer.append(raw_bytes)
                if self.silence_counter > 60: # ~2 sec
                    self._save_recording(self.recording_buffer)
                    self.recording_buffer = []
                    self.is_recording = False
                    logger.info("Voice ended.")

    def _pyaudio_loop(self):
        """Legacy PyAudio loop."""
        while self.running and self.stream.is_active():
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self._process_chunk(data)
            except Exception:
                break

    def _save_recording(self, buffer):
        """Save buffer as WAV."""
        if not buffer: return
        import wave
        
        timestamp = int(time.time())
        filename = f"audio_{timestamp}.wav"
        filepath = self.output_dir / filename
        
        wf = wave.open(str(filepath), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b''.join(buffer))
        wf.close()
        return filepath
