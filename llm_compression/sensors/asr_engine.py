import logging
import asyncio
from typing import Optional
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ASREngine:
    """
    Automatic Speech Recognition using Faster-Whisper.
    Runs locally, supports CPU and GPU inference.
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize ASR.
        
        Args:
            model_size: tiny, base, small, medium, large-v2, large-v3
            device: 'cuda', 'cpu', 'auto'
        """
        self.model_size = model_size
        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        
        # Load whisper internally
        self.model = WhisperModel(
            model_size, 
            device=device,
            compute_type="default"
        )
        self.executor = ThreadPoolExecutor(max_workers=1)
        logger.info("ASR Engine ready.")
        
    def transcribe_sync(self, audio_path: str, language: str = "zh") -> str:
        """
        Transcribe audio synchronously.
        """
        logger.debug(f"Transcribing {audio_path}...")
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        text = "".join([segment.text for segment in segments])
        logger.debug(f"Transcription result: {text}")
        return text.strip()

    async def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """
        Transcribe audio asynchronously (non-blocking).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.transcribe_sync,
            audio_path,
            language
        )
