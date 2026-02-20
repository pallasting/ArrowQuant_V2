"""
Multi-Modal Expression Layer - Generate output from memories

Implements text generation (required) with optional image/audio support.
Reconstructs memories and generates coherent output.
"""

from typing import List, Optional, Any
from dataclasses import dataclass
import numpy as np

from .memory_primitive import MemoryPrimitive
from .reconstructor import LLMReconstructor
from .llm_client import LLMClient
from .compressor import CompressedMemory


@dataclass
class ExpressionResult:
    """Result of expression generation."""
    content: str
    modality: str
    quality_score: float
    source_memories: List[str]  # Memory IDs


class MultiModalExpressor:
    """
    Multi-modal expression layer.
    
    Generates output from memories in different modalities:
    - Text (required): Generate coherent text from memories
    - Image (optional): Generate images from text descriptions
    - Audio (optional): Generate speech from text
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        reconstructor: LLMReconstructor
    ):
        """
        Initialize expressor.
        
        Args:
            llm_client: LLM client for text generation
            reconstructor: Reconstructor for memory decompression
        """
        self.llm_client = llm_client
        self.reconstructor = reconstructor
    
    async def express(
        self,
        memories: List[MemoryPrimitive],
        modality: str = "text",
        style: Optional[str] = None,
        max_length: int = 500
    ) -> ExpressionResult:
        """
        Generate output from memories.
        
        Args:
            memories: List of memories to express
            modality: Output modality ("text", "image", "audio")
            style: Optional style hint (e.g., "concise", "detailed")
            max_length: Maximum output length
            
        Returns:
            ExpressionResult with generated content
        """
        if modality == "text":
            return await self._express_text(memories, None, style, max_length)
        elif modality == "image":
            raise NotImplementedError("Image generation not yet implemented")
        elif modality == "audio":
            return await self._express_audio(memories, None, style, max_length)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
            
    async def express_audio(
        self,
        memories: List[Any],
        query: Optional[str] = None,
        style: Optional[str] = None,
        max_length: int = 500
    ) -> ExpressionResult:
        """
        Convenience method for audio expression.
        
        Args:
            memories: List of CompressedMemory or MemoryPrimitive
            query: Optional query context
            style: Optional style hint
            max_length: Maximum output length
            
        Returns:
            ExpressionResult with generated audio path
        """
        # Convert to MemoryPrimitive if needed
        mem_primitives = []
        for mem in memories:
            if isinstance(mem, MemoryPrimitive):
                mem_primitives.append(mem)
            elif isinstance(mem, CompressedMemory):
                mem_primitives.append(MemoryPrimitive(
                    id=mem.memory_id,
                    content=mem,
                    embedding=np.array(mem.embedding)
                ))
            else:
                continue
                
        return await self._express_audio(mem_primitives, query, style, max_length)
    
    async def express_text(
        self,
        memories: List[Any],
        query: Optional[str] = None,
        style: Optional[str] = None,
        max_length: int = 500
    ) -> ExpressionResult:
        """
        Convenience method for text expression.
        
        Args:
            memories: List of CompressedMemory or MemoryPrimitive
            query: Optional query context
            style: Optional style hint
            max_length: Maximum output length
            
        Returns:
            ExpressionResult with generated text
        """
        # Convert to MemoryPrimitive if needed
        mem_primitives = []
        for mem in memories:
            if isinstance(mem, MemoryPrimitive):
                mem_primitives.append(mem)
            elif isinstance(mem, CompressedMemory):
                # It's CompressedMemory, wrap it
                mem_primitives.append(MemoryPrimitive(
                    id=mem.memory_id,
                    content=mem,
                    embedding=np.array(mem.embedding)
                ))
            else:
                # Unknown type, skip
                print(f"Warning: Unknown memory type: {type(mem)}")
                continue
        
        return await self._express_text(mem_primitives, query, style, max_length)
    
    async def _express_text(
        self,
        memories: List[MemoryPrimitive],
        query: Optional[str],
        style: Optional[str],
        max_length: int
    ) -> ExpressionResult:
        """
        Generate text from memories.
        
        Args:
            memories: List of memories
            query: User query
            style: Optional style hint
            max_length: Maximum output length
            
        Returns:
            ExpressionResult with generated text
        """
        # 1. Reconstruct all memories
        reconstructed_texts = []
        memory_ids = []
        
        for memory in memories:
            reconstructed = await self.reconstructor.reconstruct(memory.content)
            reconstructed_texts.append(reconstructed.full_text)
            memory_ids.append(memory.id)
        
        # 2. Combine texts
        combined = self._combine_texts(reconstructed_texts)
        
        # 3. Generate coherent output
        generated = await self._generate_text(combined, query, style, max_length)
        
        # 4. Calculate quality (simple heuristic)
        quality = self._estimate_quality(generated, combined)
        
        return ExpressionResult(
            content=generated,
            modality="text",
            quality_score=quality,
            source_memories=memory_ids
        )
        
    async def _express_audio(
        self,
        memories: List[MemoryPrimitive],
        query: Optional[str],
        style: Optional[str],
        max_length: int
    ) -> ExpressionResult:
        """
        Generate audio from memories using TTSEngine.
        """
        import os
        import time
        import soundfile as sf
        from llm_compression.expression.tts.tts_engine import TTSEngine
        from llm_compression.expression.expression_types import TTSConfig, TTSBackend
        
        # 1. Generate textual response from memories
        text_result = await self._express_text(memories, query, style, max_length)
        
        if not text_result.content:
            return text_result
            
        # 2. Synthesize using TTSEngine
        config = TTSConfig(
            backend=TTSBackend.PIPER,
            streaming=False,
            cache_enabled=True
        )
        tts = TTSEngine(config)
        chunks = list(tts.synthesize(text_result.content))
        
        if not chunks:
            return text_result  # Fallback to text if TTS fails
            
        audio_array = chunks[0]
        
        # 3. Save to file
        output_dir = "output_audio"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/speech_{int(time.time())}.wav"
        sf.write(filename, audio_array, config.sample_rate)
        
        return ExpressionResult(
            content=filename,
            modality="audio",
            quality_score=text_result.quality_score,
            source_memories=text_result.source_memories
        )
    
    def _combine_texts(self, texts: List[str]) -> str:
        """
        Combine multiple texts into one context.
        
        Args:
            texts: List of text strings
            
        Returns:
            Combined text
        """
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Simple combination with separators
        return "\n\n".join(f"[Memory {i+1}]: {text}" for i, text in enumerate(texts))
    
    async def _generate_text(
        self,
        context: str,
        query: Optional[str],
        style: Optional[str],
        max_length: int
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            context: Memory context
            query: User query
            style: Optional style hint
            max_length: Maximum output length
            
        Returns:
            Generated text
        """
        # Build prompt
        style_instruction = ""
        if style == "concise":
            style_instruction = "Provide a concise summary. "
        elif style == "detailed":
            style_instruction = "Provide a detailed explanation. "
        
        # Include query if available
        if query:
            if context:
                prompt = f"""{style_instruction}User query: {query}

Relevant memories:
{context}

Based on the above memories, respond to the user's query:"""
            else:
                # No memories - respond directly to query
                prompt = f"""{style_instruction}User query: {query}

No relevant memories found. Respond naturally to the user's query:"""
        else:
            # No query - just synthesize from context
            prompt = f"""{style_instruction}Based on the following information, generate a coherent response:

{context}

Response:"""
        
        # Generate
        response = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=max_length,
            temperature=0.7
        )
        
        return response.text.strip()
    
    def _estimate_quality(self, generated: str, context: str) -> float:
        """
        Estimate quality of generated text.
        
        Simple heuristic based on:
        - Length (not too short, not too long)
        - Keyword overlap with context
        
        Args:
            generated: Generated text
            context: Original context
            
        Returns:
            Quality score (0.0-1.0)
        """
        if not generated:
            return 0.0
        
        # Length score (prefer 50-500 chars)
        length = len(generated)
        if length < 20:
            length_score = length / 20.0
        elif length > 1000:
            length_score = max(0.5, 1.0 - (length - 1000) / 1000.0)
        else:
            length_score = 1.0
        
        # Keyword overlap score
        context_words = set(context.lower().split())
        generated_words = set(generated.lower().split())
        
        if not context_words:
            overlap_score = 1.0
        else:
            overlap = len(context_words & generated_words)
            overlap_score = min(1.0, overlap / max(10, len(context_words) * 0.3))
        
        # Combined score
        return (length_score * 0.3 + overlap_score * 0.7)
