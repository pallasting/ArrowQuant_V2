"""
Natural Language Generation Engine for the Expression & Presentation Layer.

This module implements the NLGEngine class that provides text generation
capabilities using multiple backends (OpenAI, Anthropic, Local, Template).

The engine supports:
- Multiple NLG backends with automatic fallback
- Streaming and complete text generation
- Style-aware prompt construction
- Context-aware response generation
- Template-based fallback for reliability

Requirements: 2.1, 2.2
"""

from typing import Iterator, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime

from llm_compression.expression.expression_types import (
    NLGBackend, 
    NLGConfig, 
    ExpressionStyle, 
    ExpressionContext
)
from llm_compression.expression.emotion import TextStyleMapper
from llm_compression.logger import logger
from llm_compression.errors import CompressionError


class NLGError(CompressionError):
    """NLG-specific error."""
    pass


class TemplateEngine:
    """
    Template-based text generation fallback.
    
    Provides simple template-based responses when LLM backends fail.
    Used as a reliable fallback to ensure the system always produces output.
    """
    
    def __init__(self):
        self.templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize template library."""
        return {
            "greeting": {
                "formal": "Good {time_of_day}. How may I assist you today?",
                "casual": "Hey there! What's up?",
                "technical": "Hello. Please specify your technical requirements.",
                "empathetic": "Hello! I'm here to help. How are you feeling?",
                "playful": "Hi! Ready for some fun? What can I do for you?"
            },
            "acknowledgment": {
                "formal": "I understand your request and will process it accordingly.",
                "casual": "Got it! Let me work on that for you.",
                "technical": "Request acknowledged. Processing parameters.",
                "empathetic": "I hear you. Let me help you with that.",
                "playful": "Awesome! Let's make this happen!"
            },
            "error": {
                "formal": "I apologize, but I encountered an error processing your request.",
                "casual": "Oops! Something went wrong. Let me try again.",
                "technical": "Error encountered. Please check your input parameters.",
                "empathetic": "I'm sorry, I'm having trouble with that. Let me help you another way.",
                "playful": "Whoops! That didn't work. Let's try something else!"
            },
            "farewell": {
                "formal": "Thank you for using our service. Have a pleasant {time_of_day}.",
                "casual": "See you later! Take care!",
                "technical": "Session terminated. All processes completed successfully.",
                "empathetic": "Take care of yourself. I'm here if you need me.",
                "playful": "Bye for now! It was fun chatting with you!"
            },
            "default": {
                "formal": "I will address your inquiry to the best of my abilities.",
                "casual": "Let me see what I can do about that.",
                "technical": "Processing your request with available parameters.",
                "empathetic": "I understand this is important to you. Let me help.",
                "playful": "Interesting! Let me think about this one."
            }
        }
    
    def generate(
        self, 
        prompt: str, 
        style: ExpressionStyle, 
        context: Optional[ExpressionContext] = None
    ) -> str:
        """
        Generate response using templates.
        
        Args:
            prompt: Input prompt (used to select template)
            style: Expression style
            context: Expression context for variable substitution
            
        Returns:
            Generated template response
        """
        # Determine template category from prompt
        template_category = self._select_template_category(prompt)
        
        # Adjust style based on context formality if provided
        effective_style = style
        if context and context.formality_level > 0.7:
            effective_style = ExpressionStyle.FORMAL
        elif context and context.formality_level < 0.3:
            effective_style = ExpressionStyle.CASUAL
        
        # Get style-specific template
        style_name = effective_style.value
        template = self.templates[template_category].get(
            style_name, 
            self.templates[template_category]["formal"]
        )
        
        # Substitute variables if context provided
        if context:
            template = self._substitute_variables(template, context)
        
        return template
    
    def _select_template_category(self, prompt: str) -> str:
        """Select template category based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Check for error/failure patterns first (more specific)
        if any(word in prompt_lower for word in ["error", "fail", "problem", "issue"]):
            return "error"
        # Check for greeting patterns
        elif any(word in prompt_lower for word in ["hello", "hi", "greet", "good morning", "good afternoon"]):
            return "greeting"
        # Check for farewell patterns
        elif any(word in prompt_lower for word in ["bye", "goodbye", "farewell", "see you"]):
            return "farewell"
        # Check for acknowledgment patterns
        elif any(word in prompt_lower for word in ["ok", "yes", "acknowledge", "understand"]):
            return "acknowledgment"
        else:
            return "default"
    
    def _substitute_variables(self, template: str, context: ExpressionContext) -> str:
        """Substitute variables in template."""
        variables = {
            "time_of_day": context.time_of_day,
            "user_id": context.user_id,
            "language": context.language
        }
        
        for var, value in variables.items():
            template = template.replace(f"{{{var}}}", str(value))
        
        return template


class ConversationHistory:
    """
    Manages conversation history with context window management.
    
    Tracks conversation turns and manages context window size to stay
    within model token limits while preserving conversation coherence.
    """
    
    def __init__(self, max_turns: int = 10, max_tokens: int = 4000):
        """
        Initialize conversation history manager.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
            max_tokens: Maximum total tokens to keep in context
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, role: str, content: str):
        """
        Add a conversation turn to history.
        
        Args:
            role: Role of the speaker ("user" or "assistant")
            content: Content of the message
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        self._trim_history()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get conversation history as messages for LLM.
        
        Returns:
            List of message dictionaries with role and content
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the conversation context.
        
        Returns:
            String summary of conversation history
        """
        if not self.history:
            return "No previous conversation."
        
        summary_parts = []
        for msg in self.history[-5:]:  # Last 5 messages
            role = msg["role"].capitalize()
            content = msg["content"][:100]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def _trim_history(self):
        """Trim history to stay within limits."""
        # Trim by number of turns
        if len(self.history) > self.max_turns * 2:  # 2 messages per turn
            # Keep most recent turns
            self.history = self.history[-(self.max_turns * 2):]
        
        # Trim by token count (rough estimate: 4 chars per token)
        total_chars = sum(len(msg["content"]) for msg in self.history)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > self.max_tokens:
            # Remove oldest messages until under limit
            while estimated_tokens > self.max_tokens and len(self.history) > 2:
                removed = self.history.pop(0)
                estimated_tokens -= len(removed["content"]) // 4
    
    def get_turn_count(self) -> int:
        """Get number of conversation turns."""
        return len(self.history) // 2
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message."""
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None


class NLGEngine:
    """
    Natural Language Generation engine with multiple backend support.
    
    Supports multiple backends:
    - OpenAI GPT models (cloud)
    - Anthropic Claude (cloud)  
    - Local models via Ollama (local)
    - Template-based fallback (always available)
    
    Features:
    - Automatic backend initialization
    - Streaming and complete generation modes
    - Style-aware prompt construction
    - Graceful fallback to templates on errors
    - Context-aware response generation
    - Conversation history tracking
    - Context window management
    """
    
    def __init__(self, config: NLGConfig):
        """
        Initialize NLG engine with configuration.
        
        Args:
            config: NLG configuration specifying backend and parameters
        """
        self.config = config
        self.backend = None
        self.template_engine = TemplateEngine()
        self.text_style_mapper = TextStyleMapper()
        self.conversation_history = ConversationHistory(
            max_turns=10,
            max_tokens=4000
        )
        
        # Initialize backend
        try:
            self.backend = self._init_backend()
            logger.info(f"NLG engine initialized with {config.backend.value} backend")
        except Exception as e:
            logger.warning(f"Failed to initialize {config.backend.value} backend: {e}")
            logger.info("NLG engine will use template fallback only")
    
    def _init_backend(self):
        """
        Initialize the specified NLG backend.
        
        Returns:
            Initialized backend client or None if initialization fails
            
        Raises:
            NLGError: If backend initialization fails
        """
        try:
            if self.config.backend == NLGBackend.OPENAI:
                return self._init_openai_backend()
            elif self.config.backend == NLGBackend.ANTHROPIC:
                return self._init_anthropic_backend()
            elif self.config.backend == NLGBackend.LOCAL:
                return self._init_local_backend()
            elif self.config.backend == NLGBackend.TEMPLATE:
                return None  # Template engine doesn't need initialization
            else:
                raise NLGError(f"Unsupported NLG backend: {self.config.backend}")
        except Exception as e:
            raise NLGError(f"Failed to initialize {self.config.backend.value} backend: {e}")
    
    def _init_openai_backend(self):
        """Initialize OpenAI backend."""
        try:
            from openai import OpenAI
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise NLGError("OPENAI_API_KEY environment variable not set")
            
            client = OpenAI(api_key=api_key)
            
            # Test the connection with a simple request
            try:
                client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    timeout=5
                )
            except Exception as e:
                logger.warning(f"OpenAI connection test failed: {e}")
            
            return client
            
        except ImportError:
            raise NLGError("OpenAI library not installed. Run: pip install openai")
    
    def _init_anthropic_backend(self):
        """Initialize Anthropic backend."""
        try:
            from anthropic import Anthropic
            import os
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise NLGError("ANTHROPIC_API_KEY environment variable not set")
            
            client = Anthropic(api_key=api_key)
            return client
            
        except ImportError:
            raise NLGError("Anthropic library not installed. Run: pip install anthropic")
    
    def _init_local_backend(self):
        """Initialize local Ollama backend."""
        try:
            import ollama
            
            # Test connection to Ollama
            try:
                ollama.list()
            except Exception as e:
                raise NLGError(f"Cannot connect to Ollama server: {e}")
            
            return ollama
            
        except ImportError:
            raise NLGError("Ollama library not installed. Run: pip install ollama")
    
    def generate(\r\n        self,\r\n        prompt: str,\r\n        style: ExpressionStyle,\r\n        context: ExpressionContext,\r\n        streaming: Optional[bool] = None,\r\n        emotion: Optional[str] = None,\r\n        emotion_intensity: Optional[float] = None\r\n    ) -> Iterator[str]:
        """
        Generate text response with style and context awareness.
        
        Args:
            prompt: Input prompt for generation
            style: Expression style to apply
            context: Expression context for personalization
            streaming: Enable streaming output (overrides config if provided)
            
        Yields:
            Generated text tokens (streaming) or complete response
            
        Raises:
            NLGError: If all backends fail including template fallback
        """
        streaming = streaming if streaming is not None else self.config.streaming
        
        # Build system prompt with style and emotion instructions\r\n        system_prompt = self._build_system_prompt(\r\n            style=style,\r\n            context=context,\r\n            emotion=emotion,\r\n            emotion_intensity=emotion_intensity\r\n        )
        
        # Try main backend first
        if self.backend is not None:
            try:
                if streaming:
                    yield from self._generate_streaming(prompt, system_prompt)
                else:
                    yield self._generate_complete(prompt, system_prompt)
                return
            except Exception as e:
                logger.error(f"NLG generation failed with {self.config.backend.value}: {e}")
        
        # Fallback to template engine
        logger.info("Falling back to template engine")
        try:
            response = self.template_engine.generate(prompt, style, context)
            yield response
        except Exception as e:
            raise NLGError(f"Template fallback also failed: {e}")
    
    def _build_system_prompt(\r\n        self,\r\n        style: ExpressionStyle,\r\n        context: ExpressionContext,\r\n        emotion: Optional[str] = None,\r\n        emotion_intensity: Optional[float] = None\r\n    ) -> str:
        """
        Build system prompt with style and context instructions.
        
        Args:
            style: Expression style to apply
            context: Expression context
            
        Returns:
            System prompt string with style instructions
        """
        base = "You are a helpful AI assistant."
        
        # Style-specific instructions
        style_instructions = {
            ExpressionStyle.FORMAL: "Respond in a formal, professional manner with proper etiquette.",
            ExpressionStyle.CASUAL: "Respond in a casual, friendly manner like talking to a friend.",
            ExpressionStyle.TECHNICAL: "Respond with technical precision, detail, and accuracy.",
            ExpressionStyle.EMPATHETIC: "Respond with empathy, understanding, and emotional support.",
            ExpressionStyle.PLAYFUL: "Respond in a playful, lighthearted, and fun manner."
        }
        
        style_instruction = style_instructions.get(style, "")
        
        # Context-specific adjustments
        context_instructions = []
        
        if context.language != "en":
            context_instructions.append(f"Respond in {context.language} language.")
        
        if context.formality_level > 0.7:
            context_instructions.append("Use formal language and respectful tone.")
        elif context.formality_level < 0.3:
            context_instructions.append("Use casual language and relaxed tone.")
        
        if context.current_emotion != "neutral":
            context_instructions.append(f"Consider the user's emotional state: {context.current_emotion}.")
        
        # Combine all instructions
        all_instructions = [base, style_instruction] + context_instructions
        return " ".join(filter(None, all_instructions))
    
    def _generate_streaming(
        self,
        prompt: str,
        system_prompt: str
    ) -> Iterator[str]:
        """
        Generate streaming response using the configured backend.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt with style instructions
            
        Yields:
            Generated text chunks
        """
        if self.config.backend == NLGBackend.OPENAI:
            yield from self._openai_streaming(prompt, system_prompt)
        elif self.config.backend == NLGBackend.ANTHROPIC:
            yield from self._anthropic_streaming(prompt, system_prompt)
        elif self.config.backend == NLGBackend.LOCAL:
            yield from self._local_streaming(prompt, system_prompt)
        else:
            # Non-streaming backends - generate complete then yield
            response = self._generate_complete(prompt, system_prompt)
            yield response
    
    def _generate_complete(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:
        """
        Generate complete response using the configured backend.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt with style instructions
            
        Returns:
            Complete generated response
        """
        if self.config.backend == NLGBackend.OPENAI:
            return self._openai_complete(prompt, system_prompt)
        elif self.config.backend == NLGBackend.ANTHROPIC:
            return self._anthropic_complete(prompt, system_prompt)
        elif self.config.backend == NLGBackend.LOCAL:
            return self._local_complete(prompt, system_prompt)
        else:
            raise NLGError(f"Complete generation not supported for {self.config.backend}")
    
    def _openai_streaming(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """OpenAI streaming generation."""
        try:
            response = self.backend.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                timeout=self.config.timeout_seconds
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise NLGError(f"OpenAI streaming generation failed: {e}")
    
    def _openai_complete(self, prompt: str, system_prompt: str) -> str:
        """OpenAI complete generation."""
        try:
            response = self.backend.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise NLGError(f"OpenAI complete generation failed: {e}")
    
    def _anthropic_streaming(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Anthropic streaming generation."""
        try:
            # Anthropic uses a different message format
            with self.backend.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise NLGError(f"Anthropic streaming generation failed: {e}")
    
    def _anthropic_complete(self, prompt: str, system_prompt: str) -> str:
        """Anthropic complete generation."""
        try:
            response = self.backend.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text if response.content else ""
            
        except Exception as e:
            raise NLGError(f"Anthropic complete generation failed: {e}")
    
    def _local_streaming(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Local Ollama streaming generation."""
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            response = self.backend.generate(
                model=self.config.model,
                prompt=full_prompt,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            for chunk in response:
                if chunk.get("response"):
                    yield chunk["response"]
                    
        except Exception as e:
            raise NLGError(f"Local streaming generation failed: {e}")
    
    def _local_complete(self, prompt: str, system_prompt: str) -> str:
        """Local Ollama complete generation."""
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            response = self.backend.generate(
                model=self.config.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            return response.get("response", "")
            
        except Exception as e:
            raise NLGError(f"Local complete generation failed: {e}")
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get current backend status and health information.
        
        Returns:
            Dictionary with backend status information
        """
        status = {
            "backend": self.config.backend.value,
            "model": self.config.model,
            "initialized": self.backend is not None,
            "template_fallback_available": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test backend health if initialized
        if self.backend is not None:
            try:
                if self.config.backend == NLGBackend.OPENAI:
                    # Quick health check
                    self.backend.models.list()
                    status["health"] = "healthy"
                elif self.config.backend == NLGBackend.LOCAL:
                    # Check Ollama connection
                    self.backend.list()
                    status["health"] = "healthy"
                else:
                    status["health"] = "unknown"
            except Exception as e:
                status["health"] = "unhealthy"
                status["error"] = str(e)
        else:
            status["health"] = "not_initialized"
        
        return status
    
    def generate_with_history(
        self,
        prompt: str,
        style: ExpressionStyle,
        context: ExpressionContext,
        use_history: bool = True,
        streaming: Optional[bool] = None
    ) -> Iterator[str]:
        """Generate text response with conversation history context."""
        streaming = streaming if streaming is not None else self.config.streaming
        
        if use_history:
            self.conversation_history.add_turn("user", prompt)
        
        system_prompt = self._build_system_prompt(style, context)
        response_parts = []
        
        if self.backend is not None:
            try:
                if use_history and len(self.conversation_history.history) > 0:
                    if streaming:
                        for chunk in self._generate_streaming_with_history(system_prompt):
                            response_parts.append(chunk)
                            yield chunk
                    else:
                        response = self._generate_complete_with_history(system_prompt)
                        response_parts.append(response)
                        yield response
                else:
                    if streaming:
                        for chunk in self._generate_streaming(prompt, system_prompt):
                            response_parts.append(chunk)
                            yield chunk
                    else:
                        response = self._generate_complete(prompt, system_prompt)
                        response_parts.append(response)
                        yield response
                
                if use_history:
                    full_response = "".join(response_parts)
                    self.conversation_history.add_turn("assistant", full_response)
                
                return
            except Exception as e:
                logger.error(f"NLG generation failed with {self.config.backend.value}: {e}")
        
        logger.info("Falling back to template engine")
        try:
            response = self.template_engine.generate(prompt, style, context)
            if use_history:
                self.conversation_history.add_turn("assistant", response)
            yield response
        except Exception as e:
            raise NLGError(f"Template fallback also failed: {e}")
    
    def _generate_streaming_with_history(self, system_prompt: str) -> Iterator[str]:
        """Generate streaming response using conversation history."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history.get_messages())
        
        if self.config.backend == NLGBackend.OPENAI:
            yield from self._openai_streaming_with_messages(messages)
        elif self.config.backend == NLGBackend.ANTHROPIC:
            yield from self._anthropic_streaming_with_messages(messages)
        elif self.config.backend == NLGBackend.LOCAL:
            yield from self._local_streaming_with_messages(messages)
        else:
            raise NLGError(f"History-aware generation not supported for {self.config.backend}")
    
    def _generate_complete_with_history(self, system_prompt: str) -> str:
        """Generate complete response using conversation history."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history.get_messages())
        
        if self.config.backend == NLGBackend.OPENAI:
            return self._openai_complete_with_messages(messages)
        elif self.config.backend == NLGBackend.ANTHROPIC:
            return self._anthropic_complete_with_messages(messages)
        elif self.config.backend == NLGBackend.LOCAL:
            return self._local_complete_with_messages(messages)
        else:
            raise NLGError(f"History-aware generation not supported for {self.config.backend}")
    
    def _openai_streaming_with_messages(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """OpenAI streaming generation with message history."""
        try:
            response = self.backend.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                timeout=self.config.timeout_seconds
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise NLGError(f"OpenAI streaming generation failed: {e}")
    
    def _openai_complete_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI complete generation with message history."""
        try:
            response = self.backend.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise NLGError(f"OpenAI complete generation failed: {e}")
    
    def _anthropic_streaming_with_messages(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Anthropic streaming generation with message history."""
        try:
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
            conv_messages = [m for m in messages if m["role"] != "system"]
            
            with self.backend.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_msg,
                messages=conv_messages
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise NLGError(f"Anthropic streaming generation failed: {e}")
    
    def _anthropic_complete_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Anthropic complete generation with message history."""
        try:
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
            conv_messages = [m for m in messages if m["role"] != "system"]
            
            response = self.backend.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_msg,
                messages=conv_messages
            )
            
            return response.content[0].text if response.content else ""
            
        except Exception as e:
            raise NLGError(f"Anthropic complete generation failed: {e}")
    
    def _local_streaming_with_messages(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Local Ollama streaming generation with message history."""
        try:
            full_prompt = self._messages_to_prompt(messages)
            
            response = self.backend.generate(
                model=self.config.model,
                prompt=full_prompt,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            for chunk in response:
                if chunk.get("response"):
                    yield chunk["response"]
                    
        except Exception as e:
            raise NLGError(f"Local streaming generation failed: {e}")
    
    def _local_complete_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Local Ollama complete generation with message history."""
        try:
            full_prompt = self._messages_to_prompt(messages)
            
            response = self.backend.generate(
                model=self.config.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            return response.get("response", "")
            
        except Exception as e:
            raise NLGError(f"Local complete generation failed: {e}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to a single prompt string for Ollama."""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"\n\nUser: {content}")
            elif role == "assistant":
                prompt_parts.append(f"\nAssistant: {content}")
        
        prompt_parts.append("\nAssistant:")
        return "".join(prompt_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_history_summary(self) -> str:
        """Get a summary of the conversation history."""
        return self.conversation_history.get_context_summary()
    
    def get_turn_count(self) -> int:
        """Get the number of conversation turns."""
        return self.conversation_history.get_turn_count()
