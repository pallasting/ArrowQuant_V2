"""
Unit tests for NLGEngine context management features.

Tests conversation history tracking, context window management,
and multi-turn coherence support.

Requirements: 2.2, 2.6
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from llm_compression.expression.nlg import NLGEngine, ConversationHistory
from llm_compression.expression.expression_types import (
    NLGConfig,
    NLGBackend,
    ExpressionStyle,
    ExpressionContext
)


class TestConversationHistory:
    """Test the ConversationHistory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.history = ConversationHistory(max_turns=5, max_tokens=1000)
    
    def test_init(self):
        """Test ConversationHistory initialization."""
        assert self.history.max_turns == 5
        assert self.history.max_tokens == 1000
        assert len(self.history.history) == 0
    
    def test_add_turn(self):
        """Test adding conversation turns."""
        self.history.add_turn("user", "Hello")
        assert len(self.history.history) == 1
        assert self.history.history[0]["role"] == "user"
        assert self.history.history[0]["content"] == "Hello"
        assert "timestamp" in self.history.history[0]
        
        self.history.add_turn("assistant", "Hi there!")
        assert len(self.history.history) == 2
        assert self.history.history[1]["role"] == "assistant"
        assert self.history.history[1]["content"] == "Hi there!"
    
    def test_get_messages(self):
        """Test getting messages for LLM."""
        self.history.add_turn("user", "Hello")
        self.history.add_turn("assistant", "Hi!")
        self.history.add_turn("user", "How are you?")
        
        messages = self.history.get_messages()
        
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}
        
        # Timestamps should not be included
        assert "timestamp" not in messages[0]
    
    def test_get_context_summary(self):
        """Test getting conversation summary."""
        # Empty history
        summary = self.history.get_context_summary()
        assert summary == "No previous conversation."
        
        # Add some messages
        self.history.add_turn("user", "Hello")
        self.history.add_turn("assistant", "Hi there!")
        self.history.add_turn("user", "How are you?")
        
        summary = self.history.get_context_summary()
        
        assert "User: Hello" in summary
        assert "Assistant: Hi there!" in summary
        assert "User: How are you?" in summary
    
    def test_get_context_summary_truncates_long_messages(self):
        """Test that summary truncates long messages."""
        long_message = "x" * 200
        self.history.add_turn("user", long_message)
        
        summary = self.history.get_context_summary()
        
        # Should truncate to 100 characters
        assert len(summary.split(": ")[1]) <= 100
    
    def test_clear(self):
        """Test clearing history."""
        self.history.add_turn("user", "Hello")
        self.history.add_turn("assistant", "Hi!")
        
        assert len(self.history.history) == 2
        
        self.history.clear()
        
        assert len(self.history.history) == 0
    
    def test_trim_by_turn_count(self):
        """Test trimming history by turn count."""
        # Add more turns than max_turns
        for i in range(7):
            self.history.add_turn("user", f"Message {i}")
            self.history.add_turn("assistant", f"Response {i}")
        
        # Should keep only max_turns * 2 messages (5 turns = 10 messages)
        assert len(self.history.history) <= self.history.max_turns * 2
        
        # Should keep most recent messages
        messages = self.history.get_messages()
        assert "Message 6" in messages[-2]["content"]
        assert "Response 6" in messages[-1]["content"]
    
    def test_trim_by_token_count(self):
        """Test trimming history by token count."""
        # Create history with small token limit
        small_history = ConversationHistory(max_turns=100, max_tokens=100)
        
        # Add messages that exceed token limit
        # Rough estimate: 4 chars per token, so 400 chars = 100 tokens
        for i in range(5):
            small_history.add_turn("user", "x" * 100)  # ~25 tokens each
            small_history.add_turn("assistant", "y" * 100)  # ~25 tokens each
        
        # Should trim to stay under token limit
        total_chars = sum(len(msg["content"]) for msg in small_history.history)
        estimated_tokens = total_chars // 4
        
        assert estimated_tokens <= small_history.max_tokens + 50  # Allow some margin
    
    def test_get_turn_count(self):
        """Test getting turn count."""
        assert self.history.get_turn_count() == 0
        
        self.history.add_turn("user", "Hello")
        assert self.history.get_turn_count() == 0  # Incomplete turn
        
        self.history.add_turn("assistant", "Hi!")
        assert self.history.get_turn_count() == 1  # Complete turn
        
        self.history.add_turn("user", "How are you?")
        self.history.add_turn("assistant", "I'm good!")
        assert self.history.get_turn_count() == 2
    
    def test_get_last_user_message(self):
        """Test getting last user message."""
        assert self.history.get_last_user_message() is None
        
        self.history.add_turn("user", "First message")
        assert self.history.get_last_user_message() == "First message"
        
        self.history.add_turn("assistant", "Response")
        assert self.history.get_last_user_message() == "First message"
        
        self.history.add_turn("user", "Second message")
        assert self.history.get_last_user_message() == "Second message"
    
    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        assert self.history.get_last_assistant_message() is None
        
        self.history.add_turn("user", "Hello")
        assert self.history.get_last_assistant_message() is None
        
        self.history.add_turn("assistant", "First response")
        assert self.history.get_last_assistant_message() == "First response"
        
        self.history.add_turn("user", "Another question")
        self.history.add_turn("assistant", "Second response")
        assert self.history.get_last_assistant_message() == "Second response"


class TestNLGEngineContextManagement:
    """Test NLGEngine context management features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = NLGConfig(
            backend=NLGBackend.TEMPLATE,
            model="test-model",
            streaming=False
        )
        
        self.context = ExpressionContext(
            user_id="test_user",
            conversation_history=[],
            current_emotion="neutral",
            formality_level=0.5,
            language="en"
        )
    
    def test_engine_has_conversation_history(self):
        """Test that NLGEngine initializes with conversation history."""
        engine = NLGEngine(self.config)
        
        assert hasattr(engine, 'conversation_history')
        assert isinstance(engine.conversation_history, ConversationHistory)
        assert engine.conversation_history.max_turns == 10
        assert engine.conversation_history.max_tokens == 4000
    
    def test_generate_with_history_adds_to_history(self):
        """Test that generate_with_history adds messages to history."""
        engine = NLGEngine(self.config)
        
        # Generate response
        prompt = "Hello, how are you?"
        list(engine.generate_with_history(
            prompt, 
            ExpressionStyle.CASUAL, 
            self.context,
            use_history=True
        ))
        
        # Check history was updated
        assert engine.conversation_history.get_turn_count() == 1
        assert engine.conversation_history.get_last_user_message() == prompt
        assert engine.conversation_history.get_last_assistant_message() is not None
    
    def test_generate_with_history_disabled(self):
        """Test that history can be disabled."""
        engine = NLGEngine(self.config)
        
        # Generate without history
        list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            self.context,
            use_history=False
        ))
        
        # History should be empty
        assert engine.conversation_history.get_turn_count() == 0
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        engine = NLGEngine(self.config)
        
        # Add some history
        list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            self.context
        ))
        
        assert engine.conversation_history.get_turn_count() > 0
        
        # Clear history
        engine.clear_history()
        
        assert engine.conversation_history.get_turn_count() == 0
    
    def test_get_history_summary(self):
        """Test getting history summary."""
        engine = NLGEngine(self.config)
        
        # Empty history
        summary = engine.get_history_summary()
        assert "No previous conversation" in summary
        
        # Add some history
        list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            self.context
        ))
        
        summary = engine.get_history_summary()
        assert "User: Hello" in summary
    
    def test_get_turn_count(self):
        """Test getting turn count."""
        engine = NLGEngine(self.config)
        
        assert engine.get_turn_count() == 0
        
        list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            self.context
        ))
        
        assert engine.get_turn_count() == 1
    
    @patch('openai.OpenAI')
    def test_multi_turn_conversation_openai(self, mock_openai):
        """Test multi-turn conversation with OpenAI backend."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock responses
        mock_response1 = Mock()
        mock_response1.choices = [Mock()]
        mock_response1.choices[0].message.content = "Hi! I'm doing well."
        
        mock_response2 = Mock()
        mock_response2.choices = [Mock()]
        mock_response2.choices[0].message.content = "I can help with many things!"
        
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Connection test
            mock_response1,  # First turn
            mock_response2   # Second turn
        ]
        
        config = NLGConfig(backend=NLGBackend.OPENAI, streaming=False)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            engine = NLGEngine(config)
        
        # First turn
        response1 = list(engine.generate_with_history(
            "Hello, how are you?",
            ExpressionStyle.CASUAL,
            self.context
        ))
        
        assert response1 == ["Hi! I'm doing well."]
        assert engine.get_turn_count() == 1
        
        # Second turn - should include first turn in context
        response2 = list(engine.generate_with_history(
            "What can you help me with?",
            ExpressionStyle.CASUAL,
            self.context
        ))
        
        assert response2 == ["I can help with many things!"]
        assert engine.get_turn_count() == 2
        
        # Verify second call included history
        second_call = mock_client.chat.completions.create.call_args_list[2]
        messages = second_call[1]["messages"]
        
        # Should have system + 2 user messages + 1 assistant message
        assert len(messages) >= 4
        assert any(msg["content"] == "Hello, how are you?" for msg in messages if msg["role"] == "user")
        assert any(msg["content"] == "Hi! I'm doing well." for msg in messages if msg["role"] == "assistant")
    
    @patch('openai.OpenAI')
    def test_streaming_with_history(self, mock_openai):
        """Test streaming generation with history."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " there"
        
        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = None
        
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Connection test
            [mock_chunk1, mock_chunk2, mock_chunk3]  # Streaming response
        ]
        
        config = NLGConfig(backend=NLGBackend.OPENAI, streaming=True)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            engine = NLGEngine(config)
        
        # Generate with streaming
        response = list(engine.generate_with_history(
            "Hi",
            ExpressionStyle.CASUAL,
            self.context,
            streaming=True
        ))
        
        assert response == ["Hello", " there"]
        
        # History should contain complete response
        assert engine.conversation_history.get_last_assistant_message() == "Hello there"
    
    def test_messages_to_prompt_conversion(self):
        """Test conversion of messages to prompt for local models."""
        engine = NLGEngine(self.config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        prompt = engine._messages_to_prompt(messages)
        
        assert "You are helpful." in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi!" in prompt
        assert "User: How are you?" in prompt
        assert prompt.endswith("Assistant:")
    
    def test_context_window_management(self):
        """Test that context window is managed properly."""
        # Create engine with small context window
        engine = NLGEngine(self.config)
        engine.conversation_history = ConversationHistory(max_turns=2, max_tokens=200)
        
        # Add multiple turns
        for i in range(5):
            list(engine.generate_with_history(
                f"Message {i}",
                ExpressionStyle.CASUAL,
                self.context
            ))
        
        # Should only keep recent turns
        assert engine.get_turn_count() <= 2
        
        # Should keep most recent messages
        last_user = engine.conversation_history.get_last_user_message()
        assert "Message 4" in last_user
    
    def test_multi_turn_coherence(self):
        """Test that multi-turn conversations maintain coherence."""
        engine = NLGEngine(self.config)
        
        # Simulate a multi-turn conversation
        turns = [
            "Hello",
            "What's your name?",
            "Nice to meet you",
            "Can you help me?"
        ]
        
        for turn in turns:
            list(engine.generate_with_history(
                turn,
                ExpressionStyle.CASUAL,
                self.context
            ))
        
        # Verify all turns are in history
        assert engine.get_turn_count() == len(turns)
        
        # Verify messages are in order
        messages = engine.conversation_history.get_messages()
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        
        assert user_messages == turns


class TestContextManagementIntegration:
    """Integration tests for context management."""
    
    def test_template_engine_with_history(self):
        """Test that template engine works with history tracking."""
        config = NLGConfig(backend=NLGBackend.TEMPLATE)
        engine = NLGEngine(config)
        
        context = ExpressionContext(
            user_id="test",
            conversation_history=[],
            formality_level=0.5
        )
        
        # Multi-turn conversation
        response1 = list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            context
        ))
        
        response2 = list(engine.generate_with_history(
            "How are you?",
            ExpressionStyle.CASUAL,
            context
        ))
        
        # Both should generate responses
        assert len(response1) == 1
        assert len(response2) == 1
        
        # History should track both turns
        assert engine.get_turn_count() == 2
    
    def test_history_persists_across_generate_calls(self):
        """Test that history persists across multiple generate calls."""
        config = NLGConfig(backend=NLGBackend.TEMPLATE)
        engine = NLGEngine(config)
        
        context = ExpressionContext(
            user_id="test",
            conversation_history=[]
        )
        
        # Use both generate and generate_with_history
        list(engine.generate_with_history(
            "First message",
            ExpressionStyle.CASUAL,
            context
        ))
        
        # History should persist
        assert engine.get_turn_count() == 1
        
        list(engine.generate_with_history(
            "Second message",
            ExpressionStyle.CASUAL,
            context
        ))
        
        assert engine.get_turn_count() == 2
    
    def test_error_handling_with_history(self):
        """Test error handling doesn't corrupt history."""
        config = NLGConfig(backend=NLGBackend.TEMPLATE)
        engine = NLGEngine(config)
        
        context = ExpressionContext(
            user_id="test",
            conversation_history=[]
        )
        
        # Add successful turn
        list(engine.generate_with_history(
            "Hello",
            ExpressionStyle.CASUAL,
            context
        ))
        
        initial_count = engine.get_turn_count()
        
        # Even with errors, history should remain consistent
        try:
            list(engine.generate_with_history(
                "",  # Empty prompt
                ExpressionStyle.CASUAL,
                context
            ))
        except:
            pass
        
        # History should still be valid
        assert engine.get_turn_count() >= initial_count
