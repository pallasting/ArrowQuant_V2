"""
Protocol Adapter for Antigravity Manager API

Automatically selects the optimal protocol (OpenAI/Claude/Gemini) based on the model.
Supports localhost:8045 Antigravity Manager API with multi-protocol capability.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from requests import post


class ProtocolType(Enum):
    """Supported protocol types."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class ModelConfig:
    """Model configuration with optimal protocol."""
    name: str
    protocol: ProtocolType
    endpoint: str
    max_tokens_default: int = 1000
    supports_streaming: bool = True


class ProtocolAdapter:
    """
    Smart protocol adapter for Antigravity Manager API.

    Automatically selects the best protocol format based on model type
    to maximize performance and token efficiency.
    """

    # Model registry with optimal protocol mappings
    MODEL_CONFIGS = {
        # OpenAI models - use OpenAI protocol
        "gpt-4": ModelConfig("gpt-4", ProtocolType.OPENAI, "/v1/chat/completions", 4096),
        "gpt-4-turbo": ModelConfig("gpt-4-turbo", ProtocolType.OPENAI, "/v1/chat/completions", 4096),
        "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", ProtocolType.OPENAI, "/v1/chat/completions", 4096),

        # Claude models - use Claude protocol (most efficient!)
        "claude-opus-4": ModelConfig("claude-opus-4", ProtocolType.CLAUDE, "/v1/messages", 4096),
        "claude-opus-4-6": ModelConfig("claude-opus-4-6", ProtocolType.CLAUDE, "/v1/messages", 4096),
        "claude-sonnet-4": ModelConfig("claude-sonnet-4", ProtocolType.CLAUDE, "/v1/messages", 4096),
        "claude-sonnet-4-5": ModelConfig("claude-sonnet-4-5", ProtocolType.CLAUDE, "/v1/messages", 4096),
        "claude-haiku-4": ModelConfig("claude-haiku-4", ProtocolType.CLAUDE, "/v1/messages", 4096),

        # Gemini models - use OpenAI protocol wrapper
        "gemini-pro": ModelConfig("gemini-pro", ProtocolType.OPENAI, "/v1/chat/completions", 2048),
        "gemini-flash": ModelConfig("gemini-flash", ProtocolType.OPENAI, "/v1/chat/completions", 2048),
        "gemini-3-pro-preview": ModelConfig("gemini-3-pro-preview", ProtocolType.OPENAI, "/v1/chat/completions", 8192),
        "gemini-3-flash-preview": ModelConfig("gemini-3-flash-preview", ProtocolType.OPENAI, "/v1/chat/completions", 8192),
    }

    def __init__(self, base_url: str = "http://localhost:8045", api_key: Optional[str] = None):
        """
        Initialize protocol adapter.

        Args:
            base_url: Antigravity Manager API base URL
            api_key: API authentication key
        """
        self.base_url = base_url
        self.api_key = api_key

    def get_model_config(self, model: str) -> ModelConfig:
        """
        Get configuration for specified model.

        Args:
            model: Model name

        Returns:
            Model configuration with optimal protocol
        """
        # Exact match
        if model in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model]

        # Fuzzy match based on model family
        model_lower = model.lower()
        if "claude" in model_lower:
            return ModelConfig(model, ProtocolType.CLAUDE, "/v1/messages", 4096)
        elif "gpt" in model_lower:
            return ModelConfig(model, ProtocolType.OPENAI, "/v1/chat/completions", 4096)
        elif "gemini" in model_lower:
            return ModelConfig(model, ProtocolType.OPENAI, "/v1/chat/completions", 2048)
        else:
            # Default to OpenAI protocol (most compatible)
            return ModelConfig(model, ProtocolType.OPENAI, "/v1/chat/completions", 1000)

    def build_request_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Build OpenAI protocol request."""
        return {
            "url": f"{self.base_url}/v1/chat/completions",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            "payload": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }

    def build_request_claude(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Claude protocol request."""
        return {
            "url": f"{self.base_url}/v1/messages",
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            "payload": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }

    def parse_response_openai(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI protocol response."""
        if "choices" in response and len(response["choices"]) > 0:
            return {
                "content": response["choices"][0]["message"]["content"],
                "finish_reason": response["choices"][0].get("finish_reason"),
                "usage": response.get("usage", {}),
                "model": response.get("model"),
                "protocol": "openai"
            }
        elif "error" in response:
            raise Exception(f"API Error: {response['error']}")
        else:
            raise Exception(f"Unexpected response format: {response}")

    def parse_response_claude(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Claude protocol response."""
        if "content" in response:
            # Handle both list and string content
            if isinstance(response["content"], list):
                content = response["content"][0]["text"]
            else:
                content = response["content"]

            return {
                "content": content,
                "finish_reason": response.get("stop_reason"),
                "usage": response.get("usage", {}),
                "model": response.get("model"),
                "protocol": "claude"
            }
        elif "error" in response:
            raise Exception(f"API Error: {response['error']}")
        else:
            raise Exception(f"Unexpected response format: {response}")

    def complete(
        self,
        prompt: str,
        model: str = "claude-opus-4",
        max_tokens: int = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Universal completion method with automatic protocol selection.

        Args:
            prompt: User prompt text
            model: Model to use (auto-selects optimal protocol)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional protocol-specific parameters

        Returns:
            Generated text response
        """
        # Get optimal configuration for model
        config = self.get_model_config(model)
        max_tokens = max_tokens or config.max_tokens_default

        # Build messages
        messages = []
        if system_prompt:
            if config.protocol == ProtocolType.CLAUDE:
                # Claude uses system parameter instead of system message
                kwargs["system"] = system_prompt
            else:
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Build request based on protocol
        if config.protocol == ProtocolType.OPENAI:
            request_data = self.build_request_openai(model, messages, max_tokens, temperature, **kwargs)
            parser = self.parse_response_openai
        elif config.protocol == ProtocolType.CLAUDE:
            request_data = self.build_request_claude(model, messages, max_tokens, temperature, **kwargs)
            parser = self.parse_response_claude
        else:
            raise ValueError(f"Unsupported protocol: {config.protocol}")

        # Make API call
        try:
            response = post(
                request_data["url"],
                headers=request_data["headers"],
                json=request_data["payload"],
                timeout=60
            )
            response.raise_for_status()
            json_response = response.json()

            # Parse response
            parsed = parser(json_response)
            return parsed["content"]

        except Exception as e:
            raise Exception(f"API call failed: {e}")

    def complete_with_metadata(
        self,
        prompt: str,
        model: str = "claude-opus-4",
        max_tokens: int = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete with full metadata (tokens, model, protocol).

        Returns dict with: content, usage, model, protocol, finish_reason
        """
        config = self.get_model_config(model)
        max_tokens = max_tokens or config.max_tokens_default

        messages = []
        if system_prompt:
            if config.protocol == ProtocolType.CLAUDE:
                kwargs["system"] = system_prompt
            else:
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Build and execute request
        if config.protocol == ProtocolType.OPENAI:
            request_data = self.build_request_openai(model, messages, max_tokens, temperature, **kwargs)
            parser = self.parse_response_openai
        elif config.protocol == ProtocolType.CLAUDE:
            request_data = self.build_request_claude(model, messages, max_tokens, temperature, **kwargs)
            parser = self.parse_response_claude
        else:
            raise ValueError(f"Unsupported protocol: {config.protocol}")

        response = post(
            request_data["url"],
            headers=request_data["headers"],
            json=request_data["payload"],
            timeout=60
        )
        response.raise_for_status()

        return parser(response.json())


# Convenience functions
def quick_complete(prompt: str, model: str = "claude-opus-4", api_key: str = None) -> str:
    """Quick completion with default settings."""
    adapter = ProtocolAdapter(api_key=api_key)
    return adapter.complete(prompt, model=model)


def get_optimal_protocol(model: str) -> str:
    """Get optimal protocol name for a model."""
    adapter = ProtocolAdapter()
    config = adapter.get_model_config(model)
    return config.protocol.value
