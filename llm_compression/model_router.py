"""
Intelligent LLM Model Router - Phase 2.0 Enhancement

Automatically selects the optimal model based on task characteristics.
Supports multiple vendors via unified API gateway (port 8046).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


class TaskType(Enum):
    """Task types for model selection."""
    COMPRESSION = "compression"           # Summary generation
    RECONSTRUCTION = "reconstruction"     # Text expansion
    REASONING = "reasoning"              # Complex analysis
    CODE_GENERATION = "code_generation"  # Code tasks
    TRANSLATION = "translation"          # Cross-language
    QUICK_QUERY = "quick_query"          # Simple Q&A
    MULTIMODAL = "multimodal"           # Vision/Audio


class ModelTier(Enum):
    """Model capability tiers."""
    THINKING = "thinking"      # Deep reasoning (Opus-level)
    PREMIUM = "premium"        # High quality (Pro-level)
    STANDARD = "standard"      # Fast response (Flash-level)
    SPECIALIZED = "specialized"  # Domain-specific


@dataclass
class ModelEndpoint:
    """Model endpoint configuration."""
    name: str
    vendor: str
    path: str
    tier: ModelTier
    best_for: list[TaskType]
    max_tokens: int = 8192
    cost_factor: float = 1.0  # Relative cost (1.0 = baseline)
    protocol: str = "openai"  # "openai" or "claude"


class ModelRouter:
    """
    Intelligent model router for multi-vendor LLM gateway.

    Selects optimal model based on:
    - Task type and complexity
    - Quality requirements
    - Cost constraints
    - Response time needs
    """

    # Model registry
    MODELS = {
        # Thinking tier - for complex reasoning
        "gemini-thinking": ModelEndpoint(
            name="gemini-claude-opus-4-6-thinking",
            vendor="gemini-antigravity",
            path="/gemini-antigravity/v1/messages",
            tier=ModelTier.THINKING,
            best_for=[TaskType.REASONING, TaskType.RECONSTRUCTION],
            max_tokens=8192,
            cost_factor=3.0
        ),

        "claude-opus": ModelEndpoint(
            name="claude-opus-4-6",
            vendor="claude-kiro-oauth",
            path="/claude-kiro-oauth/v1/messages",
            tier=ModelTier.THINKING,
            best_for=[TaskType.REASONING, TaskType.RECONSTRUCTION],
            max_tokens=8192,
            cost_factor=2.5
        ),

        # Premium tier - high quality, balanced
        "gemini-pro": ModelEndpoint(
            name="gemini-3-pro-preview",
            vendor="gemini-cli-oauth",
            path="/gemini-cli-oauth/v1/messages",
            tier=ModelTier.PREMIUM,
            best_for=[TaskType.COMPRESSION, TaskType.QUICK_QUERY],
            max_tokens=8192,
            cost_factor=1.0
        ),

        # Standard tier - fast response
        "gemini-flash": ModelEndpoint(
            name="gemini-3-flash-preview",
            vendor="gemini-cli-oauth",
            path="/gemini-cli-oauth/v1/messages",
            tier=ModelTier.STANDARD,
            best_for=[TaskType.QUICK_QUERY, TaskType.COMPRESSION],
            max_tokens=8192,
            cost_factor=0.3
        ),

        # Specialized tier - domain experts
        "qwen-coder": ModelEndpoint(
            name="qwen3-coder-plus",
            vendor="openai-qwen-oauth",
            path="/openai-qwen-oauth/v1/messages",
            tier=ModelTier.SPECIALIZED,
            best_for=[TaskType.CODE_GENERATION],
            max_tokens=32768,
            cost_factor=0.8
        ),

        "kimi-chinese": ModelEndpoint(
            name="kimi-k2.5",
            vendor="openai-iflow",
            path="/openai-iflow/v1/messages",
            tier=ModelTier.SPECIALIZED,
            best_for=[TaskType.TRANSLATION, TaskType.MULTIMODAL],
            max_tokens=8192,
            cost_factor=0.5
        ),
    }

    def __init__(
        self,
        base_url: str = "http://192.168.1.99:8046",
        api_key: Optional[str] = None,
        prefer_cost_efficient: bool = False
    ):
        """
        Initialize router.

        Args:
            base_url: API gateway base URL
            api_key: API authentication key
            prefer_cost_efficient: Prioritize cost over quality
        """
        self.base_url = base_url
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.prefer_cost_efficient = prefer_cost_efficient

    def select_model(
        self,
        task_type: TaskType,
        quality_threshold: float = 0.85,
        max_cost_factor: float = 3.0,
        prefer_vendor: Optional[str] = None
    ) -> ModelEndpoint:
        """
        Select optimal model for task.

        Args:
            task_type: Type of task to perform
            quality_threshold: Minimum quality requirement (0.0-1.0)
            max_cost_factor: Maximum acceptable cost multiplier
            prefer_vendor: Preferred vendor (if available)

        Returns:
            Selected model endpoint configuration
        """
        # Filter candidates by task type
        candidates = [
            model for model in self.MODELS.values()
            if task_type in model.best_for and model.cost_factor <= max_cost_factor
        ]

        if not candidates:
            # Fallback to general-purpose model
            candidates = [self.MODELS["gemini-pro"]]

        # Apply vendor preference
        if prefer_vendor:
            vendor_matches = [m for m in candidates if m.vendor == prefer_vendor]
            if vendor_matches:
                candidates = vendor_matches

        # Select based on strategy
        if self.prefer_cost_efficient:
            # Lowest cost that meets quality threshold
            return min(candidates, key=lambda m: m.cost_factor)
        else:
            # Highest tier available
            tier_priority = {
                ModelTier.THINKING: 4,
                ModelTier.PREMIUM: 3,
                ModelTier.SPECIALIZED: 2,
                ModelTier.STANDARD: 1
            }
            return max(candidates, key=lambda m: tier_priority[m.tier])

    def get_endpoint_config(self, model: ModelEndpoint) -> Dict[str, Any]:
        """
        Get full endpoint configuration for API call.

        Args:
            model: Selected model endpoint

        Returns:
            Configuration dict with URL, headers, model name
        """
        return {
            "url": f"{self.base_url}{model.path}",
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            },
            "model": model.name,
            "max_tokens": model.max_tokens
        }

    def estimate_cost(
        self,
        model: ModelEndpoint,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate API call cost (relative units).

        Args:
            model: Model endpoint
            input_tokens: Estimated input token count
            output_tokens: Estimated output token count

        Returns:
            Estimated cost in relative units
        """
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * model.cost_factor


# Convenience functions for common tasks

def get_compression_model(prefer_fast: bool = False) -> ModelEndpoint:
    """Get optimal model for memory compression."""
    router = ModelRouter(prefer_cost_efficient=prefer_fast)
    return router.select_model(TaskType.COMPRESSION)


def get_reconstruction_model(high_quality: bool = True) -> ModelEndpoint:
    """Get optimal model for memory reconstruction."""
    router = ModelRouter(prefer_cost_efficient=not high_quality)
    return router.select_model(
        TaskType.RECONSTRUCTION,
        quality_threshold=0.9 if high_quality else 0.7
    )


def get_reasoning_model() -> ModelEndpoint:
    """Get optimal model for complex reasoning tasks."""
    router = ModelRouter()
    return router.select_model(TaskType.REASONING)
