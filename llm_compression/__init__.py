"""
LLM 集成压缩系统

AI-OS 记忆系统 Phase 1 的核心组件，通过大语言模型实现 10-50x 的记忆压缩比。
"""

__version__ = "1.0.0"
__author__ = "AI-OS Team"

from llm_compression.config import Config
from llm_compression.logger import setup_logger
from llm_compression.llm_client import (
    LLMClient,
    LLMResponse,
    LLMAPIError,
    LLMTimeoutError,
    RetryPolicy,
    RateLimiter,
    LLMConnectionPool
)
from llm_compression.model_selector import (
    ModelSelector,
    MemoryType,
    QualityLevel,
    ModelConfig,
    ModelStats
)
from llm_compression.quality_evaluator import (
    QualityEvaluator,
    QualityMetrics
)
from llm_compression.compressor import (
    LLMCompressor,
    CompressedMemory,
    CompressionMetadata,
    CompressionError
)
from llm_compression.reconstructor import (
    LLMReconstructor,
    ReconstructedMemory,
    ReconstructionError
)
from llm_compression.memory_primitive import (
    MemoryPrimitive
)
from llm_compression.connection_learner import (
    ConnectionLearner
)
from llm_compression.expression_layer import (
    MultiModalExpressor,
    ExpressionResult
)
from llm_compression.internal_feedback import (
    InternalFeedbackSystem,
    QualityScore,
    Correction,
    CorrectionType
)
from llm_compression.network_navigator import (
    NetworkNavigator,
    ActivationResult
)
from llm_compression.cognitive_loop import (
    CognitiveLoop,
    CognitiveResult
)
from llm_compression.conversation_memory import (
    ConversationMemory,
    ConversationTurn
)
from llm_compression.personalization import (
    PersonalizationEngine,
    UserProfile
)
from llm_compression.conversational_agent import (
    ConversationalAgent,
    AgentResponse
)
from llm_compression.visualizer import (
    MemoryVisualizer
)
from llm_compression.arrow_storage import (
    ArrowStorage,
    StorageError,
    SCHEMA_REGISTRY,
    create_experiences_compressed_schema,
    create_identity_compressed_schema,
    create_preferences_compressed_schema,
    create_context_compressed_schema,
    create_summary_table_schema
)
from llm_compression.arrow_native_compressor import ArrowNativeCompressor
from llm_compression.openclaw_interface import (
    OpenClawMemoryInterface
)
from llm_compression.batch_processor import (
    BatchProcessor,
    BatchProgress,
    CompressionCache
)
from llm_compression.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics
)
from llm_compression.embedding_provider import (
    EmbeddingProvider,
    ArrowEngineProvider,
    LocalEmbedderProvider,
    get_default_provider,
    reset_provider,
)

__all__ = [
    "Config",
    "setup_logger",
    "LLMClient",
    "LLMResponse",
    "LLMAPIError",
    "LLMTimeoutError",
    "RetryPolicy",
    "RateLimiter",
    "LLMConnectionPool",
    "ModelSelector",
    "MemoryType",
    "QualityLevel",
    "ModelConfig",
    "ModelStats",
    "QualityEvaluator",
    "QualityMetrics",
    "LLMCompressor",
    "CompressedMemory",
    "CompressionMetadata",
    "CompressionError",
    "LLMReconstructor",
    "ReconstructedMemory",
    "ReconstructionError",
    "MemoryPrimitive",
    "ConnectionLearner",
    "MultiModalExpressor",
    "ExpressionResult",
    "InternalFeedbackSystem",
    "QualityScore",
    "Correction",
    "CorrectionType",
    "NetworkNavigator",
    "ActivationResult",
    "CognitiveLoop",
    "CognitiveResult",
    "ConversationMemory",
    "ConversationTurn",
    "PersonalizationEngine",
    "UserProfile",
    "ConversationalAgent",
    "AgentResponse",
    "MemoryVisualizer",
    "ArrowStorage",
    "StorageError",
    "SCHEMA_REGISTRY",
    "create_experiences_compressed_schema",
    "create_identity_compressed_schema",
    "create_preferences_compressed_schema",
    "create_context_compressed_schema",
    "create_summary_table_schema",
    "OpenClawMemoryInterface",
    "BatchProcessor",
    "BatchProgress",
    "CompressionCache",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "EmbeddingProvider",
    "ArrowEngineProvider",
    "LocalEmbedderProvider",
    "get_default_provider",
    "reset_provider",
    "ArrowNativeCompressor",
]
