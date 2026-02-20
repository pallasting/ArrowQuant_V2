"""
Natural Language Generation (NLG) module.

Provides text generation capabilities with multiple backends:
- OpenAI: GPT models for high-quality generation
- Anthropic: Claude models for nuanced responses
- Local: Ollama-based local models
- Template: Fallback template-based generation

Public API:
    - NLGEngine: Main NLG engine with backend abstraction
    - TemplateEngine: Template-based fallback generation
    - NLGError: NLG-specific exceptions
"""

from .nlg_engine import NLGEngine, NLGError, TemplateEngine, ConversationHistory

__all__ = [
    "NLGEngine",
    "NLGError", 
    "TemplateEngine",
    "ConversationHistory"
]

__version__ = "0.1.0"
