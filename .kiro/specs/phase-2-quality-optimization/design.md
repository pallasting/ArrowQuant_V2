# Design Document: Phase 2.0 Quality Optimization + Advanced Features

## Overview

Phase 2.0 transforms the LLM compression system from a proof-of-concept to a production-ready solution. The design addresses critical quality issues from Phase 1.1 (0.101 quality score, 0% keyword retention, empty reconstruction bug), implements intelligent adaptive compression, enables multi-model ensemble capabilities, and achieves seamless OpenClaw integration.

### Design Goals

1. **Quality**: Fix reconstruction bugs and achieve 0.85+ quality score
2. **Intelligence**: Implement adaptive compression with context awareness
3. **Scalability**: Enable multi-model ensemble for optimal quality-speed tradeoffs
4. **Integration**: Production-ready OpenClaw Memory system integration
5. **Observability**: Comprehensive monitoring and performance profiling

### Non-Goals (Phase 3.0)

- Distributed processing across multiple nodes
- Multi-GPU parallelization
- Horizontal scaling infrastructure
- Real-time streaming compression

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenClaw Application                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              OpenClawMemoryAdapter                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Memory Interface → Compression API Mapping      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│           Adaptive Compression Engine                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Content    │  │   Quality    │  │   Model      │ │
│  │  Classifier  │→ │   Selector   │→ │   Router     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Model Ensemble                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ TinyLlama│  │ Gemma3   │  │ Qwen2.5  │             │
│  │ (Fast)   │  │(Balanced)│  │ (High)   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Fixed LLMReconstructor                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Summary Expansion → Diff Application → Verify   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Module Organization

Phase 2.0 is organized into four modules:

1. **Quality Fixes Module** (Week 1): Core bug fixes and quality improvements
2. **Adaptive Compression Module** (Week 2): Context-aware and mode-based compression
3. **Multi-Model Ensemble Module** (Week 3): Intelligent model selection and ensemble
4. **OpenClaw Integration Module** (Week 4): Production deployment and integration

## Components and Interfaces

### 1. Fixed LLMReconstructor

**Purpose**: Reconstruct original text from compressed memory (fixes empty text bug)

**Interface**:
```python
class LLMReconstructor:
    def reconstruct(self, compressed: CompressedMemory) -> str:
        """Reconstruct full text from compressed memory"""
        
    def _expand_summary(self, summary: str, entities: List[Entity]) -> str:
        """Expand semantic summary to full text using LLM"""
        
    def _apply_diff(self, expanded: str, diff: bytes) -> str:
        """Apply diff to correct details"""
        
    def _validate_reconstruction(self, reconstructed: str, original_length: int) -> bool:
        """Validate reconstruction quality"""
```

**Key Changes**:
- Fix `_expand_summary()` to properly generate text from summary
- Add timeout handling and retry logic for LLM calls
- Implement fallback reconstruction using diff-only approach
- Add comprehensive validation before returning results

### 2. Enhanced Summary Generator

**Purpose**: Generate high-quality semantic summaries using LLM

**Interface**:
```python
class SummaryGenerator:
    def generate(self, text: str, max_tokens: int = 100) -> Summary:
        """Generate semantic summary using LLM"""
        
    def validate_quality(self, summary: Summary, original: str) -> float:
        """Validate summary quality score"""
        
    def optimize_prompt(self, text: str, content_type: ContentType) -> str:
        """Generate optimized prompt based on content type"""
```

**Implementation**:
- Use structured prompts optimized for different content types
- Implement timeout handling (5s default, configurable)
- Add quality validation before accepting summary
- Fallback to improved extraction strategy (not simple truncation)

### 3. NER-Based Entity Extractor

**Purpose**: Extract entities using advanced NER models

**Interface**:
```python
class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with spaCy or transformers NER model"""
        
    def extract(self, text: str) -> List[Entity]:
        """Extract entities with confidence scores"""
        
    def extract_by_type(self, text: str, entity_types: List[str]) -> Dict[str, List[Entity]]:
        """Extract specific entity types"""
```

**Entity Types**:
- PERSON: Names of people
- ORG: Organizations
- GPE: Geopolitical entities (locations)
- DATE: Dates and times
- CARDINAL: Numbers
- Custom types via configuration

### 4. Compression Mode Selector

**Purpose**: Implement quality-speed tradeoff modes

**Interface**:
```python
class CompressionMode(Enum):
    FAST = "fast"        # < 3s, TinyLlama
    BALANCED = "balanced" # < 10s, Gemma3
    HIGH = "high"        # < 20s, Qwen2.5-7B

class ModeSelector:
    def select_model(self, mode: CompressionMode) -> str:
        """Select model based on compression mode"""
        
    def get_parameters(self, mode: CompressionMode) -> CompressionParams:
        """Get compression parameters for mode"""
```

**Mode Configuration**:
- Fast: TinyLlama, max_tokens=50, temperature=0.7
- Balanced: Gemma3, max_tokens=100, temperature=0.5
- High: Qwen2.5-7B, max_tokens=150, temperature=0.3

### 5. Content Classifier

**Purpose**: Classify content type for context-aware compression

**Interface**:
```python
class ContentType(Enum):
    CODE = "code"
    CONVERSATION = "conversation"
    DOCUMENTATION = "documentation"

class ContentClassifier:
    def classify(self, text: str) -> ContentType:
        """Classify content type with confidence"""
        
    def get_compression_strategy(self, content_type: ContentType) -> CompressionStrategy:
        """Return optimal strategy for content type"""
```

**Classification Heuristics**:
- Code: High ratio of special characters, indentation patterns, keywords
- Conversation: Speaker markers, question patterns, informal language
- Documentation: Structured headings, formal language, lists

### 6. Incremental Update Manager

**Purpose**: Support efficient updates without full recompression

**Interface**:
```python
class IncrementalUpdateManager:
    def can_update_incrementally(self, compressed: CompressedMemory, new_content: str) -> bool:
        """Determine if incremental update is feasible"""
        
    def update(self, compressed: CompressedMemory, new_content: str) -> CompressedMemory:
        """Perform incremental update"""
        
    def compute_diff(self, original: str, new_content: str) -> bytes:
        """Compute diff for incremental update"""
```

**Update Strategy**:
- Append-only updates: Extend summary and add new entities
- Modification updates: Recompute affected sections only
- Threshold: If changes > 30%, trigger full recompression

### 7. Model Ensemble Framework

**Purpose**: Combine multiple models for improved quality

**Interface**:
```python
class EnsembleStrategy(Enum):
    PARALLEL = "parallel"      # All models run in parallel
    SEQUENTIAL = "sequential"  # Models run in sequence
    VOTING = "voting"         # Majority voting on outputs

class EnsembleCompressor:
    def __init__(self, models: List[str], strategy: EnsembleStrategy):
        """Initialize ensemble with models and strategy"""
        
    async def compress(self, text: str, mode: CompressionMode) -> CompressedMemory:
        """Compress using ensemble of models"""
        
    def aggregate_results(self, results: List[CompressionResult]) -> CompressedMemory:
        """Aggregate results from multiple models"""
```

**Ensemble Roles**:
- Model A (Gemma3): Generate semantic summary
- Model B (Qwen2.5): Extract entities and key information
- Model C (Llama3.1): Validate quality and consistency

### 8. Intelligent Model Router

**Purpose**: Automatically select optimal model based on content

**Interface**:
```python
class ModelRouter:
    def route(self, text: str, requirements: CompressionRequirements) -> str:
        """Select best model for given text and requirements"""
        
    def evaluate_routing_rules(self, text: str) -> Dict[str, float]:
        """Evaluate all routing rules and return scores"""
```

**Routing Rules**:
1. Text length < 500 chars → TinyLlama (fast)
2. Text length > 2000 chars → Qwen2.5 (quality)
3. Code content → Specialized code model
4. Latency budget < 5s → TinyLlama
5. Quality requirement > 0.9 → Ensemble mode

### 9. Model Performance Profiler

**Purpose**: Track and analyze model performance metrics

**Interface**:
```python
class ModelProfiler:
    def track_compression(self, model: str, latency: float, quality: float, cost: float):
        """Track compression metrics for model"""
        
    def get_model_stats(self, model: str) -> ModelStats:
        """Get performance statistics for model"""
        
    def compare_models(self, models: List[str]) -> pd.DataFrame:
        """Compare performance across models"""
        
    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
```

**Tracked Metrics**:
- Latency (p50, p95, p99)
- Quality score (mean, std)
- Cost per compression
- Success rate
- Model utilization

### 10. OpenClaw Memory Adapter

**Purpose**: Integrate with OpenClaw Memory interface

**Interface**:
```python
class OpenClawMemoryAdapter:
    def store(self, memory: OpenClawMemory) -> str:
        """Store memory with automatic compression"""
        
    def retrieve(self, memory_id: str) -> OpenClawMemory:
        """Retrieve and decompress memory"""
        
    def search(self, query: str, top_k: int = 10) -> List[OpenClawMemory]:
        """Search compressed memories"""
        
    def update(self, memory_id: str, updates: Dict) -> OpenClawMemory:
        """Update existing memory (incremental)"""
```

**Mapping**:
- OpenClawMemory.content → CompressedMemory.summary + diff
- OpenClawMemory.metadata → CompressedMemory.compression_metadata
- OpenClawMemory.embedding → Preserved unchanged

### 11. API Compatibility Layer

**Purpose**: Expose compression services via REST and CLI

**REST API Endpoints**:
```
POST /api/v1/compress
  Body: { "text": str, "mode": str, "content_type": str }
  Response: { "compressed_id": str, "compression_ratio": float, "quality": float }

POST /api/v1/decompress
  Body: { "compressed_id": str }
  Response: { "text": str, "quality_score": float }

GET /api/v1/status
  Response: { "models": List[str], "health": str, "metrics": Dict }

POST /api/v1/batch/compress
  Body: { "texts": List[str], "mode": str }
  Response: { "results": List[CompressionResult] }
```

**CLI Commands**:
```bash
llm-compress compress --input file.txt --mode balanced --output compressed.bin
llm-compress decompress --input compressed.bin --output reconstructed.txt
llm-compress status --models
llm-compress benchmark --dataset test_data/
```

## Data Models

### CompressedMemory (Enhanced)

```python
@dataclass
class CompressedMemory:
    # Core fields
    memory_id: str
    summary: str
    summary_hash: str
    entities: List[Entity]
    diff_data: bytes
    
    # Phase 2.0 additions
    content_type: ContentType
    compression_mode: CompressionMode
    model_used: str
    quality_score: float
    
    # Metadata
    original_size: int
    compressed_size: int
    compression_ratio: float
    timestamp: datetime
    version: int  # For incremental updates
```

### Entity (Enhanced)

```python
@dataclass
class Entity:
    text: str
    type: str  # PERSON, ORG, GPE, DATE, CARDINAL
    start: int
    end: int
    confidence: float  # 0.0 - 1.0
    metadata: Dict[str, Any]
```

### CompressionRequirements

```python
@dataclass
class CompressionRequirements:
    mode: CompressionMode
    content_type: Optional[ContentType]
    quality_threshold: float = 0.85
    latency_budget: float = 10.0  # seconds
    max_cost: Optional[float] = None
```

### ModelStats

```python
@dataclass
class ModelStats:
    model_name: str
    total_compressions: int
    avg_latency: float
    p95_latency: float
    avg_quality: float
    avg_cost: float
    success_rate: float
    last_updated: datetime
```

## Correctness Properties

