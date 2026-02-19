# Phase 2.0 API Reference

**Version**: 1.0  
**Date**: 2026-02-16

---

## Core APIs

### 1. MemoryPrimitive

```python
@dataclass
class MemoryPrimitive:
    """Fundamental memory unit"""
    
    # Attributes
    id: str
    content: CompressedMemory
    embedding: np.ndarray
    connections: Dict[str, float]
    activation: float
    access_count: int
    success_count: int
    last_access: datetime
    created_at: datetime
    
    # Methods
    def activate(self, strength: float) -> None
    def record_success(self) -> None
    def get_success_rate(self) -> float
```

### 2. ConnectionLearner

```python
class ConnectionLearner:
    """Learn connections between memories"""
    
    def __init__(
        self,
        co_activation_weight: float = 0.3,
        similarity_weight: float = 0.3
    )
    
    def learn_connection(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive
    ) -> float
    
    def hebbian_learning(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive,
        learning_rate: float = 0.1
    ) -> None
    
    def causal_learning(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive,
        context: Context
    ) -> None
    
    def record_co_activation(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive
    ) -> None
```

### 3. NetworkNavigator

```python
class NetworkNavigator:
    """Navigate memory network"""
    
    def __init__(self, max_hops: int = 3, decay: float = 0.7)
    
    def retrieve(
        self,
        query: str,
        network: MemoryNetwork,
        max_results: int = 10
    ) -> List[MemoryPrimitive]
    
    def multi_path_retrieval(
        self,
        query: str,
        network: MemoryNetwork,
        max_results: int = 10
    ) -> List[MemoryPrimitive]
```

### 4. MultiModalExpressor

```python
class MultiModalExpressor:
    """Multi-modal expression"""
    
    def __init__(self)
    
    def express(
        self,
        memories: List[MemoryPrimitive],
        modality: str = "text",
        style: Optional[str] = None
    ) -> Any
```

### 5. FeedbackLayer

```python
class FeedbackLayer:
    """Dual feedback system"""
    
    def __init__(self)
    
    # Internal feedback
    class InternalFeedbackSystem:
        def evaluate_output(
            self,
            output: str,
            original_query: str,
            used_memories: List[MemoryPrimitive]
        ) -> QualityScore
        
        def self_correct(
            self,
            output: str,
            quality_score: QualityScore,
            used_memories: List[MemoryPrimitive]
        ) -> Optional[Correction]
    
    # External feedback
    class ExternalFeedbackSystem:
        def process_user_feedback(
            self,
            output: Any,
            feedback: UserFeedback,
            context: Context
        ) -> CognitiveAdjustment
```

### 6. CognitiveLoop

```python
class CognitiveLoop:
    """Complete cognitive loop"""
    
    def __init__(self)
    
    def process(
        self,
        input: Any,
        output_modality: str = "text"
    ) -> Tuple[Any, LearningSignal]
    
    def apply_external_feedback(
        self,
        learning_signal: LearningSignal,
        user_feedback: UserFeedback
    ) -> None
```

### 7. ContinuousLearningEngine

```python
class ContinuousLearningEngine:
    """Continuous learning engine"""
    
    def __init__(self)
    
    def interact(
        self,
        user_input: Any,
        output_modality: str = "text"
    ) -> Any
    
    def receive_feedback(self, feedback: UserFeedback) -> None
    
    def evolve(self) -> None
```

---

## Data Types

### UserFeedback

```python
@dataclass
class UserFeedback:
    type: str  # "positive", "negative", "corrective"
    intensity: float  # 0.0 - 1.0
    correct_answer: Optional[str] = None
    comment: Optional[str] = None
```

### QualityScore

```python
@dataclass
class QualityScore:
    overall: float
    consistency: float
    completeness: float
    accuracy: float
    coherence: float
```

### CognitiveAdjustment

```python
@dataclass
class CognitiveAdjustment:
    action: str  # "strengthen", "weaken", "learn", "update"
    targets: List[MemoryPrimitive]
    strength: float
    reason: str
    learning_need: Optional[str] = None
    correct_output: Optional[str] = None
```

### LearningSignal

```python
@dataclass
class LearningSignal:
    internal_quality: QualityScore
    used_memories: List[MemoryPrimitive]
    output: Any
    context: Context
```

---

## Usage Examples

### Basic Usage

```python
# Initialize engine
engine = ContinuousLearningEngine()

# Interact
output = engine.interact("What is quantum computing?")
print(output)

# Provide feedback
engine.receive_feedback(UserFeedback(
    type="positive",
    intensity=0.9
))
```

### Multi-Modal Expression

```python
# Text output
text = engine.interact("Explain photosynthesis", modality="text")

# Image output (if configured)
image = engine.interact("Draw a quantum computer", modality="image")

# Audio output (if configured)
audio = engine.interact("Narrate the explanation", modality="audio")
```

### Continuous Learning

```python
# Day 1
output1 = engine.interact("What is AI?")
engine.receive_feedback(UserFeedback(type="positive", intensity=0.8))

# Day 7 (system has learned)
output7 = engine.interact("How does AI learn?")
# System recalls previous context about AI

# Day 30 (system has evolved)
output30 = engine.interact("AI applications?")
# System suggests based on learned interests
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-16
