# Phase 2.0 Architecture Design

**Version**: 1.0  
**Date**: 2026-02-16  
**Status**: Approved  
**Type**: Self-Organizing Cognitive Loop System

---

## Executive Summary

Phase 2.0 transforms the memory compression system (Phase 1.0/1.1) into a complete **Self-Organizing Cognitive Loop** - an AI system that can perceive, remember, learn, express, and self-correct like a biological cognitive system.

### Core Innovation

> **We don't build a "memory system", we build a "cognitive system"**  
> **We don't preset architecture, we let the system self-organize**  
> **We stand on giants' shoulders (pre-trained models), only learn connections**

---

## Design Philosophy

### Biological Inspiration

```
Human Cognition:
👁️ Perception → 🧠 Encoding → 💭 Working Memory → 🗄️ Long-term Memory → 📈 Learning

Our System:
📥 Multi-modal Input → 🗜️ Compression → 🌐 Self-organizing Network → 🎨 Expression → 🔄 Feedback
```

### Key Principles

1. **Minimalism**: Build memory primitives, let system self-organize
2. **Standing on Giants**: Use pre-trained models (BERT/CLIP/Whisper)
3. **Self-Organization**: No preset structure, learn from data
4. **Closed Loop**: Continuous learning from feedback
5. **Multi-Modal**: Text, image, audio input/output

---

## System Architecture

### 5-Layer Cognitive Loop

```
┌─────────────────────────────────────────────────────────────┐
│                  1️⃣ Perception Layer                         │
│                  (感知层 - Pre-trained Models)                │
├─────────────────────────────────────────────────────────────┤
│  Input: Physical World (text/image/audio/video)             │
│  Models: BERT, CLIP, Whisper                                │
│  Output: Feature Vectors                                    │
│  Status: ✅ Available (use existing models)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  2️⃣ Compression Layer                        │
│                  (压缩层 - Phase 1.1 Complete)                │
├─────────────────────────────────────────────────────────────┤
│  Input: Feature Vectors                                     │
│  Process: Deduplication, Diff Extraction, Compression       │
│  Output: Memory Primitives                                  │
│  Performance: 2800x compression, < 1ms retrieval            │
│  Status: ✅ Complete                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  3️⃣ Learning Layer                           │
│                  (学习层 - Phase 2.0 Core)                    │
├─────────────────────────────────────────────────────────────┤
│  Input: Memory Primitives                                   │
│  Process: Self-organizing Connections, Network Building     │
│  Output: Knowledge Network                                  │
│  Algorithm: Hebbian Learning + Causal Learning              │
│  Status: ⚠️ Phase 2.0 (Week 1-3)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  4️⃣ Expression Layer                         │
│                  (表达层 - Phase 2.0 New)                     │
├─────────────────────────────────────────────────────────────┤
│  Input: Knowledge Network + Query                           │
│  Process: Retrieval → Reconstruction → Generation           │
│  Output: Physical World (text/image/audio)                  │
│  Models: GPT/Llama, DALL-E/SD, TTS                          │
│  Status: ⚠️ Phase 2.0 (Week 1)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  5️⃣ Feedback Layer                           │
│                  (反馈层 - Phase 2.0 New)                     │
├─────────────────────────────────────────────────────────────┤
│  Input: Output + Feedback Signals                           │
│  Process: Internal (self-correction) + External (user)      │
│  Output: Learning Signals → Back to Learning Layer          │
│  Status: ⚠️ Phase 2.0 (Week 2-3)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         └──────> Loop back to Learning Layer
```

---

## Core Components

### 1. Memory Primitive (记忆基元)

**The fundamental unit of the system**

```python
@dataclass
class MemoryPrimitive:
    """Minimal memory unit - foundation of everything"""
    
    # Core data
    id: str                          # Unique identifier
    content: CompressedMemory        # Compressed content (Phase 1.1)
    embedding: np.ndarray            # Feature vector
    
    # Self-organizing properties
    connections: Dict[str, float]    # {memory_id: connection_strength}
    activation: float = 0.0          # Current activation level
    
    # Statistics (for learning)
    access_count: int = 0            # How many times accessed
    success_count: int = 0           # How many times successfully used
    last_access: datetime = None     # Last access timestamp
    created_at: datetime             # Creation timestamp
```

**Design Rationale**:
- Minimal structure (only essential fields)
- Self-organizing (connections learned, not preset)
- Statistics-driven (learn from usage patterns)

### 2. Connection Learner (连接学习器)

**Learns connections between memories**

```python
class ConnectionLearner:
    """Learn memory connections - core algorithm"""
    
    def learn_connection(
        self,
        memory_a: MemoryPrimitive,
        memory_b: MemoryPrimitive,
        context: Context
    ) -> float:
        """
        Learn if two memories should connect and connection strength
        
        Learning Rules (Hebbian-like):
        - Co-activation → Strengthen connection
        - Causal relationship → Strengthen connection
        - Content similarity → Strengthen connection
        - Long-term unused → Decay connection
        """
        
        # 1. Co-occurrence learning (accessed together)
        co_activation = self._calculate_co_activation(memory_a, memory_b)
        
        # 2. Similarity learning (similar content)
        similarity = cosine_similarity(memory_a.embedding, memory_b.embedding)
        
        # 3. Causal learning (A causes B)
        causality = self._detect_causality(memory_a, memory_b, context)
        
        # 4. Success learning (connection leads to success)
        success_rate = self._calculate_success_rate(memory_a, memory_b)
        
        # Weighted combination
        connection_strength = (
            0.3 * co_activation +
            0.3 * similarity +
            0.2 * causality +
            0.2 * success_rate
        )
        
        return connection_strength
```

**Learning Algorithms**:
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Causal Learning**: Detect temporal causality (A → B)
- **Success Learning**: Reinforce connections that lead to success
- **Decay**: Prune unused connections (synaptic pruning)

### 3. Network Navigator (网络导航器)

**Navigate the self-organized network**

```python
class NetworkNavigator:
    """Navigate memory network - retrieval algorithm"""
    
    def retrieve(
        self,
        query: str,
        max_results: int = 10
    ) -> List[MemoryPrimitive]:
        """
        Retrieve relevant memories from self-organized network
        
        Not simple vector search, but:
        1. Find initial activation points
        2. Spread activation along connections
        3. Collect highly activated memories
        """
        
        # 1. Initial activation (vector similarity)
        query_embedding = self._embed(query)
        initial_memories = self._find_similar(query_embedding, top_k=5)
        
        # 2. Activation spreading (along connections)
        activated_memories = self._spread_activation(
            initial_memories,
            max_hops=3,   # Max 3 layers
            decay=0.7     # 30% decay per layer
        )
        
        # 3. Sort and return
        return sorted(
            activated_memories,
            key=lambda m: m.activation,
            reverse=True
        )[:max_results]
```

**Retrieval Algorithm**:
- **Activation Spreading**: Like neural network propagation
- **Multi-hop**: Follow connections up to 3 layers
- **Decay**: Activation decays with distance
- **Ranking**: By final activation level

### 4. Multi-Modal Expressor (多模态表达器)

**Generate multi-modal outputs**

```python
class MultiModalExpressor:
    """Multi-modal expression layer"""
    
    def express(
        self,
        memories: List[MemoryPrimitive],
        modality: str,
        style: Optional[str] = None
    ) -> Any:
        """Generate multi-modal output from memories"""
        
        # 1. Reconstruct memory content
        reconstructed = self.reconstruct(memories)
        
        # 2. Generate based on modality
        if modality == "text":
            return self.text_generator.generate(reconstructed, style)
        elif modality == "image":
            return self.image_generator.generate(reconstructed, style)
        elif modality == "audio":
            return self.audio_generator.generate(reconstructed, style)
        elif modality == "video":
            return self.generate_video(reconstructed, style)
```

**Supported Modalities**:
- **Text**: GPT/Llama for text generation
- **Image**: DALL-E/Stable Diffusion for image generation
- **Audio**: TTS/Music generation
- **Video**: Combined image + audio

### 5. Feedback System (反馈系统)

**Dual feedback mechanism**

```python
class FeedbackLayer:
    """Dual feedback: internal + external"""
    
    def __init__(self):
        self.internal = InternalFeedbackSystem()  # Self-correction
        self.external = ExternalFeedbackSystem()  # User feedback
    
    def process_feedback(
        self,
        output: Any,
        internal_feedback: InternalFeedback,
        external_feedback: Optional[ExternalFeedback]
    ) -> LearningSignal:
        """Process dual feedback and generate learning signal"""
        
        # Internal feedback: self-correction
        quality_score = self.internal.evaluate_quality(output)
        if quality_score < threshold:
            correction = self.internal.self_correct(output)
        
        # External feedback: cognitive adjustment
        if external_feedback:
            adjustment = self.external.adjust_cognition(external_feedback)
        
        # Generate learning signal
        return LearningSignal(
            correction=correction,
            adjustment=adjustment,
            target_memories=self.identify_targets()
        )
```

**Feedback Types**:
- **Internal Feedback**: Quality checks, consistency, completeness
- **External Feedback**: User satisfaction, corrections, preferences

---

## Key Algorithms

### Hebbian Learning

```python
def hebbian_learning(memory_a: MemoryPrimitive, memory_b: MemoryPrimitive):
    """Neurons that fire together, wire together"""
    
    if co_activated(memory_a, memory_b):
        # Strengthen connection
        strength = memory_a.connections.get(memory_b.id, 0.0)
        memory_a.connections[memory_b.id] = min(1.0, strength + 0.1)
        memory_b.connections[memory_a.id] = min(1.0, strength + 0.1)
```

### Activation Spreading

```python
def spread_activation(
    initial_memories: List[MemoryPrimitive],
    max_hops: int = 3,
    decay: float = 0.7
) -> List[MemoryPrimitive]:
    """Spread activation along connections (like neural network)"""
    
    activated = {}
    queue = [(m, 1.0, 0) for m in initial_memories]  # (memory, activation, hop)
    
    while queue:
        memory, activation, hop = queue.pop(0)
        
        if hop > max_hops:
            continue
        
        # Accumulate activation
        if memory.id in activated:
            activated[memory.id].activation += activation
        else:
            memory.activation = activation
            activated[memory.id] = memory
        
        # Propagate to connected memories
        for connected_id, strength in memory.connections.items():
            if connected_id not in activated:
                connected_memory = get_memory(connected_id)
                new_activation = activation * strength * decay
                queue.append((connected_memory, new_activation, hop + 1))
    
    return list(activated.values())
```

### Connection Pruning

```python
def prune_connections(memory: MemoryPrimitive, threshold: float = 0.1):
    """Prune weak connections (synaptic pruning)"""
    
    for conn_id, strength in list(memory.connections.items()):
        # Decay over time
        strength *= 0.99
        
        # Remove if too weak
        if strength < threshold:
            del memory.connections[conn_id]
        else:
            memory.connections[conn_id] = strength
```

---

## Performance Targets

### Quality Metrics

| Metric | Phase 1.1 | Phase 2.0 Target | Measurement |
|--------|-----------|------------------|-------------|
| Reconstruction quality | 0.101 | > 0.85 | QualityEvaluator |
| Keyword retention | 0% | > 90% | Entity matching |
| Semantic similarity | 0.003 | > 0.85 | Embedding cosine |
| Compression ratio | 2800x | > 10x (maintain) | Size ratio |

### Learning Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Connection accuracy | > 80% | Relevant memories connected |
| Retrieval relevance | > 0.85 | Retrieved result quality |
| Network density | 5-10 conn/memory | Not too sparse/dense |
| Convergence speed | < 100 interactions | Network stabilization |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Compression latency | < 10s | Time to compress |
| Retrieval latency | < 100ms | Time to retrieve |
| Expression latency | < 5s | Time to generate output |
| Feedback latency | < 1s | Time to process feedback |

---

## Advantages

### 1. Standing on Giants' Shoulders

```
✅ Perception: Pre-trained models (BERT/CLIP/Whisper)
✅ Compression: 2800x compression (Phase 1.1)
✅ Learning: Only learn connections (low cost)
```

### 2. Self-Organization = Zero Preset

```
❌ No preset: Knowledge graph structure
❌ No preset: Classification taxonomy
❌ No preset: Routing rules
✅ Let data and usage patterns decide everything
```

### 3. Continuous Evolution

```
Day 1: Random connections
Day 7: Discover common paths
Day 30: Form knowledge clusters
Day 90: Self-organized hierarchy
```

### 4. Minimal Implementation

```
Core code: < 2,200 LOC
Core concepts: 5 layers
Core algorithms: Hebbian + Activation Spreading
```

---

## Comparison

### vs Traditional ML

| Aspect | Traditional ML | Our System |
|--------|---------------|------------|
| Training data | Large dataset required | Learn from usage |
| Training cost | High (GPU hours) | Low (only connections) |
| Structure | Fixed after training | Dynamic self-organization |
| Adaptation | Requires retraining | Continuous learning |

### vs Knowledge Graphs

| Aspect | Knowledge Graphs | Our System |
|--------|-----------------|------------|
| Structure | Preset ontology | Self-organized |
| Connections | Manual/rule-based | Learned from data |
| Evolution | Manual updates | Automatic evolution |
| Flexibility | Rigid schema | Adaptive structure |

### vs Agent Frameworks

| Aspect | LangChain/AutoGPT | Our System |
|--------|-------------------|------------|
| Memory | Short-term only | Long-term + compressed |
| Learning | No learning | Continuous learning |
| Feedback | No feedback loop | Dual feedback |
| Evolution | Static | Self-improving |

---

## Risk Assessment

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Reconstructor bug complexity | High | Allocate 2x time buffer |
| Network convergence issues | Medium | Implement monitoring |
| Multi-modal generation quality | Medium | Use proven models |
| Feedback loop instability | Low | Add damping mechanisms |

### Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Task dependencies | Medium | Sequential development |
| Integration complexity | Medium | Incremental integration |
| Performance degradation | Low | Continuous benchmarking |

---

## Success Criteria

### Phase 2.0 Complete When:

1. ✅ Reconstructor returns full text (quality > 0.85)
2. ✅ Memory network self-organizes (connections learned)
3. ✅ Multi-modal expression works (text/image/audio)
4. ✅ Dual feedback loop functional (internal + external)
5. ✅ Continuous learning demonstrated (improves over time)
6. ✅ All 13 tasks complete and validated

### Demonstration Scenarios

**Scenario 1: Learning from Interaction**
```
Day 1: "What is quantum computing?" → Basic answer
Day 7: "Quantum entanglement?" → Recalls previous context
Day 30: "Applications?" → Suggests based on learned interests
```

**Scenario 2: Self-Correction**
```
Output: Incorrect answer
Internal Feedback: Quality score 0.6 → Self-correct
External Feedback: User correction → Update connections
Next time: Correct answer
```

**Scenario 3: Multi-Modal Expression**
```
Input: "Explain photosynthesis"
Output (text): Detailed explanation
Output (image): Diagram of the process
Output (audio): Narrated explanation
```

---

## Future Extensions (Phase 3.0+)

- Distributed processing (multi-GPU, multi-node)
- Advanced reasoning (chain-of-thought, planning)
- Meta-learning (learn how to learn)
- Collaborative learning (multi-agent knowledge sharing)

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-16  
**Status**: Approved for Implementation
