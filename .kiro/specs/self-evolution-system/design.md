# Design: Phase 9 Self-Evolution System Completion

## 1. System Architecture

The Self-Evolution System transforms "Cognitive Dissonance" (uncertainty) into "Crystallized Intelligence" (LoRA Skill Cards).

### 1.1 Data Flow
1.  **Inference Time**: User Query -> `LoRARouter` checks confidence.
2.  **Trigger**: If `max(confidence) < THRESHOLD`, create `TrainingTask` -> Push to `SkillFactory` Queue.
3.  **Distillation**: `SkillFactory` Worker picks task -> Calls `CloudDistiller` -> Generates `List[QAPair]`.
4.  **Training**: `SkillFactory` passes `QAPair`s to `LoRATrainer` -> `LoRATrainer` fine-tunes base model -> Saves `new_skill.lora.arrow`.
5.  **Deployment**: `LoRAManager` detects new file -> Hot-swaps into `ArrowEngine`.

## 2. Component Design

### 2.1 Real LoRA Training Loop (`lora_trainer.py`)
Instead of the current `torch.randn` mock, we implement a standard PyTorch training loop tailored for low-rank adapters.

-   **Class**: `LoRATrainer`
-   **Method**: `train_qa(base_model, qa_pairs: List[QAPair], output_path: str)`
-   **Logic**:
    1.  **Data Prep**: Tokenize `QAPair`s into `(input_ids, attention_mask, labels)`.
    2.  **Model Prep**: Freeze base model weights. Inject `LoRALinear` layers (if not present) or activate existing ones. Set `requires_grad=True` ONLY for LoRA parameters.
    3.  **Optimization**: Initialize `torch.optim.AdamW` with LR=1e-4.
    4.  **Loop**:
        -   `optimizer.zero_grad()`
        -   `outputs = base_model(input_ids, labels=labels)`
        -   `loss = outputs.loss` (CrossEntropy)
        -   `loss.backward()`
        -   `optimizer.step()`
    5.  **Save**: Extract `lora_A` and `lora_B` weights -> Serialize to `.lora.arrow`.

### 2.2 Cloud Provider Factory (`skill_factory.py` & `cloud_distiller.py`)
Replace the `NotImplementedError` with a Factory Pattern.

-   **Interface**: `CloudProvider` (Abstract Base Class)
    -   `generate_qa(topic: str, n=5) -> List[QAPair]`
-   **Implementations**:
    -   `OpenAIProvider`: Uses `openai` lib (GPT-4o).
    -   `AnthropicProvider`: Uses `anthropic` lib (Claude 3.5 Sonnet).
    -   `LocalTeacherProvider`: Uses a larger local model (e.g., Qwen-72B) via vLLM (optional).
-   **Factory Method**: `get_cloud_provider(config: Config) -> CloudProvider`
    -   Reads `config.evolution.provider_type` (default: "openai").
    -   Instantiates appropriate class with API keys.

### 2.3 Dissonance Detector (`inference/lora_router.py`)
Integrated into the inference hot-path.

-   **Logic**:
    ```python
    scores = self.router_model(embedding)
    confidence, best_skill = torch.max(scores, dim=1)
    if confidence < self.dissonance_threshold:
        self.event_bus.emit("cognitive_dissonance", {
            "query": user_query,
            "confidence": confidence.item()
        })
    ```
-   **Async Handling**: The event must be handled asynchronously to not block the user's response (fallback to general chat).

## 3. Data Structures

### 3.1 Training Task
```python
@dataclass
class TrainingTask:
    id: str
    trigger_query: str      # The user input that caused dissonance
    domain: str             # Inferred domain (e.g., "coding", "medical")
    priority: int           # 1 (Low) to 5 (Critical)
    status: TaskStatus      # PENDING, DISTILLING, TRAINING, COMPLETED
```

### 3.2 LoRA Card Format (Existing)
Re-use `LoRAFormat` from `inference/lora_format.py`. Ensure the `LoRATrainer` output matches the `save` method signature.
