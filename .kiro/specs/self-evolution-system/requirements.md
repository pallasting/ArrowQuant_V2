# Requirements: Phase 9 Self-Evolution System Completion

## 1. Overview
Phase 9 "Self-Evolving Intelligence" has a functional data flow but lacks the core "brain" — the actual LoRA training loop is currently mocked in `lora_trainer.py`. This phase aims to replace the scaffolding with a real, working implementation of LoRA fine-tuning and Cloud Teacher distillation.

## 2. Core Components to Complete

### 2.1 Real LoRA Training Loop (`lora_trainer.py`)
- **Current State**: Uses `torch.randn` dummy gradients and simulated optimization.
- **Requirement**: Implement a true training loop using PyTorch.
  - **Forward Pass**: Must use `inference.lora_layer.LoRALinear` for active adapters.
  - **Backward Pass**: Must compute gradients for `lora_A` and `lora_B` matrices only (freeze base model).
  - **Loss Function**: Use `CrossEntropyLoss` on the next-token prediction task.
  - **Optimization**: Use `AdamW` optimizer for LoRA parameters.

### 2.2 Dynamic Cloud Teacher Instantiation (`skill_factory.py`)
- **Current State**: Raises `NotImplementedError` when trying to instantiate a real cloud provider.
- **Requirement**: Implement the factory logic to instantiate `OpenAIProvider` or `AnthropicProvider` based on configuration.
  - **Configuration**: Load API keys and endpoints from `config.yaml`.
  - **Fallback**: Support failover to a different provider if one is unavailable.

### 2.3 Cognitive Dissonance Trigger (`inference/lora_router.py`)
- **Current State**: Routing logic exists but doesn't trigger learning.
- **Requirement**: Implement the "Dissonance Detector".
  - **Confidence Threshold**: If the highest routing probability < `0.4` (configurable), trigger a "Learning Event".
  - **Queueing**: Push the low-confidence query to `SkillFactory`'s task queue.

## 3. Functional Requirements

### 3.1 The "Learning Loop"
1. **Detect**: User asks a question -> Router confidence low -> Trigger Dissonance.
2. **Ask**: `SkillFactory` picks up task -> Instantiates `CloudDistiller` -> Queries Cloud Teacher for QA pairs.
3. **Train**: `LoRATrainer` loads base model -> Injects new LoRA adapter -> Trains on QA pairs -> Saves `.lora.arrow`.
4. **Deploy**: `LoRAManager` hot-loads the new `.lora.arrow` -> System now answers the original question confidently.

### 3.2 Performance Constraints
- **Training Time**: < 5 minutes for a specific skill (on CPU/Consumer GPU).
- **Memory Overhead**: Training must not exceed 8GB VRAM (use gradient checkpointing if needed).
- **Storage**: Each `.lora.arrow` card should be < 10MB.

## 4. Acceptance Criteria
- [ ] `lora_trainer.py` successfully reduces loss on a dummy dataset.
- [ ] `skill_factory.py` successfully calls OpenAI/Anthropic API to get QA pairs.
- [ ] The system can automatically generate a `.lora.arrow` file from a "hard" question without human intervention.
- [ ] The generated LoRA card, when loaded, allows the local model to answer the "hard" question correctly.
