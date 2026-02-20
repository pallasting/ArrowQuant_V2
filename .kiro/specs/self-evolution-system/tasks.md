# Tasks: Phase 9 Self-Evolution System Completion

## Overview
Completion of the Self-Evolving Intelligence system by replacing mock implementations with real logic.

## Task List

### Task 9.4: Real LoRA Training Loop
- [ ] **Implementation**: Replace `lora_trainer.py` mock logic with real PyTorch training loop.
  - [ ] Implement `LoRADataset` class for tokenization.
  - [ ] Implement `train_qa` method with `CrossEntropyLoss` and `AdamW`.
  - [ ] Ensure `lora_A` and `lora_B` gradients are computed correctly.
  - [ ] Add `save_lora` method to persist trained weights to `.lora.arrow`.
- [ ] **Validation**: Run training on a small dummy dataset and verify loss decreases.

### Task 9.5: Cloud Provider Factory
- [ ] **Implementation**: Fix `skill_factory.py` and `cloud_distiller.py`.
  - [ ] Implement `OpenAIProvider` class using `openai` SDK.
  - [ ] Implement `AnthropicProvider` class using `anthropic` SDK.
  - [ ] Create `get_cloud_provider` factory method reading from `config.yaml`.
- [ ] **Validation**: Verify successful API calls to OpenAI/Anthropic (mocked for cost saving, or real test key).

### Task 9.6: Cognitive Dissonance Detector
- [ ] **Implementation**: Update `inference/lora_router.py`.
  - [ ] Add `dissonance_threshold` to config (default 0.4).
  - [ ] Implement logic to detect low-confidence routing.
  - [ ] Implement `_trigger_learning_event` method to push task to `SkillFactory`.
- [ ] **Validation**: Simulate low-confidence query and verify task creation in `SkillFactory`.

### Task 9.7: End-to-End Self-Evolution Test
- [ ] **Integration**: Connect all components.
  - [ ] Trigger: User query "Explain Quantum Computing to a 5yo" (unknown domain).
  - [ ] Distill: System calls Cloud Teacher -> Gets QA pairs.
  - [ ] Train: System trains `quantum_computing.lora.arrow`.
  - [ ] Deploy: System loads new skill.
  - [ ] Verify: System answers original query using local model + new LoRA.
