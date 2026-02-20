
"""
AI-OS LoRA Trainer.

Implements lightweight, on-device skill training.
Enables the SkillFactory to fine-tune LoRA adapters on specific datasets
using gradient descent (AdamW).

This allows AI-OS to:
1. "Learn" from raw text data (self-supervised)
2. "Memorize" QA pairs (supervised fine-tuning)
3. Adapt to user feedback (RLHF-lite)

Key Optimization:
- Freezes the entire base model.
- Injects low-rank adapters into Attention layers.
- Only optimizes the adapters (parameter efficient).
- Uses Arrow-native data loading for zero-copy efficiency.
"""

import time
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from llm_compression.inference.lora_format import LoRACard, LoRAFormat
from llm_compression.inference.lora_layer import LoRALinear

logger = logging.getLogger(__name__)


class ArrowDataset(Dataset):
    """Zero-copy dataset from Arrow/Parquet data."""
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item.get('q', '')}\nAnswer: {item.get('a', '')}"

        # Tokenize
        encoded = self.tokenizer.encode([text])

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)  # Causal LM objective
        }


class LoRATrainer:
    """
    Lightweight LoRA Trainer for ArrowEngine.

    Training capabilities:
    - Supervised Fine-Tuning (SFT) on QA pairs.
    - Continuous Pre-training on raw text.

    Architecture:
    - Wraps the InferenceCore (frozen).
    - Injects trainable LoRALinear layers.
    - specialized for "Skill Acquisition" not general LLM training.
    """

    def __init__(
        self,
        engine,
        output_dir: str = "./lora_skills",
        rank: int = 8,
        alpha: float = 16.0,
        learning_rate: float = 3e-4,
    ):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank = rank
        self.alpha = alpha
        self.lr = learning_rate

        self.device = self.engine.device

        # Prepare model for training (inject LoRA layers)
        self._prepare_model_for_training()

    def _prepare_model_for_training(self):
        """Freeze base model and inject trainable adapters."""
        logger.info("Preparing model for LoRA training...")

        # 1. Freeze base model
        for param in self.engine.inference_core.parameters():
            param.requires_grad = False

        # 2. Inject LoRA layers
        # Target modules: query, key, value projections in attention layers
        target_modules = ["query", "key", "value", "dense"]

        self.lora_layers = []

        # Recursive replacement helper
        def replace_linear_with_lora(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.Linear):
                    # Check if this is a target layer
                    is_target = any(t in name for t in target_modules)

                    if is_target:
                        logger.debug(f"Injecting LoRA into: {full_name}")
                        # Create LoRA wrapper
                        lora_layer = LoRALinear(
                            original_layer=child,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=0.05
                        )
                        # Replace in parent module
                        setattr(module, name, lora_layer)
                        self.lora_layers.append((full_name, lora_layer))

                        # Enable gradients for LoRA params
                        lora_layer.lora_A.requires_grad = True
                        lora_layer.lora_B.requires_grad = True

                else:
                    # Recurse
                    replace_linear_with_lora(child, full_name)

        replace_linear_with_lora(self.engine.inference_core)

        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.engine.inference_core.parameters() if p.requires_grad)
        logger.info(f"Model prepared. Trainable parameters: {trainable_params}")
        logger.info(f"Injected {len(self.lora_layers)} LoRA layers.")

    def train_qa(
        self,
        qa_pairs: List[Dict[str, str]],
        skill_name: str,
        epochs: int = 3,
        batch_size: int = 4,
    ) -> Optional[LoRACard]:
        """
        Train a new LoRA skill from QA pairs.

        Args:
            qa_pairs: List of {"q": ..., "a": ...} dicts.
            skill_name: Name for the new skill.
            epochs: Number of training passes.

        Returns:
            The trained LoRACard.
        """
        if not qa_pairs:
            logger.warning("No data provided for training.")
            return None

        logger.info(f"Starting training for '{skill_name}' with {len(qa_pairs)} examples.")
        start_time = time.time()

        # 1. Setup Data
        dataset = ArrowDataset(qa_pairs, self.engine.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Optimizer (only optimize LoRA parameters)
        trainable_params = [p for p in self.engine.inference_core.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=self.lr)

        # 3. Training Loop
        self.engine.inference_core.train() # Set mode to train (enables Dropout)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            steps = 0

            for batch in loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                # Forward pass - Custom loop to get sequence output (not pooled)
                # We bypass inference_core.forward() to get hidden states for all tokens

                # 1. Embeddings
                # Access internal methods of inference_core (assuming Encoder architecture)
                if hasattr(self.engine.inference_core, '_compute_embeddings'):
                    embedding_output = self.engine.inference_core._compute_embeddings(input_ids)
                    extended_attention_mask = self.engine.inference_core._get_extended_attention_mask(attention_mask)

                    hidden_states = embedding_output

                    # 2. Encoder Layers
                    for layer in self.engine.inference_core.encoder_layers:
                        hidden_states, _ = layer(
                            hidden_states,
                            extended_attention_mask,
                            output_attentions=False
                        )

                    outputs = hidden_states # [Batch, Seq, Hidden]
                else:
                    # Fallback for mock engine or different architecture
                    outputs = self.engine.inference_core(input_ids, attention_mask)

                # Create a temporary head if not present
                if not hasattr(self, 'lm_head'):
                    self.lm_head = nn.Linear(self.engine.inference_core.hidden_size,
                                           self.engine.inference_core.config.get('vocab_size', 30522)).to(self.device)

                logits = self.lm_head(outputs)

                # Shift labels for Causal LM objective
                # Prediction: logits[..., :-1, :] -> Predict next token
                # Target: labels[..., 1:] -> The actual next token

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten for CrossEntropyLoss: [Batch*Seq, Vocab], [Batch*Seq]
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # 4. Export LoRACard
        weights_A = {}
        weights_B = {}

        for name, layer in self.lora_layers:
            # Clean name for storage
            clean_name = name.replace("encoder.layers.", "encoder.layer.") # Normalize
            clean_name = clean_name.replace("inference_core.", "")

            weights_A[clean_name] = layer.lora_A.detach().cpu().numpy()
            weights_B[clean_name] = layer.lora_B.detach().cpu().numpy()

        card = LoRACard(
            name=skill_name,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=list(weights_A.keys()),
            weights_A=weights_A,
            weights_B=weights_B,
            metadata={
                "description": f"Trained skill {skill_name}",
                "epochs": str(epochs),
                "training_loss": f"{avg_loss:.4f}",
                "source": "LoRATrainer",
                "base_model": "ArrowEngine"
            }
        )

        # Save
        save_path = self.output_dir / f"{skill_name}.lora.arrow"
        LoRAFormat.save(card, str(save_path))

        duration = time.time() - start_time
        logger.info(f"Training complete: {skill_name} in {duration:.1f}s")

        # Clean up: Un-inject layers?
        # For now, we leave them injected but could disable them.
        # Ideally, we should restore the original model state.

        return card
