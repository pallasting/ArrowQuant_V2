"""
AI-OS LoRA Trainer for Diffusion Architecture

Simplified Phase 0 version for L2 evolution level.
Implements lightweight, on-device skill training using LoRA adapters.

Features (Phase 0):
- Basic LoRA training infrastructure
- Supervised fine-tuning on QA pairs
- Parameter-efficient adaptation

Future (Phase 2+):
- Integration with EvolutionRouter
- Uncertainty-driven training
- Skill card management

Architecture: 🧠 Python Brain (evolution layer)
"""

import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ai_os_diffusion.utils.logger import logger
from ai_os_diffusion.utils.errors import DiffusionError


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["query", "key", "value", "dense"])


@dataclass
class LoRACard:
    """LoRA skill card"""
    name: str
    rank: int
    alpha: float
    target_modules: List[str]
    weights_A: Dict[str, torch.Tensor]
    weights_B: Dict[str, torch.Tensor]
    metadata: Dict[str, str] = field(default_factory=dict)


class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer
    
    Implements low-rank adaptation: W' = W + BA
    where A is (rank, in_features) and B is (out_features, rank)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return original_output + lora_output * self.scaling


class QADataset(Dataset):
    """Simple QA dataset for LoRA training"""
    
    def __init__(self, qa_pairs: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.qa_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.qa_pairs[idx]
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
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }


class LoRATrainer:
    """
    Lightweight LoRA Trainer for L2 evolution level
    
    Phase 0: Basic training infrastructure
    Phase 2+: Integration with EvolutionRouter and uncertainty estimation
    
    Training capabilities:
    - Supervised fine-tuning on QA pairs
    - Parameter-efficient adaptation
    - Skill card export
    
    Architecture: 🧠 Python Brain (evolution layer)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        output_dir: str = "./lora_skills",
        config: Optional[LoRAConfig] = None
    ):
        """
        Initialize LoRA trainer
        
        Args:
            model: Base model to adapt
            tokenizer: Tokenizer for text encoding
            output_dir: Output directory for skill cards
            config: LoRA configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or LoRAConfig()
        
        self.device = next(model.parameters()).device
        
        # Prepare model for training
        self._prepare_model_for_training()
        
        logger.info(
            f"LoRATrainer initialized: rank={self.config.rank}, "
            f"alpha={self.config.alpha}, output_dir={output_dir}"
        )
    
    def _prepare_model_for_training(self) -> None:
        """
        Freeze base model and inject trainable LoRA adapters
        """
        logger.info("Preparing model for LoRA training...")
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Inject LoRA layers
        self.lora_layers = []
        
        def replace_linear_with_lora(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Linear):
                    # Check if this is a target layer
                    is_target = any(t in name for t in self.config.target_modules)
                    
                    if is_target:
                        logger.debug(f"Injecting LoRA into: {full_name}")
                        
                        # Create LoRA wrapper
                        lora_layer = LoRALinear(
                            original_layer=child,
                            rank=self.config.rank,
                            alpha=self.config.alpha,
                            dropout=self.config.dropout
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
        
        replace_linear_with_lora(self.model)
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(
            f"Model prepared. Trainable: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        logger.info(f"Injected {len(self.lora_layers)} LoRA layers")
    
    def train_qa(
        self,
        qa_pairs: List[Dict[str, str]],
        skill_name: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 3e-4
    ) -> Optional[LoRACard]:
        """
        Train a new LoRA skill from QA pairs
        
        Args:
            qa_pairs: List of {"q": ..., "a": ...} dicts
            skill_name: Name for the new skill
            epochs: Number of training passes
            batch_size: Batch size for training
            learning_rate: Learning rate
        
        Returns:
            The trained LoRACard
        
        Raises:
            DiffusionError: If training fails
        """
        if not qa_pairs:
            logger.warning("No data provided for training")
            return None
        
        logger.info(f"Starting training for '{skill_name}' with {len(qa_pairs)} examples")
        start_time = time.time()
        
        try:
            # Setup data
            dataset = QADataset(qa_pairs, self.tokenizer)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizer (only optimize LoRA parameters)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=learning_rate)
            
            # Loss function
            loss_fn = nn.CrossEntropyLoss()
            
            # Training loop
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                steps = 0
                
                for batch in loader:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Compute loss (causal LM objective)
                    # Shift logits and labels for next token prediction
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    steps += 1
                
                avg_loss = total_loss / steps if steps > 0 else 0
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Export LoRA card
            card = self._export_lora_card(skill_name, avg_loss, epochs)
            
            # Save card
            save_path = self.output_dir / f"{skill_name}.lora.pt"
            torch.save({
                'name': card.name,
                'rank': card.rank,
                'alpha': card.alpha,
                'target_modules': card.target_modules,
                'weights_A': card.weights_A,
                'weights_B': card.weights_B,
                'metadata': card.metadata,
            }, save_path)
            
            duration = time.time() - start_time
            logger.info(f"Training complete: {skill_name} in {duration:.1f}s")
            
            return card
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise DiffusionError(f"Training failed: {e}") from e
    
    def _export_lora_card(
        self,
        skill_name: str,
        final_loss: float,
        epochs: int
    ) -> LoRACard:
        """
        Export trained LoRA weights as a skill card
        
        Args:
            skill_name: Skill name
            final_loss: Final training loss
            epochs: Number of epochs trained
        
        Returns:
            LoRACard with trained weights
        """
        weights_A = {}
        weights_B = {}
        
        for name, layer in self.lora_layers:
            # Clean name for storage
            clean_name = name.replace("model.", "")
            
            weights_A[clean_name] = layer.lora_A.detach().cpu()
            weights_B[clean_name] = layer.lora_B.detach().cpu()
        
        card = LoRACard(
            name=skill_name,
            rank=self.config.rank,
            alpha=self.config.alpha,
            target_modules=list(weights_A.keys()),
            weights_A=weights_A,
            weights_B=weights_B,
            metadata={
                "description": f"Trained skill {skill_name}",
                "epochs": str(epochs),
                "training_loss": f"{final_loss:.4f}",
                "source": "LoRATrainer",
                "architecture": "diffusion"
            }
        )
        
        return card
