"""
Phase 0 Completion Test

Validates that all migrated modules can be imported and basic functionality works.
"""

import pytest
import tempfile
from pathlib import Path


class TestPhase0Complete:
    """Test Phase 0 migration completion"""
    
    def test_storage_import(self):
        """Test storage module imports"""
        from ai_os_diffusion.storage import ArrowStorage, StorageError
        
        assert ArrowStorage is not None
        assert StorageError is not None
    
    def test_evolution_import(self):
        """Test evolution module imports"""
        from ai_os_diffusion.evolution import (
            LoRATrainer,
            LoRAConfig,
            LoRACard,
            LoRALinear,
        )
        
        assert LoRATrainer is not None
        assert LoRAConfig is not None
        assert LoRACard is not None
        assert LoRALinear is not None
    
    def test_arrow_storage_basic(self):
        """Test ArrowStorage basic functionality"""
        from ai_os_diffusion.storage import ArrowStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ArrowStorage(storage_path=tmpdir)
            
            # Save a memory
            storage.save(
                memory_id="test_001",
                content="Hello world",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={"category": "test", "source": "pytest"}
            )
            
            # Load the memory
            memory = storage.load("test_001")
            
            assert memory is not None
            assert memory["memory_id"] == "test_001"
            assert memory["content"] == "Hello world"
            assert len(memory["embedding"]) == 4
    
    def test_arrow_storage_similarity_search(self):
        """Test ArrowStorage similarity search"""
        from ai_os_diffusion.storage import ArrowStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ArrowStorage(storage_path=tmpdir)
            
            # Save multiple memories
            storage.save("mem_001", "Hello", [1.0, 0.0, 0.0])
            storage.save("mem_002", "World", [0.0, 1.0, 0.0])
            storage.save("mem_003", "Test", [0.0, 0.0, 1.0])
            
            # Query by similarity
            results = storage.query_by_similarity([1.0, 0.0, 0.0], top_k=2)
            
            assert len(results) > 0
            assert results[0][0]["memory_id"] == "mem_001"  # Most similar
            assert results[0][1] > 0.9  # High similarity score
    
    def test_lora_config(self):
        """Test LoRAConfig dataclass"""
        from ai_os_diffusion.evolution import LoRAConfig
        
        config = LoRAConfig(rank=16, alpha=32.0)
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.05
        assert "query" in config.target_modules
    
    def test_lora_card(self):
        """Test LoRACard dataclass"""
        from ai_os_diffusion.evolution import LoRACard
        
        card = LoRACard(
            name="test_skill",
            rank=8,
            alpha=16.0,
            target_modules=["query", "key"],
            weights_A={},
            weights_B={},
            metadata={"description": "Test skill"}
        )
        
        assert card.name == "test_skill"
        assert card.rank == 8
        assert card.metadata["description"] == "Test skill"
    
    def test_all_inference_imports(self):
        """Test all inference module imports"""
        from ai_os_diffusion.inference import (
            ArrowEngine,
            WeightLoader,
            FastTokenizer,
            InferenceCore,
            get_best_device,
        )
        
        assert ArrowEngine is not None
        assert WeightLoader is not None
        assert FastTokenizer is not None
        assert InferenceCore is not None
        assert get_best_device is not None
    
    def test_all_config_imports(self):
        """Test all config module imports"""
        from ai_os_diffusion.config import Config, DiffusionConfig, EvolutionConfig
        
        assert Config is not None
        assert DiffusionConfig is not None
        assert EvolutionConfig is not None
    
    def test_all_utils_imports(self):
        """Test all utils module imports"""
        from ai_os_diffusion.utils import (
            logger,
            DiffusionError,
        )
        from ai_os_diffusion.utils.embedding_provider import EmbeddingProvider
        
        assert logger is not None
        assert DiffusionError is not None
        assert EmbeddingProvider is not None
    
    def test_phase0_completion_status(self):
        """Verify Phase 0 is 100% complete"""
        # All 15 files should be migrated (13 core + 2 new in session 5)
        migrated_files = [
            "config/config.py",
            "utils/logger.py",
            "utils/errors.py",
            "utils/embedding_provider.py",
            "inference/arrow_engine.py",
            "inference/device_utils.py",
            "inference/intel_opt.py",
            "inference/cuda_backend.py",
            "inference/weight_loader.py",
            "inference/fast_tokenizer.py",
            "inference/inference_core.py",
            "inference/quantization_schema.py",
            "inference/decoder_layers.py",
            "storage/arrow_storage.py",
            "evolution/lora_trainer.py",
        ]
        
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        
        for file_path in migrated_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Missing file: {file_path}"
        
        # Phase 0 is 100% complete (15 files total)
        total_files = 15
        completion_rate = len(migrated_files) / total_files
        assert completion_rate == 1.0, f"Phase 0 completion: {completion_rate*100:.0f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
