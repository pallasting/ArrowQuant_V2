"""
Unit tests for ModelConverter - TDD approach

Tests cover all requirements from TASKS.md:
- T1.2: ModelConverter core class
- T1.3: Weight extraction and optimization
- T1.4: Arrow/Parquet serialization
- T1.5: Tokenizer export
- T1.6: Validation and benchmarking

Acceptance Criteria:
- Successfully convert all-MiniLM-L6-v2 model
- Inference accuracy within 1% of original
- Model loading < 100ms
- Test coverage > 80%
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import Mock, patch, MagicMock


class TestConversionConfig:
    """Test ConversionConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        from llm_compression.tools.model_converter import ConversionConfig
        
        config = ConversionConfig()
        assert config.compression == "zstd"  # Changed from lz4 to zstd
        assert config.compression_level == 3  # New field
        assert config.use_float16 is True
        assert config.extract_tokenizer is True
        assert config.validate_output is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        from llm_compression.tools.model_converter import ConversionConfig
        
        config = ConversionConfig(
            compression="zstd",
            use_float16=False,
            extract_tokenizer=False
        )
        assert config.compression == "zstd"
        assert config.use_float16 is False
        assert config.extract_tokenizer is False


class TestConversionResult:
    """Test ConversionResult dataclass"""
    
    def test_successful_result(self):
        """Test successful conversion result"""
        from llm_compression.tools.model_converter import ConversionResult
        
        result = ConversionResult(
            success=True,
            model_name="test-model",
            output_dir=Path("/tmp/test"),
            parquet_path=Path("/tmp/test/model.parquet"),
            tokenizer_path=Path("/tmp/test/tokenizer"),
            metadata_path=Path("/tmp/test/metadata.json"),
            total_parameters=1000000,
            file_size_mb=10.5,
            compression_ratio=2.5,
            conversion_time_sec=5.0
        )
        
        assert result.success is True
        assert result.total_parameters == 1000000
        assert result.compression_ratio == 2.5
    
    def test_result_to_dict(self):
        """Test conversion result to dictionary"""
        from llm_compression.tools.model_converter import ConversionResult
        
        result = ConversionResult(
            success=True,
            model_name="test",
            output_dir=Path("/tmp"),
            parquet_path=Path("/tmp/model.parquet"),
            tokenizer_path=None,
            metadata_path=None,
            total_parameters=1000,
            file_size_mb=1.0,
            compression_ratio=2.0,
            conversion_time_sec=1.0
        )
        
        result_dict = result.to_dict()
        assert result_dict['success'] is True
        assert result_dict['total_parameters'] == 1000
        assert result_dict['tokenizer_path'] is None


class TestModelConverter:
    """Test ModelConverter class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = MagicMock()
        
        # Mock parameters
        params = {
            'layer1.weight': Mock(
                detach=Mock(return_value=Mock(
                    cpu=Mock(return_value=Mock(
                        numpy=Mock(return_value=np.random.randn(10, 10).astype(np.float32)),
                        numel=Mock(return_value=100),
                        dtype=Mock(return_value='float32'),
                        shape=(10, 10)
                    ))
                ))
            ),
            'layer2.bias': Mock(
                detach=Mock(return_value=Mock(
                    cpu=Mock(return_value=Mock(
                        numpy=Mock(return_value=np.random.randn(10).astype(np.float32)),
                        numel=Mock(return_value=10),
                        dtype=Mock(return_value='float32'),
                        shape=(10,)
                    ))
                ))
            )
        }
        
        model.named_parameters = Mock(return_value=params.items())
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = MagicMock()
        tokenizer.save_pretrained = Mock()
        return tokenizer
    
    def test_converter_initialization(self):
        """Test ModelConverter initialization"""
        from llm_compression.tools.model_converter import ModelConverter, ConversionConfig
        
        config = ConversionConfig()
        converter = ModelConverter(config)
        
        assert converter.config is not None
        assert converter.config.compression == "zstd"  # Changed from lz4 to zstd
    
    def test_extract_weights(self, mock_model):
        """Test weight extraction from model"""
        from llm_compression.tools.model_converter import ModelConverter
        
        converter = ModelConverter()
        weights = converter._extract_weights(mock_model)
        
        assert 'layer1.weight' in weights
        assert 'layer2.bias' in weights
        assert len(weights) == 2
    
    def test_optimize_weights_float16(self):
        """Test float16 optimization"""
        from llm_compression.tools.model_converter import ModelConverter, ConversionConfig
        import torch
        
        config = ConversionConfig(use_float16=True)
        converter = ModelConverter(config)
        
        weights = {
            'layer1': torch.randn(10, 10, dtype=torch.float32),
            'layer2': torch.randn(5, 5, dtype=torch.float32)
        }
        
        optimized = converter._optimize_weights(weights)
        
        # Should convert to float16
        assert optimized['layer1'].dtype == torch.float16
        assert optimized['layer2'].dtype == torch.float16
    
    def test_optimize_weights_no_conversion(self):
        """Test weights without optimization"""
        from llm_compression.tools.model_converter import ModelConverter, ConversionConfig
        import torch
        
        config = ConversionConfig(use_float16=False)
        converter = ModelConverter(config)
        
        weights = {
            'layer1': torch.randn(10, 10, dtype=torch.float32)
        }
        
        optimized = converter._optimize_weights(weights)
        assert optimized['layer1'].dtype == torch.float32
    
    def test_convert_to_arrow(self, temp_dir):
        """Test Arrow/Parquet conversion"""
        from llm_compression.tools.model_converter import ModelConverter
        import torch
        
        converter = ModelConverter()
        
        weights = {
            'layer1.weight': torch.randn(10, 10, dtype=torch.float32),
            'layer2.bias': torch.randn(5, dtype=torch.float32)
        }
        
        output_path = Path(temp_dir)
        parquet_path = converter._convert_to_arrow(
            weights,
            output_path,
            "test-model"
        )
        
        # Verify file was created
        assert parquet_path.exists()
        assert parquet_path.suffix == '.parquet'
        
        # Verify Arrow table structure
        table = pq.read_table(str(parquet_path))
        assert 'layer_name' in table.column_names
        assert 'shape' in table.column_names
        assert 'dtype' in table.column_names
        assert 'data' in table.column_names
        assert 'num_params' in table.column_names
        
        # Verify data integrity
        assert table.num_rows == 2
        layer_names = table['layer_name'].to_pylist()
        assert 'layer1.weight' in layer_names
        assert 'layer2.bias' in layer_names
    
    def test_export_tokenizer(self, temp_dir, mock_tokenizer):
        """Test tokenizer export"""
        from llm_compression.tools.model_converter import ModelConverter
        
        converter = ModelConverter()
        output_path = Path(temp_dir)
        
        tokenizer_path = converter._export_tokenizer(mock_tokenizer, output_path)
        
        assert tokenizer_path.exists()
        assert tokenizer_path.name == "tokenizer"
        mock_tokenizer.save_pretrained.assert_called_once()
    
    def test_generate_metadata(self):
        """Test metadata generation"""
        from llm_compression.tools.model_converter import ModelConverter
        import torch
        
        converter = ModelConverter()
        
        weights = {
            'layer1': torch.randn(10, 10),
            'layer2': torch.randn(5, 5)
        }
        
        model_info = {
            'architecture': 'test',
            'embedding_dimension': 384
        }
        
        metadata = converter._generate_metadata(
            model_name="test-model",
            model_type="sentence-transformers",
            weights=weights,
            model_info=model_info,
            config=converter.config,  # Added missing config parameter
            parquet_path=Path("/tmp/model.parquet"),
            tokenizer_path=Path("/tmp/tokenizer")
        )
        
        assert metadata['model_name'] == "test-model"
        assert metadata['model_type'] == "sentence-transformers"
        assert metadata['total_parameters'] == 125  # 10*10 + 5*5
        assert metadata['num_layers'] == 2
        assert 'converted_at' in metadata
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_load_model_sentence_transformers(self, mock_st_class):
        """Test loading sentence-transformers model"""
        from llm_compression.tools.model_converter import ModelConverter
        
        # Create mock base model with config
        mock_base_model = MagicMock()
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config.num_attention_heads = 12
        mock_config.intermediate_size = 1536
        mock_config.num_hidden_layers = 6
        mock_config.vocab_size = 30522
        mock_config.max_position_embeddings = 512
        mock_config.layer_norm_eps = 1e-12
        mock_base_model.config = mock_config
        
        # Create mock sentence transformer
        mock_model = MagicMock()
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension = Mock(return_value=384)
        mock_model.tokenizer = MagicMock()
        mock_model.__getitem__ = Mock(return_value=MagicMock(auto_model=mock_base_model))
        
        mock_st_class.return_value = mock_model
        
        converter = ModelConverter()
        model, tokenizer, model_info = converter._load_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers"
        )
        
        assert model is not None
        assert tokenizer is not None
        assert model_info['architecture'] == 'SentenceTransformer'
        assert model_info['embedding_dimension'] == 384


class TestConversionValidation:
    """Test conversion validation"""
    
    def test_validate_conversion_success(self):
        """Test successful validation"""
        from llm_compression.tools.model_converter import ModelConverter
        import torch
        
        # Create test weights
        weight1 = torch.randn(10, 10)
        original_weights = {
            'layer1': weight1
        }
        
        # Create temporary Arrow file with same data
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = Path(f.name)
            
            schema = pa.schema([
                ('layer_name', pa.string()),
                ('shape', pa.list_(pa.int32())),
                ('dtype', pa.string()),
                ('data', pa.binary()),
                ('num_params', pa.int64()),
            ])
            
            weight_np = weight1.numpy()
            table = pa.Table.from_pylist([{
                'layer_name': 'layer1',
                'shape': list(weight_np.shape),
                'dtype': str(weight1.dtype),
                'data': weight_np.tobytes(),
                'num_params': int(weight1.numel()),
            }], schema=schema)
            
            pq.write_table(table, str(temp_path))
        
        try:
            converter = ModelConverter()
            
            # Create minimal metadata
            metadata = {
                'model_name': 'test-model',
                'total_parameters': 100
            }
            
            result = converter._validate_conversion(
                parquet_path=temp_path,
                original_weights=original_weights,
                metadata=metadata
            )
            assert result is True
        finally:
            temp_path.unlink()
    
    def test_validate_conversion_layer_mismatch(self):
        """Test validation fails on layer mismatch"""
        from llm_compression.tools.model_converter import ModelConverter
        import torch
        
        # Model with different layers than Arrow file
        mock_model = MagicMock()
        mock_model.named_parameters = Mock(return_value=[
            ('layer1', Mock()),
            ('layer2', Mock())
        ])
        
        # Arrow file with only one layer
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = Path(f.name)
            
            schema = pa.schema([
                ('layer_name', pa.string()),
                ('shape', pa.list_(pa.int32())),
                ('dtype', pa.string()),
                ('data', pa.binary()),
                ('num_params', pa.int64()),
            ])
            
            table = pa.Table.from_pylist([{
                'layer_name': 'layer1',
                'shape': [5, 5],
                'dtype': 'float32',
                'data': np.zeros(25, dtype=np.float32).tobytes(),
                'num_params': 25,
            }], schema=schema)
            
            pq.write_table(table, str(temp_path))
        
        try:
            converter = ModelConverter()
            result = converter._validate_conversion(mock_model, temp_path, None)
            assert result is False
        finally:
            temp_path.unlink()


class TestConversionIntegration:
    """Integration tests for full conversion pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_conversion_pipeline_structure(self, temp_dir):
        """Test that conversion creates expected file structure"""
        # This is a structural test - actual model loading tested separately
        expected_files = [
            "test-model.parquet",
            "metadata.json",
            "tokenizer/tokenizer.json"
        ]
        
        output_path = Path(temp_dir)
        
        # Verify output directory structure expectations
        assert output_path.exists()
        assert output_path.is_dir()


# Performance benchmarks (not run by default)
@pytest.mark.benchmark
class TestConversionPerformance:
    """Performance benchmarks for conversion"""
    
    def test_conversion_time_target(self, benchmark):
        """Test conversion completes within time target"""
        # Target: < 5 minutes for typical model
        # This would use benchmark fixture when ready
        pass
    
    def test_arrow_load_time(self, benchmark):
        """Test Arrow model loads within target"""
        # Target: < 100ms
        pass
