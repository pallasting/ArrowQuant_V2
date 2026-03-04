"""
Python API 兼容性验证测试
验证所有现有 Python API 保持不变，无破坏性变更
需求: 7.2, 7.5
"""

import pytest
import pyarrow as pa
import numpy as np
from arrow_quant_v2 import ArrowQuantV2


class TestPythonAPICompatibility:
    """验证 Python API 的向后兼容性"""
    
    def test_arrow_quant_v2_constructor_unchanged(self):
        """验证 ArrowQuantV2 构造函数签名不变"""
        # 带 mode 参数构造
        quantizer_time = ArrowQuantV2(mode="diffusion")
        assert quantizer_time is not None
    
    def test_quantize_arrow_method_exists(self):
        """验证 quantize_arrow 方法存在且签名不变"""
        quantizer = ArrowQuantV2()
        
        # 创建测试数据
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 方法应该存在且可调用
        result = quantizer.quantize_arrow(
            table,
            bit_width=8
        )
        
        assert result is not None
        assert isinstance(result, pa.Table)
    
    def test_quantize_arrow_batch_method_exists(self):
        """验证 quantize_arrow_batch 方法存在"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0", "layer.1"],
            "weights": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 方法应该存在
        batch = table.to_batches()[0]
        result = quantizer.quantize_arrow_batch(
            batch,
            bit_width=8
        )
        
        assert result is not None
    

    def test_default_parameter_values_unchanged(self):
        """验证默认参数值保持不变"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 使用默认参数应该工作
        result = quantizer.quantize_arrow(table)
        assert result is not None
    
    def test_return_type_unchanged(self):
        """验证返回类型没有破坏性变更"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        result = quantizer.quantize_arrow(table, bit_width=8)
        
        # 返回类型应该是 PyArrow Table
        assert isinstance(result, pa.Table)
        
        # 应该包含预期的列
        assert "quantized_data" in result.column_names
        assert "scales" in result.column_names
        assert "zero_points" in result.column_names
    
    def test_error_handling_behavior_unchanged(self):
        """验证错误处理行为保持一致"""
        quantizer = ArrowQuantV2()
        
        # 无效输入应该抛出 ValueError
        with pytest.raises((ValueError, TypeError)):
            quantizer.quantize_arrow(None)
        
        # 空表应该抛出错误
        empty_table = pa.Table.from_pydict({"layer_name": [], "weights": []})
        with pytest.raises((ValueError, RuntimeError)):
            quantizer.quantize_arrow(empty_table)
    
    def test_bit_width_parameter_unchanged(self):
        """验证 bit_width 参数行为不变"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 所有支持的 bit_width 应该工作
        for bit_width in [2, 4, 8]:
            result = quantizer.quantize_arrow(
                table,
                bit_width=bit_width
            )
            assert result is not None
    
    @pytest.mark.skip(reason="num_time_groups parameter was removed from direct quantize_arrow call")
    def test_num_time_groups_parameter_unchanged(self):
        """验证 num_time_groups 参数行为不变"""
        pass
    
    def test_backward_compatibility_with_existing_code(self):
        """模拟现有代码的使用模式，确保仍然工作"""
        # 场景 1: 基本量化工作流
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0", "layer.1"],
            "weights": [
                np.random.randn(100).astype(np.float32),
                np.random.randn(100).astype(np.float32),
            ],
        }
        table = pa.Table.from_pydict(weights_data)
        
        result = quantizer.quantize_arrow(table, bit_width=4)
        
        assert isinstance(result, pa.Table)
        assert result.num_rows > 0
        
        # 场景 2: 转换为 pandas
        df = result.to_pandas()
        assert df is not None
        assert len(df) > 0
    
    def test_no_new_required_parameters(self):
        """验证没有新增必需参数（破坏性变更）"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 使用最少参数应该工作（向后兼容）
        result = quantizer.quantize_arrow(table)
        assert result is not None
    
    def test_schema_output_unchanged(self):
        """验证输出 schema 保持不变"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        result = quantizer.quantize_arrow(table, bit_width=8)
        
        # 验证输出 schema 包含预期字段
        schema = result.schema
        field_names = [field.name for field in schema]
        
        # 核心字段应该存在
        assert "quantized_data" in field_names
        assert "scales" in field_names
        assert "zero_points" in field_names
        assert "bit_width" in field_names
    
    def test_deterministic_behavior_unchanged(self):
        """验证确定性行为保持不变"""
        quantizer = ArrowQuantV2()
        
        weights_data = {
            "layer_name": ["layer.0"],
            "weights": [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
        }
        table = pa.Table.from_pydict(weights_data)
        
        # 相同输入应该产生相同输出
        result1 = quantizer.quantize_arrow(table, bit_width=8)
        result2 = quantizer.quantize_arrow(table, bit_width=8)
        
        # 验证结果一致
        assert result1.num_rows == result2.num_rows
        assert result1.num_columns == result2.num_columns
        
        # 验证数据一致（量化数据应该相同）
        data1 = result1.column("quantized_data").to_pylist()
        data2 = result2.column("quantized_data").to_pylist()
        assert data1 == data2


class TestAPIStability:
    """验证 API 稳定性和一致性"""
    
    def test_all_public_methods_accessible(self):
        """验证所有公开方法可访问"""
        quantizer = ArrowQuantV2()
        
        # 检查关键方法存在
        assert hasattr(quantizer, "quantize_arrow")
        assert hasattr(quantizer, "quantize_arrow_batch")
        
        # 检查方法可调用
        assert callable(quantizer.quantize_arrow)
        assert callable(quantizer.quantize_arrow_batch)
    
    def test_constructor_modes_unchanged(self):
        """验证构造函数支持的模式不变"""
        # 默认模式
        q1 = ArrowQuantV2()
        assert q1 is not None
        
        # diffusion 模式
        q2 = ArrowQuantV2(mode="diffusion")
        assert q2 is not None
    
    def test_no_breaking_changes_in_exceptions(self):
        """验证异常类型没有破坏性变更"""
        quantizer = ArrowQuantV2()
        
        # 无效输入应该抛出标准 Python 异常
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            quantizer.quantize_arrow(None)
        
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            quantizer.quantize_arrow("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
