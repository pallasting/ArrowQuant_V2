"""
Unit tests for INT2 bit packing/unpacking.

Tests the ArrowQuantizer._pack_int2 and _unpack_int2 static methods
to ensure correct bit packing as per ArrowQuant design.
"""

import numpy as np
import pytest

from llm_compression.inference.arrow_quantizer import ArrowQuantizer


class TestINT2Packing:
    """Test INT2 bit packing and unpacking."""
    
    def test_pack_unpack_roundtrip(self):
        """Test that pack -> unpack is lossless."""
        # Test data: [-2, -1, 0, 1]
        data = np.array([-2, -1, 0, 1], dtype=np.int8)
        
        # Pack
        packed = ArrowQuantizer._pack_int2(data)
        
        # Unpack
        unpacked = ArrowQuantizer._unpack_int2(packed, len(data))
        
        # Verify roundtrip
        np.testing.assert_array_equal(unpacked, data)
    
    def test_pack_format(self):
        """Test that packing follows the correct bit layout."""
        # [0, 1, -1, -2] -> [2, 3, 1, 0] (after +2 mapping)
        # byte = val_0 | (val_1 << 2) | (val_2 << 4) | (val_3 << 6)
        # byte = 2 | (3 << 2) | (1 << 4) | (0 << 6)
        # byte = 0b00_01_11_10 = 0x1E = 30
        data = np.array([0, 1, -1, -2], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        assert len(packed) == 1
        assert packed[0] == 0x1E  # 0b00_01_11_10
    
    def test_pack_multiple_bytes(self):
        """Test packing data that spans multiple bytes."""
        # 8 values = 2 bytes
        data = np.array([0, 1, -1, -2, 1, 0, -2, -1], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        assert len(packed) == 2
        
        # Verify unpacking
        unpacked = ArrowQuantizer._unpack_int2(packed, len(data))
        np.testing.assert_array_equal(unpacked, data)
    
    def test_pack_with_padding(self):
        """Test packing data that requires padding."""
        # 5 values -> needs padding to 8 (2 bytes)
        data = np.array([1, 0, -1, -2, 1], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        # Should be 2 bytes (5 values + 3 padding)
        assert len(packed) == 2
        
        # Unpack only the original 5 values
        unpacked = ArrowQuantizer._unpack_int2(packed, len(data))
        np.testing.assert_array_equal(unpacked, data)
    
    def test_pack_large_array(self):
        """Test packing a large array."""
        # 1000 values
        np.random.seed(42)
        data = np.random.randint(-2, 2, size=1000, dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        # Should be 250 bytes (1000 / 4)
        assert len(packed) == 250
        
        # Verify roundtrip
        unpacked = ArrowQuantizer._unpack_int2(packed, len(data))
        np.testing.assert_array_equal(unpacked, data)
    
    def test_pack_all_values(self):
        """Test packing all possible INT2 values."""
        # All possible values: [-2, -1, 0, 1]
        data = np.array([-2, -1, 0, 1, -2, -1, 0, 1], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        unpacked = ArrowQuantizer._unpack_int2(packed, len(data))
        
        np.testing.assert_array_equal(unpacked, data)
    
    def test_compression_ratio(self):
        """Test that packing achieves 4x compression."""
        # 1024 values
        data = np.random.randint(-2, 2, size=1024, dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        # Original: 1024 bytes (int8)
        # Packed: 256 bytes (4 values per byte)
        assert len(data) == 1024
        assert len(packed) == 256
        assert len(data) / len(packed) == 4.0
    
    def test_unpack_exact_size(self):
        """Test unpacking with exact size specification."""
        data = np.array([1, 0, -1, -2, 1, 0, -1], dtype=np.int8)  # 7 values
        
        packed = ArrowQuantizer._pack_int2(data)
        
        # Unpack exactly 7 values (not 8)
        unpacked = ArrowQuantizer._unpack_int2(packed, 7)
        
        assert len(unpacked) == 7
        np.testing.assert_array_equal(unpacked, data)
    
    def test_pack_empty_array(self):
        """Test packing an empty array."""
        data = np.array([], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        assert len(packed) == 0
    
    def test_pack_single_value(self):
        """Test packing a single value."""
        data = np.array([1], dtype=np.int8)
        
        packed = ArrowQuantizer._pack_int2(data)
        
        # Should be 1 byte (1 value + 3 padding)
        assert len(packed) == 1
        
        unpacked = ArrowQuantizer._unpack_int2(packed, 1)
        np.testing.assert_array_equal(unpacked, data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
