"""
GPU Backend Support Tests

Tests for multi-backend GPU support:
- NVIDIA CUDA
- AMD ROCm
- Intel XPU
- Apple MPS
- Vulkan
"""

import pytest
import torch


class TestGPUBackends:
    """Test GPU backend detection and functionality"""
    
    def test_device_detection(self):
        """Test automatic device detection"""
        from ai_os_diffusion.inference import get_best_device
        
        device = get_best_device()
        assert device in ["cuda", "xpu", "mps", "vulkan", "cpu"]
        print(f"Detected device: {device}")
    
    def test_device_info(self):
        """Test device information retrieval"""
        from ai_os_diffusion.inference import get_best_device, get_device_info
        
        device = get_best_device()
        info = get_device_info(device)
        
        assert "device_type" in info
        assert "backend" in info or "name" in info
        assert info["device_type"] == device
        
        print(f"\nDevice Info: {info}")
    
    def test_rocm_detection(self):
        """Test AMD ROCm detection"""
        from ai_os_diffusion.inference import is_rocm_platform
        
        is_rocm = is_rocm_platform()
        print(f"ROCm available: {is_rocm}")
        
        if is_rocm:
            # Additional ROCm tests
            assert torch.cuda.is_available()
            assert hasattr(torch.version, "hip")
            print(f"HIP Version: {torch.version.hip}")
    
    def test_vulkan_detection(self):
        """Test Vulkan backend detection"""
        from ai_os_diffusion.inference import is_vulkan_available
        
        is_vulkan = is_vulkan_available()
        print(f"Vulkan available: {is_vulkan}")
    
    def test_intel_xpu_detection(self):
        """Test Intel XPU detection"""
        from ai_os_diffusion.inference import is_intel_ipex_available
        
        has_ipex = is_intel_ipex_available()
        print(f"Intel IPEX available: {has_ipex}")
        
        if has_ipex:
            try:
                import intel_extension_for_pytorch as ipex
                print(f"IPEX version: {ipex.__version__}")
            except:
                pass
    
    def test_cuda_rocm_backend(self):
        """Test CUDA/ROCm backend"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/ROCm not available")
        
        from ai_os_diffusion.inference import get_device_info, is_rocm_platform
        
        info = get_device_info("cuda")
        
        assert "name" in info
        assert "memory_total_gb" in info
        
        if is_rocm_platform():
            assert info["backend"] == "ROCm (AMD)"
            assert "rocm_version" in info
            print(f"ROCm Device: {info['name']}")
            print(f"ROCm Version: {info['rocm_version']}")
        else:
            assert info["backend"] == "CUDA (NVIDIA)"
            print(f"CUDA Device: {info['name']}")
    
    def test_multi_gpu_detection(self):
        """Test multi-GPU detection"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/ROCm not available")
        
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        
        if device_count > 1:
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {name}")
    
    def test_print_device_info(self):
        """Test device info printing"""
        from ai_os_diffusion.inference import print_device_info
        
        # Should not raise any errors
        print_device_info()
    
    def test_move_to_device(self):
        """Test tensor movement to device"""
        from ai_os_diffusion.inference import get_best_device, move_to_device
        
        device = get_best_device()
        
        # Test with tensor
        tensor = torch.randn(2, 3)
        moved = move_to_device(tensor, device)
        assert str(moved.device).startswith(device.split(":")[0])
        
        # Test with dict
        data = {"a": torch.randn(2, 3), "b": torch.randn(3, 4)}
        moved_dict = move_to_device(data, device)
        assert str(moved_dict["a"].device).startswith(device.split(":")[0])
        
        # Test with list
        data_list = [torch.randn(2, 3), torch.randn(3, 4)]
        moved_list = move_to_device(data_list, device)
        assert str(moved_list[0].device).startswith(device.split(":")[0])


class TestROCmBackend:
    """Test AMD ROCm specific functionality"""
    
    def test_rocm_module_import(self):
        """Test ROCm module import"""
        try:
            from ai_os_diffusion.inference.rocm_backend import (
                is_rocm_available,
                get_rocm_info,
                ROCmOptimizer,
            )
            assert is_rocm_available is not None
            assert get_rocm_info is not None
            assert ROCmOptimizer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ROCm backend: {e}")
    
    def test_rocm_info(self):
        """Test ROCm information retrieval"""
        from ai_os_diffusion.inference.rocm_backend import get_rocm_info
        
        info = get_rocm_info()
        assert "available" in info
        
        if info["available"]:
            assert "hip_version" in info
            assert "device_count" in info
            print(f"ROCm Info: {info}")
    
    def test_rocm_optimizer(self):
        """Test ROCm optimizer"""
        from ai_os_diffusion.inference.rocm_backend import (
            is_rocm_available,
            ROCmOptimizer,
        )
        
        if not is_rocm_available():
            pytest.skip("ROCm not available")
        
        optimizer = ROCmOptimizer(device_id=0)
        assert optimizer is not None
        
        # Test batch size recommendation
        batch_size = optimizer.get_recommended_batch_size(
            model_size_mb=100,
            sequence_length=512
        )
        assert batch_size > 0
        print(f"Recommended batch size: {batch_size}")


class TestBackendCompatibility:
    """Test backend compatibility and fallback"""
    
    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU available"""
        from ai_os_diffusion.inference import get_device_info
        
        info = get_device_info("cpu")
        assert info["device_type"] == "cpu"
        assert "features" in info
        print(f"CPU Features: {info['features']}")
    
    def test_backend_priority(self):
        """Test backend selection priority"""
        from ai_os_diffusion.inference import get_best_device
        
        device = get_best_device()
        
        # Priority: CUDA/ROCm > XPU > MPS > Vulkan > CPU
        if torch.cuda.is_available():
            assert device == "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            assert device == "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device in ["vulkan", "cpu"]
    
    def test_all_backends_info(self):
        """Print information for all available backends"""
        from ai_os_diffusion.inference import get_device_info
        
        backends = ["cuda", "xpu", "mps", "vulkan", "cpu"]
        
        print("\n" + "="*60)
        print("Available GPU Backends")
        print("="*60)
        
        for backend in backends:
            try:
                info = get_device_info(backend)
                if "error" not in info:
                    print(f"\n{backend.upper()}:")
                    print(f"  Backend: {info.get('backend', 'N/A')}")
                    print(f"  Name: {info.get('name', 'N/A')}")
                    if "memory_total_gb" in info:
                        print(f"  Memory: {info['memory_total_gb']:.2f} GB")
            except Exception as e:
                print(f"\n{backend.upper()}: Not available ({e})")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
