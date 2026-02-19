"""
Environment Check Test

Check Python version, dependencies, system resources, etc.
"""

import sys
import platform

def test_environment():
    """Check runtime environment"""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Python version
    python_version = sys.version_info
    print(f"\nv Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print(f"  ! Warning: Python version < 3.10, upgrade recommended")
    else:
        print(f"  v Python version meets requirements (>= 3.10)")
    
    # Operating system
    print(f"\nv Operating system: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    
    # Check dependencies
    dependencies = {
        'torch': 'PyTorch',
        'pyarrow': 'PyArrow',
        'numpy': 'NumPy',
        'transformers': 'Transformers (optional)',
        'sentence_transformers': 'Sentence-Transformers (for comparison)',
    }
    
    print(f"\nDependency check:")
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  v {name}: {version}")
        except ImportError:
            if module in ['transformers']:
                print(f"  ! {name}: Not installed (optional)")
            else:
                print(f"  X {name}: Not installed (required)")
                all_ok = False
    
    # Check llm_compression package
    print(f"\nProject package check:")
    try:
        import llm_compression
        print(f"  v llm_compression: Installed")
        
        # Check key modules
        from llm_compression.inference.arrow_engine import ArrowEngine
        print(f"  v ArrowEngine: Importable")
        
        from llm_compression.embedding_provider import get_default_provider
        print(f"  v EmbeddingProvider: Importable")
        
    except ImportError as e:
        print(f"  X llm_compression: Import failed")
        print(f"     Error: {e}")
        all_ok = False
    
    # System resources
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\nSystem resources:")
        print(f"  Total memory: {mem.total / (1024**3):.1f} GB")
        print(f"  Available memory: {mem.available / (1024**3):.1f} GB")
        print(f"  Memory usage: {mem.percent}%")
        
        if mem.available < 2 * (1024**3):
            print(f"  ! Warning: Available memory < 2GB, may affect performance")
        else:
            print(f"  v Available memory is sufficient")
        
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        print(f"\n  CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
        
    except ImportError:
        print(f"\n! psutil not installed, skipping system resource check")
    
    # Check CUDA
    print(f"\nGPU check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  v CUDA available")
            print(f"     Device count: {torch.cuda.device_count()}")
            print(f"     Current device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  i CUDA not available, will use CPU")
    except:
        print(f"  i Cannot check CUDA status")
    
    print(f"\n" + "=" * 60)
    if all_ok:
        print("v Environment check passed")
        return 0
    else:
        print("X Environment check failed, please install missing dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(test_environment())
