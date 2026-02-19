#!/usr/bin/env python3
"""
Arrow Optimization - Installation Verification Script

This script verifies that all required dependencies for the Arrow-optimized
embedding system are correctly installed.
"""

import sys
import importlib
from typing import List, Tuple


def check_python_version() -> bool:
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.10)")
        return False


def check_package(name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def main():
    """Main verification function"""
    print("=" * 60)
    print("Arrow Optimization - Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Checking Python version...")
    if not check_python_version():
        all_ok = False
    print()
    
    # Core dependencies (from requirements.txt)
    print("Checking core dependencies...")
    core_deps = [
        ("pyarrow", "pyarrow"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
    ]
    
    for name, import_name in core_deps:
        ok, version = check_package(name, import_name)
        if ok:
            print(f"✅ {name} {version}")
        else:
            print(f"❌ {name} (not installed)")
            all_ok = False
    print()
    
    # Arrow optimization dependencies
    print("Checking Arrow optimization dependencies...")
    arrow_deps = [
        ("tokenizers", "tokenizers"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("prometheus-client", "prometheus_client"),
    ]
    
    arrow_ok = True
    for name, import_name in arrow_deps:
        ok, version = check_package(name, import_name)
        if ok:
            print(f"✅ {name} {version}")
        else:
            print(f"⚠️  {name} (not installed - optional)")
            arrow_ok = False
    
    if not arrow_ok:
        print()
        print("💡 To install Arrow optimization dependencies:")
        print("   pip install -r requirements-arrow.txt")
    print()
    
    # Optional dependencies
    print("Checking optional dependencies...")
    optional_deps = [
        ("pytest", "pytest"),
        ("pytest-benchmark", "pytest_benchmark"),
        ("locust", "locust"),
        ("httpx", "httpx"),
    ]
    
    for name, import_name in optional_deps:
        ok, version = check_package(name, import_name)
        if ok:
            print(f"✅ {name} {version}")
        else:
            print(f"⚠️  {name} (optional - for testing)")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok and arrow_ok:
        print("✅ All dependencies installed!")
        print()
        print("Next steps:")
        print("1. Read: docs/arrow-optimization/QUICK_START.md")
        print("2. Run: python examples/arrow-optimization/quick_start.py")
        print("3. Start: See docs/arrow-optimization/TASKS.md")
    elif all_ok:
        print("✅ Core dependencies installed")
        print("⚠️  Arrow optimization dependencies missing")
        print()
        print("Install with: pip install -r requirements-arrow.txt")
    else:
        print("❌ Some core dependencies missing")
        print()
        print("Install with: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if (all_ok and arrow_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
