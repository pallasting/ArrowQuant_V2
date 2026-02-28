"""
Fix absolute imports to relative imports in ai-os-diffusion package.
This allows the package to work despite having a hyphen in the directory name.
"""

import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: from ai_os_diffusion.module import ...
    # Replace with: from .module import ... (for __init__.py files)
    # Replace with: from ..module import ... (for files in subdirectories)
    
    # Determine the relative import level
    parts = file_path.relative_to(Path("ai-os-diffusion")).parts
    if file_path.name == '__init__.py':
        # In __init__.py, use single dot for same-level imports
        if len(parts) == 1:  # Top-level __init__.py
            # from ai_os_diffusion.xxx import -> from .xxx import
            content = re.sub(
                r'from ai_os_diffusion\.(\w+)',
                r'from .\1',
                content
            )
        else:
            # Subpackage __init__.py
            # from ai_os_diffusion.subpkg.module import -> from .module import
            # from ai_os_diffusion.other import -> from ..other import
            package_name = parts[0]
            content = re.sub(
                rf'from ai_os_diffusion\.{package_name}\.(\w+)',
                r'from .\1',
                content
            )
            content = re.sub(
                r'from ai_os_diffusion\.(\w+)',
                r'from ..\1',
                content
            )
    else:
        # Regular module file
        # from ai_os_diffusion.same_package.module import -> from .module import
        # from ai_os_diffusion.other_package import -> from ..other_package import
        if len(parts) > 1:
            package_name = parts[0]
            content = re.sub(
                rf'from ai_os_diffusion\.{package_name}\.(\w+)',
                r'from .\1',
                content
            )
            content = re.sub(
                r'from ai_os_diffusion\.(\w+)',
                r'from ..\1',
                content
            )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all Python files in ai-os-diffusion."""
    ai_os_dir = Path("ai-os-diffusion")
    python_files = list(ai_os_dir.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files")
    print("=" * 60)
    
    fixed_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            print(f"✅ Fixed: {file_path}")
            fixed_count += 1
    
    print("=" * 60)
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
