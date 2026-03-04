#!/usr/bin/env python3
"""
Diagnose PyO3 pymethods export issue by analyzing the Rust source code.
"""

import re

def analyze_pymethods_block():
    with open('src/python.rs', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find ArrowQuantV2 pymethods block
    start_marker = 'impl ArrowQuantV2 {'
    start_idx = content.find(start_marker, content.find('#[pymethods]'))
    
    if start_idx == -1:
        print("ERROR: Could not find ArrowQuantV2 impl block")
        return
    
    # Find the end of the pymethods block
    brace_count = 0
    in_block = False
    end_idx = start_idx
    
    for i, char in enumerate(content[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
            in_block = True
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and in_block:
                end_idx = i
                break
    
    pymethods_block = content[start_idx:end_idx+1]
    
    # Find all method definitions
    method_pattern = r'fn\s+(\w+)\s*\([^)]*\)'
    methods = []
    
    for match in re.finditer(method_pattern, pymethods_block):
        method_name = match.group(1)
        if not method_name.startswith('_'):
            # Find if it has #[pyo3(...)] attribute
            method_start = match.start()
            # Look backwards for attributes
            prev_text = pymethods_block[max(0, method_start-500):method_start]
            has_pyo3_attr = '#[pyo3(' in prev_text or '#[new]' in prev_text
            
            methods.append({
                'name': method_name,
                'has_pyo3_attr': has_pyo3_attr,
                'position': match.start()
            })
    
    print(f"Found {len(methods)} methods in ArrowQuantV2 pymethods block:\n")
    
    for i, method in enumerate(methods, 1):
        attr_status = "✓" if method['has_pyo3_attr'] else "✗"
        print(f"{i:2}. {method['name']:30} [pyo3 attr: {attr_status}]")
    
    # Check for potential issues
    print("\n" + "="*60)
    print("POTENTIAL ISSUES:")
    print("="*60)
    
    # Check if there are methods without #[pyo3] attributes
    methods_without_attr = [m for m in methods if not m['has_pyo3_attr']]
    if methods_without_attr:
        print(f"\n⚠️  {len(methods_without_attr)} methods without #[pyo3] attributes:")
        for m in methods_without_attr:
            print(f"   - {m['name']}")
    
    # Check for very long pymethods block
    block_size = len(pymethods_block)
    print(f"\n📏 Pymethods block size: {block_size:,} characters ({block_size/1024:.1f} KB)")
    if block_size > 100000:
        print("   ⚠️  Block is very large (>100KB) - this might cause PyO3 issues")
    
    # Check for syntax issues around method boundaries
    print("\n🔍 Checking method boundaries...")
    for i in range(len(methods) - 1):
        current = methods[i]
        next_method = methods[i+1]
        gap = next_method['position'] - current['position']
        if gap < 50:
            print(f"   ⚠️  Very small gap between {current['name']} and {next_method['name']} ({gap} chars)")

if __name__ == "__main__":
    analyze_pymethods_block()
