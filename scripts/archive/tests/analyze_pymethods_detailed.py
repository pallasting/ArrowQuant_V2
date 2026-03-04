#!/usr/bin/env python3
"""Detailed analysis of pymethods block structure"""

def analyze_detailed():
    with open('src/python.rs', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the ArrowQuantV2 pymethods block
    in_block = False
    block_start = None
    brace_depth = 0
    methods = []
    
    for i, line in enumerate(lines, 1):
        if '#[pymethods]' in line and i > 500:  # Skip PyDiffusionQuantConfig
            # Check next few lines for ArrowQuantV2
            for j in range(i, min(i+5, len(lines))):
                if 'impl ArrowQuantV2' in lines[j]:
                    in_block = True
                    block_start = i
                    print(f"Found ArrowQuantV2 pymethods block at line {i}")
                    break
        
        if in_block:
            # Track brace depth
            if 'impl ArrowQuantV2' in line and '{' in line:
                brace_depth = 1
            else:
                brace_depth += line.count('{') - line.count('}')
            
            # Look for method definitions
            stripped = line.strip()
            if stripped.startswith('fn ') and '(' in line:
                method_name = stripped.split('fn ')[1].split('(')[0].strip()
                
                # Check for pyo3 signature attribute in previous lines
                has_pyo3_attr = False
                for j in range(max(0, i-5), i):
                    if '#[pyo3' in lines[j]:
                        has_pyo3_attr = True
                        break
                
                methods.append({
                    'line': i,
                    'name': method_name,
                    'has_pyo3_attr': has_pyo3_attr,
                    'brace_depth': brace_depth
                })
            
            # Check if block ended
            if brace_depth == 0 and i > block_start + 10:
                print(f"Block ended at line {i}\n")
                break
    
    print(f"Found {len(methods)} methods:\n")
    for m in methods:
        pyo3_marker = "✓" if m['has_pyo3_attr'] else "✗"
        print(f"  Line {m['line']:4d}: {m['name']:35s} PyO3: {pyo3_marker}  Depth: {m['brace_depth']}")
    
    # Check which methods are exported
    print("\n" + "="*70)
    print("Checking Python exports...")
    print("="*70 + "\n")
    
    try:
        import arrow_quant_v2
        q = arrow_quant_v2.ArrowQuantV2()
        exported = [m for m in dir(q) if not m.startswith('_')]
        
        print(f"Exported methods ({len(exported)}):")
        for m in sorted(exported):
            print(f"  ✓ {m}")
        
        print(f"\nMissing methods:")
        for m in methods:
            if m['name'] not in exported and m['name'] != 'new':
                print(f"  ✗ {m['name']} (line {m['line']})")
    except Exception as e:
        print(f"Error importing module: {e}")

if __name__ == '__main__':
    analyze_detailed()
