#!/usr/bin/env python3
"""
Verify pymethods block structure in python.rs
"""

def analyze_pymethods():
    with open('src/python.rs', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pymethods_blocks = []
    current_block = None
    brace_count = 0
    in_pymethods = False
    
    for i, line in enumerate(lines, 1):
        if '#[pymethods]' in line:
            if current_block:
                pymethods_blocks.append(current_block)
            current_block = {'start': i, 'methods': [], 'impl_line': None, 'class': None}
            in_pymethods = True
            continue
        
        if in_pymethods and current_block and current_block['impl_line'] is None:
            if 'impl ' in line and '{' in line:
                # Extract class name
                class_name = line.split('impl ')[1].split('{')[0].strip()
                current_block['class'] = class_name
                current_block['impl_line'] = i
                brace_count = 1  # Start counting from the opening brace
                continue
        
        if in_pymethods and current_block and current_block['impl_line']:
            # Count braces to track block scope
            brace_count += line.count('{') - line.count('}')
            
            # Look for method definitions (must have 'fn ' at start of trimmed line or after attributes)
            stripped = line.strip()
            if (stripped.startswith('fn ') or (stripped.startswith('pub fn ') or stripped.startswith('pub(crate) fn '))) and '(' in line:
                method_name = line.strip().split('fn ')[1].split('(')[0].strip()
                current_block['methods'].append((i, method_name))
            
            # Check if we've closed the impl block
            if brace_count == 0:
                current_block['end'] = i
                pymethods_blocks.append(current_block)
                current_block = None
                in_pymethods = False
    
    # Filter for ArrowQuantV2 only
    arrow_blocks = [b for b in pymethods_blocks if b.get('class') == 'ArrowQuantV2']
    
    print(f"Found {len(arrow_blocks)} pymethods blocks for ArrowQuantV2:\n")
    
    for idx, block in enumerate(arrow_blocks, 1):
        print(f"Block {idx}: Lines {block['start']}-{block.get('end', '?')}")
        print(f"  Class: {block.get('class')}")
        print(f"  Methods ({len(block['methods'])}):")
        for line_num, method in block['methods']:
            print(f"    Line {line_num}: {method}")
        print()
    
    return arrow_blocks

if __name__ == '__main__':
    blocks = analyze_pymethods()
    
    if len(blocks) > 1:
        print("WARNING: Multiple pymethods blocks detected!")
        print("PyO3 only exports methods from the FIRST block.")
    elif len(blocks) == 1:
        print(f"✓ Single pymethods block with {len(blocks[0]['methods'])} methods")
