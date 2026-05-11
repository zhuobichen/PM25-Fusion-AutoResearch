# -*- coding: utf-8 -*-
"""Debug fold extraction"""
import re
import os

method_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'CodeWorkSpace', '新融合方法代码')
method_dir = os.path.abspath(method_dir)

# Test with ARK_OLS
fpath = os.path.join(method_dir, 'ARK_OLS.py')
with open(fpath, 'r', encoding='utf-8') as f:
    code = f.read()

# Find fold loop
m = re.search(r'for\s+fold_id\s+in\s+range\s*\(\s*1\s*,\s*11\s*\)', code)
if m:
    print('=== FOLD LOOP START ===')
    # Find the end of the fold loop body
    lines = code[m.start():].split('\n')
    fold_lines = []
    base_indent = None
    for i, line in enumerate(lines):
        if i == 0:
            fold_lines.append(line)
            continue
        if line.strip() == '':
            fold_lines.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent and line.strip():
            break
        fold_lines.append(line)

    print(f'Total fold lines: {len(fold_lines)}')
    print('--- First 20 lines ---')
    for fl in fold_lines[:20]:
        print(repr(fl))
    print('--- Last 10 lines ---')
    for fl in fold_lines[-10:]:
        print(repr(fl))
