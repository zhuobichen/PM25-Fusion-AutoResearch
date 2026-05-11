# -*- coding: utf-8 -*-
import re, os
method_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'CodeWorkSpace', '新融合方法代码')
method_dir = os.path.abspath(method_dir)
for f in os.listdir(method_dir):
    if f.endswith('.py') and not f.startswith('_'):
        name = f[:-3]
        with open(os.path.join(method_dir, f), 'r', encoding='utf-8') as fh:
            code = fh.read()
        classes = re.findall(r'^class\s+(\w+)\s*[:\(]', code, re.MULTILINE)
        if classes:
            filtered = [c for c in classes if 'Model' not in c and 'Wrapper' not in c and 'Setting' not in c]
            if not filtered and classes:
                print(f'{name}: all classes filtered: {classes}')
