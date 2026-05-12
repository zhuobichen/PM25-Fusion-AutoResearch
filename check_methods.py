import json, os

with open('method_registry.json', 'r', encoding='utf-8') as f:
    reg = json.load(f)
reg_methods = set(reg.get('methods', {}).keys())

smart_dir = 'SmartToCode'
results = []
for root, dirs, files in os.walk(smart_dir):
    for f in sorted(files):
        if f.endswith('.md') and f not in ('INVENTORY.md','method_fingerprint.md5','innovation_note.md'):
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8') as fh:
                content = fh.read()
            found_in_reg = False
            matched = []
            for m in reg_methods:
                if len(m) > 2 and m in content:
                    found_in_reg = True
                    matched.append(m)
            status = 'REGISTERED' if found_in_reg else 'NEW'
            results.append((status, f, matched))

print(f'Total instruction files: {len(results)}')
print()
for status, f, matched in results:
    if status == 'NEW':
        print(f'[NEW] {f}')
print()
print('--- Summary ---')
new_count = sum(1 for s,_,_ in results if s == 'NEW')
reg_count = sum(1 for s,_,_ in results if s == 'REGISTERED')
print(f'REGISTERED: {reg_count}')
print(f'NEW (not in registry): {new_count}')
