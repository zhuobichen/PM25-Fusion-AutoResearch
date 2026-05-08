"""Fix: sys.path.insert uses os before os is imported"""
import os
import re

PROJECT_ROOT = r'E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch'
SKIP = {'.git', '.claude', '__pycache__', 'shared', 'node_modules', 'skills'}


def fix_import_order(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Pattern: sys.path.insert uses os.path... but import os comes later
    # Fix: ensure 'import os\n' or 'import os\r\n' is present before any sys.path.insert

    # Check if we have this pattern
    if "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" not in content:
        return False

    # Check if os is already imported properly
    lines = content.split('\n')

    # Find the first sys.path.insert line that uses os
    sys_line_idx = None
    os_import_idx = None

    for i, line in enumerate(lines):
        if 'os.path.dirname' in line and 'sys.path.insert' in line:
            sys_line_idx = i
        if line.strip() == 'import os' and os_import_idx is None:
            os_import_idx = i

    if sys_line_idx is not None and (os_import_idx is None or os_import_idx > sys_line_idx):
        # os is imported after sys.path.insert, or not at all
        # Insert 'import os' before sys.path.insert
        if os_import_idx is None:
            # No import os at all - add it
            lines.insert(sys_line_idx, 'import os')
        elif os_import_idx > sys_line_idx:
            # import os comes after - need to reorder
            # Move import os before sys.path.insert
            os_line = lines.pop(os_import_idx)
            lines.insert(sys_line_idx, os_line)

        content = '\n'.join(lines)

    # Clean up: remove duplicate blank lines (more than 2 consecutive)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Remove duplicate import os (keep only one before sys.path.insert)
    # Count import os occurrences
    import_os_lines = [i for i, l in enumerate(content.split('\n')) if l.strip() == 'import os']
    if len(import_os_lines) > 1:
        lines = content.split('\n')
        # Keep the first one, remove others
        for idx in reversed(import_os_lines[1:]):
            lines.pop(idx)
        content = '\n'.join(lines)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    count = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for fname in files:
            if fname.endswith('.py'):
                filepath = os.path.join(root, fname)
                if fix_import_order(filepath):
                    count += 1
                    print(f"  [FIXED] {os.path.relpath(filepath, PROJECT_ROOT)}")
    print(f"\nFixed {count} files")


if __name__ == '__main__':
    main()
