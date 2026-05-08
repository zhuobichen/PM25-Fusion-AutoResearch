"""Fix incomplete data_path() calls - missing closing paren"""
import os
import re

PROJECT_ROOT = r'E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch'

# Skip dirs
SKIP = {'.git', '.claude', '__pycache__', 'shared', 'node_modules', 'skills'}


def fix_data_path_calls(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix: data_path('test_data/xxx.csv'  ->  data_path('test_data/xxx.csv')
    # The closing ) is missing after the closing quote
    content = re.sub(
        r"data_path\('(test_data/[^']+)'(\s*)$",
        r"data_path('\1')\2",
        content,
        flags=re.MULTILINE
    )

    # Fix duplicate import os
    while 'import os\nimport os\n' in content:
        content = content.replace('import os\nimport os\n', 'import os\n')
    while 'import os\r\nimport os\r\n' in content:
        content = content.replace('import os\r\nimport os\r\n', 'import os\r\n')

    # Remove empty lines between sys.path.insert and import os
    content = re.sub(
        r"sys\.path\.insert\(0, os\.path\.dirname[^)]+\)\s*\n\s*\n\s*\n",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n",
        content
    )

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
                if fix_data_path_calls(filepath):
                    count += 1
                    print(f"  [FIXED] {os.path.relpath(filepath, PROJECT_ROOT)}")
    print(f"\nFixed {count} files")


if __name__ == '__main__':
    main()
