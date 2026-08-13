# -*- coding: utf-8 -*-
"""
批量修复脚本：绝对路径 → 相对路径，fold_split_table.csv → fold_split_table_daily.csv
"""
import os
import re
import glob

PROJECT_ROOT = r'E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch'

# 需要处理的目录（跳过 legacy_tests, 年均融合方法历史迭代, fusion_scripts, 文档拆分, paper_output）
TARGET_DIRS = [
    'agents',
    'test_result/基准方法',
    'test_result/创新方法',
    'Innovation/success',
    'Innovation/failed',
    'CodeWorkSpace/复现方法代码',
    'CodeWorkSpace/新融合方法代码',
    'CodeWorkSpace/改造后VNA_eVNA_aVNA',
    'Code/Downscaler',
    'Code/VNAeVNAaVNA',
    'test_result/历史最佳方案',
]

# 不在这些子目录中处理
SKIP_DIRS = [
    'test_result/legacy_tests',
    'CodeWorkSpace/年均融合方法',
    'fusion_scripts',
    '文档拆分',
    'paper_output',
    'PaperDownload',
    'PaperDownloadMd',
    'MethodToSmart',
    'SmartToCode',
    'LocalPaperLibrary',
    'skills',
    '.git',
    '.claude',
    '__pycache__',
    'shared',  # skip our own shared module
]

# 旧路径模式 -> 替换内容
HARDCODED_PATH = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'


def find_py_files():
    """查找所有需要处理的 Python 文件"""
    files = []
    for dir_name in TARGET_DIRS:
        full_dir = os.path.join(PROJECT_ROOT, dir_name)
        if not os.path.exists(full_dir):
            continue
        for root, dirs, filenames in os.walk(full_dir):
            # 过滤不需要的目录
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
            for fname in filenames:
                if fname.endswith('.py') and fname != '__pycache__':
                    files.append(os.path.join(root, fname))
    return files


def fix_file(filepath):
    """修复单个文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # === 修复 1: 替换硬编码路径 (sys.path.insert) ===
    old_insert = f"sys.path.insert(0, '{HARDCODED_PATH}')"
    if old_insert in content:
        content = content.replace(
            old_insert,
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
        )
        # 确保有 import os
        if 'import os\n' not in content and 'import os\r\n' not in content:
            # Check if os is already imported
            if 'import os' not in content.split('\n')[0:30]:
                content = 'import os\n' + content

    old_insert_code = f"sys.path.insert(0, '{HARDCODED_PATH}/Code')"
    if old_insert_code in content:
        content = content.replace(old_insert_code, '')

    old_insert_ds = f"sys.path.insert(0, '{HARDCODED_PATH}/Code/Downscaler')"
    if old_insert_ds in content:
        content = content.replace(old_insert_ds, '')

    # === 修复 2: 替换 ROOT_DIR / root_dir 赋值 ===
    patterns = [
        (f"ROOT_DIR = '{HARDCODED_PATH}'", "ROOT_DIR = str(get_project_root())"),
        (f"root_dir = '{HARDCODED_PATH}'", "root_dir = str(get_project_root())"),
        (f"ROOT_DIR = r'{HARDCODED_PATH}'", "ROOT_DIR = str(get_project_root())"),
        (f"root_dir = r'{HARDCODED_PATH}'", "root_dir = str(get_project_root())"),
    ]
    for old, new in patterns:
        if old in content:
            content = content.replace(old, new)
            # Add import
            if 'from shared.paths import' not in content:
                content = content.replace(
                    'import os\n',
                    'import os\nfrom shared.paths import get_project_root, data_path\n'
                )
                if 'from shared.paths import get_project_root, data_path\n' not in content:
                    # Try other import pattern
                    content = content.replace(
                        'import os\r\n',
                        'import os\r\nfrom shared.paths import get_project_root, data_path\r\n'
                    )

    # === 修复 3: 替换 fold_split_table.csv -> fold_split_table_daily.csv ===
    if 'fold_split_table.csv' in content and 'fold_split_table_daily.csv' not in content:
        content = content.replace('fold_split_table.csv', 'fold_split_table_daily.csv')

    # === 修复 4: 替换硬编码路径的数据文件引用 ===
    replacements = [
        (f"'{HARDCODED_PATH}/test_data/", "data_path('test_data/"),
        (f"f'{{ROOT_DIR}}/test_data/", "data_path('test_data/"),
        (f"f'{{root_dir}}/test_data/", "data_path('test_data/"),
        (f"ROOT_DIR + '/test_data/", "data_path('test_data/"),
        (f"root_dir + '/test_data/", "data_path('test_data/"),
        (f"os.path.join(ROOT_DIR, 'test_data/", "data_path('test_data/"),
        (f"os.path.join(root_dir, 'test_data/", "data_path('test_data/"),
    ]
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            # Fix the closing: data_path('test_data/xxx.csv')  needs the closing )
            # The old pattern: f'{ROOT_DIR}/test_data/xxx'  becomes: data_path('test_data/xxx')
            # We need to fix the trailing quote
            if new == "data_path('test_data/":
                # Find patterns like: data_path('test_data/xxx.csv')
                # The old had: f'{ROOT_DIR}/test_data/xxx.csv'
                # After replace: data_path('test_data/xxx.csv'
                # Need to close: data_path('test_data/xxx.csv')
                pass  # handled by next fix

    # Fix incomplete data_path calls (missing closing quote and paren)
    # Pattern: data_path('test_data/something.csv') should be complete
    # The old: f'{ROOT_DIR}/test_data/xxx.csv' became data_path('test_data/xxx.csv' but missing ')
    # Let's fix these
    content = re.sub(
        r"data_path\('(test_data/[^']+)'(\s*)\)",  # already complete
        r"data_path('\1')",
        content
    )
    # Fix: data_path('test_data/xxx.csv'  ->  data_path('test_data/xxx.csv')
    content = re.sub(
        r"data_path\('(test_data/[^']+)'\)",  # already correct
        r"data_path('\1')",
        content
    )

    # === 修复 5: 替换 agents/ 中的硬编码 root_dir ===
    for old_root in [
        f"root_dir = '{HARDCODED_PATH}'",
        f"self.root_dir = '{HARDCODED_PATH}'",
        f"project_root = '{HARDCODED_PATH}'",
    ]:
        if old_root in content:
            content = content.replace(old_root, old_root.replace(HARDCODED_PATH, "str(get_project_root())"))
            if 'from shared.paths import' not in content:
                content = 'from shared.paths import get_project_root, data_path\n' + content

    # Only write if changed
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    files = find_py_files()
    print(f"Found {len(files)} Python files to process")
    print()

    changed = []
    unchanged = []
    errors = []

    for filepath in sorted(files):
        try:
            if fix_file(filepath):
                changed.append(filepath)
            else:
                unchanged.append(filepath)
        except Exception as e:
            errors.append((filepath, str(e)))

    print(f"Changed:  {len(changed)}")
    for f in changed:
        rel = os.path.relpath(f, PROJECT_ROOT)
        print(f"  [FIXED] {rel}")

    print(f"\nUnchanged: {len(unchanged)}")
    print(f"Errors:    {len(errors)}")
    for f, e in errors:
        print(f"  [ERROR] {os.path.relpath(f, PROJECT_ROOT)}: {e}")


if __name__ == '__main__':
    main()
