# -*- coding: utf-8 -*-
"""重新测试剩余未产出的 24 个创新方法"""
import os, sys, json, subprocess, glob, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

extra_paths = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, 'Code', 'Downscaler'),
    os.path.join(PROJECT_ROOT, 'Code'),
]
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join(extra_paths) + os.pathsep + env.get('PYTHONPATH', '')

INNOVATION_DIR = os.path.join(PROJECT_ROOT, 'test_result', '创新方法')
TIMEOUT_PER_METHOD = 7200  # 2 hours max per method

# Get remaining methods (scripts without corresponding pre_exp.json)
# Also skip methods known to hang/crash
SKIP_METHODS = {'AdaptiveOnlineEnsemble', 'AdvancedRK'}

all_scripts = sorted(glob.glob(os.path.join(INNOVATION_DIR, '*_十折标准模式.py')))
existing_results = set(
    os.path.basename(f).replace('_pre_exp.json', '')
    for f in glob.glob(os.path.join(INNOVATION_DIR, '*_pre_exp.json'))
)

remaining = [
    s for s in all_scripts
    if os.path.basename(s).split('_十折')[0] not in existing_results
    and os.path.basename(s).split('_十折')[0] not in SKIP_METHODS
]

print(f"Total scripts: {len(all_scripts)}")
print(f"Existing results: {len(existing_results)}")
print(f"Remaining to test: {len(remaining)}")
print(f"Timeout per method: {TIMEOUT_PER_METHOD}s")
print()

passed = 0
failed = 0
timed_out = 0
errors = []

start_all = time.time()
for i, script in enumerate(remaining):
    basename = os.path.basename(script)
    method_name = basename.split('_十折')[0]
    start_time = time.time()
    print(f"[{i+1}/{len(remaining)}] {method_name} ... ", end='', flush=True)

    try:
        result = subprocess.run(
            [sys.executable, script, '--pre-only'],
            cwd=PROJECT_ROOT, env=env,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=TIMEOUT_PER_METHOD
        )
        elapsed = time.time() - start_time

        pre_json = os.path.join(INNOVATION_DIR, f'{method_name}_pre_exp.json')
        if os.path.exists(pre_json):
            with open(pre_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            r2 = data.get('metrics', {}).get('R2', '?')
            if data.get('passed'):
                passed += 1
                print(f'PASS (R2={r2:.4f}, {elapsed:.0f}s)')
            else:
                failed += 1
                print(f'FAIL (R2={r2:.4f}, {elapsed:.0f}s)')
        else:
            failed += 1
            print(f'NO OUTPUT (rc={result.returncode}, {elapsed:.0f}s)')
            errors.append((method_name, f'rc={result.returncode}, no json'))

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        timed_out += 1
        print(f'TIMEOUT ({elapsed:.0f}s)')
        errors.append((method_name, 'timeout'))

    except Exception as e:
        elapsed = time.time() - start_time
        failed += 1
        print(f'ERROR ({elapsed:.0f}s): {e}')
        errors.append((method_name, str(e)))

total_elapsed = time.time() - start_all
print(f"\n=== Retest Complete ({total_elapsed:.0f}s) ===")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Timeout: {timed_out}")
if errors:
    print(f"Errors ({len(errors)}):")
    for name, err in errors:
        print(f"  - {name}: {err}")

# Final tally
all_results = glob.glob(os.path.join(INNOVATION_DIR, '*_pre_exp.json'))
print(f"\nTotal results now: {len(all_results)}/{len(all_scripts)}")
