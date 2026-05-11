# -*- coding: utf-8 -*-
"""
为所有新方法生成十折交叉验证脚本（标准模式）
============================================
策略：读取每个方法源码中的 run_xxx_ten_fold 函数体，
提取折叠循环内的核心融合逻辑，嵌入到标准模板中。
"""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
METHOD_DIR = os.path.join(PROJECT_ROOT, 'CodeWorkSpace', '新融合方法代码')
OUTPUT_DIR = SCRIPT_DIR

METHODS = [
    'AdaptiveOnlineEnsemble', 'ARK_OLS', 'BayesianMultisourceFusion',
    'BayesianVariationalFusion', 'BMA_Fusion', 'BMSF_Geostat',
    'CMAQ梯度各向异性克里金', 'ConcentrationStratifiedPolyRK',
    'ConservativeTransport', 'CopulaSpatialFusion', 'CorrDiff_Downscaling',
    'CrossDayValidation', 'CR_ABC', 'CSPRKATO', 'CSPRKHLG', 'CSPRKINT',
    'CSP_RK_AdaptiveThreshold', 'CSP_RK_HybridLayerGPR', 'CSP_RK_Interaction',
    'EnhancedStackingEnsemble', 'EnsembleRK', 'ExtremeStackingEnsemble',
    'FeatureStackingEnsemble', 'GDIDW', 'GradientAnisotropicKriging',
    'GradientBoostingEnsemble', 'gVNA', 'gVNA_full_domain',
    'HeteroGPR_PolyRK', 'HeteroscedasticGPRPolyRK', 'HGPRK', 'HybridEAVNA',
    'LBGPR', 'LocalKernelGPR', 'LogRatioEnsemble', 'MaternGPEnsemble',
    'MKGPRK', 'MSAK', 'MSEF', 'MultiKEnsemble', 'MultiKernelGPREnsemble',
    'MultiKernelGPRPolyRK', 'MultiLevelStackingEnsemble', 'NNResidualEnsemble',
    'PDEICNN', 'PolyEnsemble', 'PolyGPRAdapt', 'QuantileHuberEnsemble',
    'ResidualDistMatchKriging', 'ResidualKriging', 'RRK', 'SLOOCV_AK',
    'SpatialQuantileMapping', 'SpatialZoneEnsemble', 'SPIN_GraphKernel_Kriging',
    'SQDM', 'StackingEnsemble', 'STRK', 'ST_CRK', 'SuperEnsemble',
    'SuperStackingEnsemble', 'TransportGuidedKernel', 'TripleEnsemble',
    'UltimateStackingEnsemble', 'VarioGPR_RK', 'VCFFM', 'VG_VNA', 'WaveletGPR',
    '多尺度残差克里金', '多尺度稳定度自适应克里金', '多项式样条克里金',
    '时空残差共克里金', '鲁棒残差克里金',
]


def read_method_source(method_name):
    """读取方法源码"""
    fpath = os.path.join(METHOD_DIR, method_name + '.py')
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        return f.read()


def find_run_function_name(source_code):
    """找到主运行函数名"""
    patterns = [
        r'def\s+(run_\w+_ten_fold)\s*\(',
        r'def\s+(run_\w+_10_fold)\s*\(',
        r'def\s+(ten_fold_\w+)\s*\(',
        r'def\s+(run_\w+_validation)\s*\(',
        r'def\s+(run_single_day)\s*\(',
    ]
    for pat in patterns:
        m = re.search(pat, source_code)
        if m:
            return m.group(1)
    return None


def find_class_name(source_code):
    """找到主类名"""
    # 排除内部辅助类，找主要的融合类
    classes = re.findall(r'^class\s+(\w+)\s*[:\(]', source_code, re.MULTILINE)
    if not classes:
        return None
    # 优先选择名称匹配方法名的类
    for c in classes:
        if 'Model' not in c and 'Wrapper' not in c and 'Setting' not in c:
            return c
    return classes[0]


def extract_special_imports(source_code):
    """提取方法特定的import语句"""
    lines = source_code.split('\n')
    imports = []
    skip_libs = {'shared.paths', 'numpy', 'pandas', 'sklearn', 'netCDF4',
                 'datetime', 'joblib', 'json', 'os', 'sys', 'warnings',
                 'pathlib', 'collections', 'copy', 'time', 'logging'}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('from ') and 'import' in stripped:
            module = stripped.split('from ')[1].split(' import')[0].strip()
            if module and not any(lib in module for lib in skip_libs):
                imports.append(stripped)
        elif stripped.startswith('import '):
            module = stripped.replace('import ', '').strip()
            if module and not any(lib in module for lib in skip_libs):
                imports.append(stripped)
    # 去重
    seen = set()
    unique = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)
    return unique


def extract_fold_body(source_code, run_func_name):
    """从run函数中提取折叠循环体内的核心逻辑"""
    # 找到函数定义
    func_pattern = rf'def\s+{re.escape(run_func_name)}\s*\([^)]*\):'
    func_match = re.search(func_pattern, source_code)
    if not func_match:
        return None

    # 找到函数体
    lines = source_code[func_match.end():].split('\n')
    func_body_lines = []
    base_indent = None
    for line in lines:
        if line.strip() == '':
            func_body_lines.append(line)
            continue
        # 检测缩进
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent and line.strip():
            break
        func_body_lines.append(line)

    func_body = '\n'.join(func_body_lines)

    # 找到 for fold_id 循环
    fold_pattern = r'for\s+fold_id\s+in\s+range\s*\(\s*1\s*,\s*11\s*\)'
    fold_match = re.search(fold_pattern, func_body)
    if not fold_match:
        # 尝试其他模式
        fold_pattern2 = r'for\s+fold\s+in\s+range\s*\(\s*1\s*,\s*11\s*\)'
        fold_match = re.search(fold_pattern2, func_body)

    if not fold_match:
        return None

    # 提取循环体
    fold_start = fold_match.end()
    fold_lines = func_body[fold_start:].split('\n')
    fold_body_lines = []
    fold_base_indent = None
    for line in fold_lines:
        if line.strip() == '':
            fold_body_lines.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if fold_base_indent is None and line.strip():
            fold_base_indent = indent
        if fold_base_indent is not None and indent < fold_base_indent and line.strip():
            break
        fold_body_lines.append(line)

    return '\n'.join(fold_body_lines)


def extract_data_loading(source_code, run_func_name):
    """提取数据加载部分（CMAQ提取等）"""
    func_pattern = rf'def\s+{re.escape(run_func_name)}\s*\([^)]*\):'
    func_match = re.search(func_pattern, source_code)
    if not func_match:
        return ''

    lines = source_code[func_match.end():].split('\n')
    func_body_lines = []
    base_indent = None
    for line in lines:
        if line.strip() == '':
            func_body_lines.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent and line.strip():
            break
        func_body_lines.append(line)

    func_body = '\n'.join(func_body_lines)

    # 找到CMAQ提取部分
    cmaq_pattern = r'(cmaq_values\s*=.*?day_df\[.CMAQ.\]\s*=\s*cmaq_values)'
    cmaq_match = re.search(cmaq_pattern, func_body, re.DOTALL)
    if cmaq_match:
        return cmaq_match.group(0)
    return ''


def generate_script_for_method(method_name):
    """为单个方法生成验证脚本"""
    source = read_method_source(method_name)
    if source is None:
        return None

    run_func = find_run_function_name(source)
    class_name = find_class_name(source)
    special_imports = extract_special_imports(source)

    # 确定方法类型
    has_class = class_name is not None
    has_run_func = run_func is not None

    # 生成脚本
    lines = []
    lines.append('# -*- coding: utf-8 -*-')
    lines.append(f'"""')
    lines.append(f'{method_name} 十折交叉验证 - 标准模式')
    lines.append(f'{"=" * 50}')
    lines.append(f'自动从 {method_name}.py 生成')
    lines.append(f'"""')
    lines.append('')
    lines.append('import sys')
    lines.append('import os')
    lines.append("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
    lines.append('')
    lines.append("from shared.paths import get_project_root, data_path")
    lines.append('import json')
    lines.append('import numpy as np')
    lines.append('import pandas as pd')
    lines.append('import netCDF4 as nc')
    lines.append('from datetime import datetime, timedelta')
    lines.append('from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error')
    lines.append('from sklearn.linear_model import LinearRegression')
    lines.append('from joblib import Parallel, delayed')

    # 添加方法特定import
    for imp in special_imports:
        lines.append(imp)

    lines.append('')
    lines.append("ROOT_DIR = str(get_project_root())")
    lines.append("CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')")
    lines.append("MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')")
    lines.append("FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')")
    lines.append("OUTPUT_DIR = f'{ROOT_DIR}/test_result/创新方法'")
    lines.append("os.makedirs(OUTPUT_DIR, exist_ok=True)")
    lines.append('')
    lines.append("BASELINE = {")
    lines.append("    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},")
    lines.append("    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},")
    lines.append("    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},")
    lines.append("    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},")
    lines.append("}")
    lines.append('')

    # compute_metrics
    lines.append('''
def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }


def get_cmaq_at_site(lon, lat, lon_grid, lat_grid, pm25_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return pm25_grid[row, col]


def get_cmaq_grid_coord(lon, lat, lon_grid, lat_grid):
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.argmin(dist)
    ny, nx = lon_grid.shape
    row, col = idx // nx, idx % nx
    return lon_grid[row, col], lat_grid[row, col]
''')

    # 核心折叠函数
    safe_name = method_name.replace('-', '_').replace('.', '_')
    lines.append(f'')
    lines.append(f'def ten_fold_{safe_name}(selected_day):')
    lines.append(f'    """{method_name} 标准模式十折验证"""')

    # 根据方法类型生成核心逻辑
    if has_class and has_run_func:
        # 有类也有run函数：使用run函数的逻辑
        lines.append(generate_from_run_func(method_name, run_func, source, class_name))
    elif has_class and not has_run_func:
        # 纯类方法
        lines.append(generate_from_class(method_name, class_name, source))
    elif has_run_func:
        # 纯函数方法
        lines.append(generate_from_run_func(method_name, run_func, source, None))
    else:
        # 特殊情况
        lines.append(generate_special_case(method_name, source))

    # stage验证和main
    lines.append(generate_stage_main(method_name, safe_name))

    return '\n'.join(lines)


def generate_from_run_func(method_name, run_func, source, class_name):
    """从run函数生成核心逻辑"""
    lines = []
    lines.append(f'    # 从 {run_func} 提取的核心逻辑')
    lines.append(f'    monitor_df = pd.read_csv(MONITOR_FILE)')
    lines.append(f'    fold_df = pd.read_csv(FOLD_FILE)')
    lines.append(f'')
    lines.append(f"    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()")
    lines.append(f"    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')")
    lines.append(f"    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])")
    lines.append(f'')
    lines.append(f'    if len(day_df) < 100:')
    lines.append(f'        return np.array([]), np.array([])')
    lines.append(f'')
    lines.append(f"    ds = nc.Dataset(CMAQ_FILE, 'r')")
    lines.append(f"    lon_cmaq = ds.variables['lon'][:]")
    lines.append(f"    lat_cmaq = ds.variables['lat'][:]")
    lines.append(f"    pred_pm25 = ds.variables['pred_PM25'][:]")
    lines.append(f"    ds.close()")
    lines.append(f'')
    lines.append(f"    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')")
    lines.append(f"    day_idx = (date_obj - datetime(2020, 1, 1)).days")
    lines.append(f'    if day_idx >= pred_pm25.shape[0]:')
    lines.append(f'        return np.array([]), np.array([])')
    lines.append(f'    cmaq_day = pred_pm25[day_idx]')
    lines.append(f'')
    lines.append(f'    cmaq_values = []')
    lines.append(f"    for _, row in day_df.iterrows():")
    lines.append(f"        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)")
    lines.append(f'        cmaq_values.append(val)')
    lines.append(f"    day_df['CMAQ'] = cmaq_values")
    lines.append(f'')

    # 检查是否需要网格数据
    if 'X_grid_full' in source or 'y_grid_model_full' in source or 'NNA' in source:
        lines.append(f'    ny, nx = lon_cmaq.shape')
        lines.append(f'    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])')
        lines.append(f'    y_grid_model_full = cmaq_day.ravel()')
        lines.append(f'')

    lines.append(f'    all_y_true = []')
    lines.append(f'    all_y_pred = []')
    lines.append(f'')
    lines.append(f'    for fold_id in range(1, 11):')
    lines.append(f"        train_df = day_df[day_df['fold'] != fold_id].copy()")
    lines.append(f"        test_df = day_df[day_df['fold'] == fold_id].copy()")
    lines.append(f"        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])")
    lines.append(f"        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])")
    lines.append(f'        if len(test_df) == 0 or len(train_df) == 0:')
    lines.append(f'            continue')
    lines.append(f'')

    # 提取折叠体
    fold_body = extract_fold_body(source, run_func)
    if fold_body:
        # 清理折叠体：去掉数据加载部分（train_df/test_df定义），去掉print语句
        fold_lines = fold_body.split('\n')
        cleaned = []
        skip_patterns = [
            "train_df = day_df[day_df",
            "test_df = day_df[day_df",
            "train_df = train_df.dropna",
            "test_df = test_df.dropna",
            "if len(test_df) == 0",
            "print(",
            "results[fold_id]",
            "all_fold_preds[fold_id]",
            "all_y_true.extend",
            "all_y_pred.extend",
        ]
        for fl in fold_lines:
            stripped = fl.strip()
            if not stripped:
                cleaned.append(fl)
                continue
            # 跳过数据加载和print
            skip = False
            for pat in skip_patterns:
                if pat in stripped:
                    skip = True
                    break
            if skip:
                continue
            # 调整缩进：原函数中折叠体缩进2级（8空格），我们需要2级（8空格）
            # 但如果原函数缩进不同，需要调整
            cleaned.append(fl)

        # 添加提取的逻辑
        for cl in cleaned:
            if cl.strip():
                lines.append(cl)
            else:
                lines.append('')

    # 添加结果收集
    lines.append(f'')
    lines.append(f'        all_y_true.extend(y_test)')
    # 确定预测变量名
    pred_var = find_prediction_variable(source, run_func)
    lines.append(f'        all_y_pred.extend({pred_var})')
    lines.append(f'')
    lines.append(f'    return np.array(all_y_true), np.array(all_y_pred)')

    return '\n'.join(lines)


def find_prediction_variable(source, run_func):
    """找到方法的预测变量名"""
    # 常见的预测变量名
    patterns = [
        r'(\w+_pred)\s*=',
        r'(\w+_fusion)\s*=',
        r'(\w+_ensemble)\s*=',
        r'(y_pred)\s*=',
        r'(fusion_pred)\s*=',
        r'(final_pred)\s*=',
        r'(rk_\w+_pred)\s*=',
    ]
    # 在run函数体内找
    func_pattern = rf'def\s+{re.escape(run_func)}\s*\([^)]*\):'
    func_match = re.search(func_pattern, source)
    if not func_match:
        return 'y_pred'

    body = source[func_match.end():func_match.end() + 5000]

    # 找最后赋值的预测变量
    found = []
    for pat in patterns:
        for m in re.finditer(pat, body):
            var = m.group(1)
            if var not in ['y_train', 'y_test', 'm_train', 'm_test', 'residual']:
                found.append(var)

    if found:
        return found[-1]
    return 'y_pred'


def generate_from_class(method_name, class_name, source):
    """为类方法生成核心逻辑"""
    lines = []
    lines.append(f'    # 基于 {class_name} 类的 fit/predict 接口')
    lines.append(f'    monitor_df = pd.read_csv(MONITOR_FILE)')
    lines.append(f'    fold_df = pd.read_csv(FOLD_FILE)')
    lines.append(f'')
    lines.append(f"    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()")
    lines.append(f"    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')")
    lines.append(f"    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])")
    lines.append(f'')
    lines.append(f'    if len(day_df) < 100:')
    lines.append(f'        return np.array([]), np.array([])')
    lines.append(f'')
    lines.append(f"    ds = nc.Dataset(CMAQ_FILE, 'r')")
    lines.append(f"    lon_cmaq = ds.variables['lon'][:]")
    lines.append(f"    lat_cmaq = ds.variables['lat'][:]")
    lines.append(f"    pred_pm25 = ds.variables['pred_PM25'][:]")
    lines.append(f"    ds.close()")
    lines.append(f'')
    lines.append(f"    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')")
    lines.append(f"    day_idx = (date_obj - datetime(2020, 1, 1)).days")
    lines.append(f'    if day_idx >= pred_pm25.shape[0]:')
    lines.append(f'        return np.array([]), np.array([])')
    lines.append(f'    cmaq_day = pred_pm25[day_idx]')
    lines.append(f'')
    lines.append(f'    cmaq_values = []')
    lines.append(f"    for _, row in day_df.iterrows():")
    lines.append(f"        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)")
    lines.append(f'        cmaq_values.append(val)')
    lines.append(f"    day_df['CMAQ'] = cmaq_values")
    lines.append(f'')
    lines.append(f'    all_y_true = []')
    lines.append(f'    all_y_pred = []')
    lines.append(f'')
    lines.append(f'    for fold_id in range(1, 11):')
    lines.append(f"        train_df = day_df[day_df['fold'] != fold_id].copy()")
    lines.append(f"        test_df = day_df[day_df['fold'] == fold_id].copy()")
    lines.append(f"        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])")
    lines.append(f"        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])")
    lines.append(f'        if len(test_df) == 0 or len(train_df) == 0:')
    lines.append(f'            continue')
    lines.append(f'')
    lines.append(f'        train_lon = train_df["Lon"].values')
    lines.append(f'        train_lat = train_df["Lat"].values')
    lines.append(f'        train_Conc = train_df["Conc"].values')
    lines.append(f'        train_mod = train_df["CMAQ"].values')
    lines.append(f'        y_test = test_df["Conc"].values')
    lines.append(f'        m_test = test_df["CMAQ"].values')
    lines.append(f'')

    # 检查构造函数参数
    init_match = re.search(rf'def\s+__init__\s*\(self[^)]*\)', source)
    if init_match:
        init_body = source[init_match.start():init_match.end() + 500]
        # 提取默认参数
        params = re.findall(r'(\w+)\s*=\s*([^,\)]+)', init_body)
        param_str = ', '.join([f'{p[0]}={p[1].strip()}' for p in params if p[0] != 'self'])
        lines.append(f'        model = {class_name}({param_str})')
    else:
        lines.append(f'        model = {class_name}()')

    # fit调用
    if 'cmaq_r2' in source:
        lines.append(f'        cmaq_r2 = r2_score(train_Conc, train_mod) if len(train_Conc) > 10 else np.nan')
        lines.append(f'        model.fit(train_lon, train_lat, train_Conc, train_mod, cmaq_r2=cmaq_r2)')
    else:
        lines.append(f'        model.fit(train_lon, train_lat, train_Conc, train_mod)')

    lines.append(f'')
    lines.append(f'        X_test = np.column_stack([test_df["Lon"].values, test_df["Lat"].values])')
    lines.append(f'        y_pred = model.predict(X_test, m_test)')
    lines.append(f'')
    lines.append(f'        all_y_true.extend(y_test)')
    lines.append(f'        all_y_pred.extend(y_pred)')
    lines.append(f'')
    lines.append(f'    return np.array(all_y_true), np.array(all_y_pred)')

    return '\n'.join(lines)


def generate_special_case(method_name, source):
    """为特殊情况生成逻辑"""
    # 检查是否有其他可用函数
    all_funcs = re.findall(r'def\s+(\w+)\s*\([^)]*\)', source)
    # 过滤掉私有方法和通用方法
    public_funcs = [f for f in all_funcs if not f.startswith('_') and f not in
                    ('compute_metrics', 'get_cmaq_at_site', 'get_cmaq_grid_coord',
                     'idw_predict', 'haversine_distance', 'main')]

    if public_funcs:
        # 尝试使用找到的函数
        func_name = public_funcs[0]
        return generate_from_run_func(method_name, func_name, source, None)

    # 最后的后备方案：通用OLS+IDW
    return '''
    # 通用融合逻辑（OLS + IDW残差插值）
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return np.array([]), np.array([])

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return np.array([]), np.array([])
    cmaq_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test_df) == 0 or len(train_df) == 0:
            continue

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # OLS线性校正
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual = y_train - ols.predict(m_train.reshape(-1, 1))
        pred_ols = ols.predict(m_test.reshape(-1, 1))

        # IDW残差插值
        residual_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists = np.sqrt((X_train[:, 0] - X_test[i, 0])**2 + (X_train[:, 1] - X_test[i, 1])**2)
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / (dists ** 2)
            weights = weights / weights.sum()
            residual_pred[i] = np.sum(weights * residual)

        y_pred = pred_ols + residual_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def generate_stage_main(method_name, safe_name):
    """生成stage验证和main函数"""
    return f'''

def run_stage_validation(stage_name, start_date, end_date):
    sep = "=" * 70
    print(sep)
    print(f"{method_name} Stage: {{stage_name}} ({{start_date}} ~ {{end_date}})")
    print(sep)

    base = BASELINE[stage_name]
    threshold_r2 = base['R2']
    print(f"VNA Baseline: R2={{base['R2']:.4f}}, RMSE={{base['RMSE']:.2f}}, MB={{base['MB']:.2f}}")

    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    print(f"Days: {{len(date_list)}}")

    n_jobs = min(8, len(date_list))
    results = Parallel(n_jobs=n_jobs)(
        delayed(ten_fold_{safe_name})(date_str)
        for date_str in date_list
    )

    all_y_true = []
    all_y_pred = []
    day_count = 0
    for y_true, y_pred in results:
        if len(y_true) > 0:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            day_count += 1

    print(f"Processed: {{day_count}} days, {{len(all_y_true)}} predictions")

    if len(all_y_true) == 0:
        return {{'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}}, False

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))

    r2_pass = metrics['R2'] > threshold_r2
    rmse_pass = metrics['RMSE'] <= base['RMSE']
    mb_pass = abs(metrics['MB']) <= abs(base['MB'])
    innovation_pass = r2_pass and rmse_pass and mb_pass

    print(f"Result: R2={{metrics['R2']:.4f}}, RMSE={{metrics['RMSE']:.2f}}, MB={{metrics['MB']:.2f}}")
    print(f"Check: R2>{{threshold_r2:.4f}}? {{'PASS' if r2_pass else 'FAIL'}} | RMSE<={{base['RMSE']}}? {{'PASS' if rmse_pass else 'FAIL'}} | |MB|<={{abs(base['MB'])}}? {{'PASS' if mb_pass else 'FAIL'}}")
    print(f"Innovation: {{'VERIFIED' if innovation_pass else 'NOT VERIFIED'}}")

    return metrics, innovation_pass


def main():
    sep = "=" * 70
    print(sep)
    print(f"{method_name} All Stages - 标准模式")
    print(sep)

    stages = {{
        'pre_exp':  ('2020-01-01', '2020-01-05'),
        'stage1':   ('2020-01-01', '2020-01-31'),
        'stage2':   ('2020-07-01', '2020-07-31'),
        'stage3':   ('2020-12-01', '2020-12-31'),
    }}

    results = {{}}
    all_pass = True

    for stage_name, (start, end) in stages.items():
        metrics, innovation_pass = run_stage_validation(stage_name, start, end)
        results[stage_name] = {{'metrics': metrics, '判定': {{'innovation_verified': innovation_pass}}}}
        if not innovation_pass:
            all_pass = False

    output_file = f'{{OUTPUT_DIR}}/{method_name}_all_stages.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(sep)
    print("SUMMARY")
    print(sep)
    for stage, data in results.items():
        m = data['metrics']
        status = 'VERIFIED' if data['判定']['innovation_verified'] else 'NOT VERIFIED'
        print(f"{{stage}}: R2={{m['R2']:.4f}}, RMSE={{m['RMSE']:.2f}}, MB={{m['MB']:.2f}} -> {{status}}")

    print(f"\\nAll stages passed: {{all_pass}}")
    print(f"Results saved: {{output_file}}")

    csv_rows = []
    for stage, data in results.items():
        m = data['metrics']
        csv_rows.append({{
            'method': '{method_name}',
            'stage': stage,
            'R2': m['R2'],
            'MAE': m['MAE'],
            'RMSE': m['RMSE'],
            'MB': m['MB'],
            'innovation_verified': data['判定']['innovation_verified']
        }})
    csv_df = pd.DataFrame(csv_rows)
    csv_file = f'{{OUTPUT_DIR}}/{method_name}_summary.csv'
    csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"CSV saved: {{csv_file}}")

    return results


if __name__ == '__main__':
    main()
'''


def main():
    print("=" * 70)
    print("Generate All Validation Scripts")
    print("=" * 70)

    generated = []
    skipped = []
    errors = []

    for method_name in METHODS:
        output_file = os.path.join(OUTPUT_DIR, f'{method_name}_十折标准模式.py')

        if os.path.exists(output_file):
            print(f"  SKIP: {method_name}")
            skipped.append(method_name)
            continue

        method_file = os.path.join(METHOD_DIR, f'{method_name}.py')
        if not os.path.exists(method_file):
            print(f"  MISSING: {method_name}")
            errors.append(method_name)
            continue

        try:
            script = generate_script_for_method(method_name)
            if script is None:
                print(f"  ERROR: {method_name} - could not generate")
                errors.append(method_name)
                continue

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(script)

            print(f"  OK: {method_name}")
            generated.append(method_name)
        except Exception as e:
            print(f"  ERROR: {method_name} - {e}")
            import traceback
            traceback.print_exc()
            errors.append(method_name)

    print("\n" + "=" * 70)
    print(f"Generated: {len(generated)}, Skipped: {len(skipped)}, Errors: {len(errors)}")
    if errors:
        print(f"Errors: {errors}")
    print("=" * 70)


if __name__ == '__main__':
    main()
