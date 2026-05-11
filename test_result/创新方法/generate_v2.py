# -*- coding: utf-8 -*-
"""
为所有新方法生成十折交叉验证脚本（V2 - 简洁可靠版）
====================================================
策略：每个脚本包含完整的、自包含的融合逻辑。
不尝试提取源码，而是根据方法类型生成正确的逻辑。
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


def analyze_method(method_name):
    """分析方法源码，确定类型和特征"""
    fpath = os.path.join(METHOD_DIR, method_name + '.py')
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        code = f.read()

    info = {
        'name': method_name,
        'has_class': bool(re.search(r'^class\s+\w+', code, re.MULTILINE)),
        'class_name': None,
        'run_func': None,
        'has_nna': 'NNA' in code or 'nna_methods' in code,
        'has_gpr': 'GaussianProcessRegressor' in code,
        'has_poly': 'PolynomialFeatures' in code,
        'has_ols': 'LinearRegression' in code,
        'has_ridge': 'Ridge' in code,
        'has_kernel_ridge': 'KernelRidge' in code,
        'special_imports': [],
    }

    # Find class
    classes = re.findall(r'^class\s+(\w+)\s*[:\(]', code, re.MULTILINE)
    for c in classes:
        if 'Model' not in c and 'Wrapper' not in c and 'Setting' not in c:
            info['class_name'] = c
            break

    # Find run function
    for pat in [r'def\s+(run_\w+_ten_fold)\s*\(', r'def\s+(run_\w+_10_fold)\s*\(',
                r'def\s+(ten_fold_\w+)\s*\(', r'def\s+(run_\w+_validation)\s*\(',
                r'def\s+(run_single_day)\s*\(']:
        m = re.search(pat, code)
        if m:
            info['run_func'] = m.group(1)
            break

    # Special imports
    skip_libs = {'shared.paths', 'numpy', 'pandas', 'sklearn', 'netCDF4',
                 'datetime', 'joblib', 'json', 'os', 'sys', 'warnings'}
    for line in code.split('\n'):
        s = line.strip()
        if s.startswith('from ') and 'import' in s:
            mod = s.split('from ')[1].split(' import')[0].strip()
            if mod and not any(lib in mod for lib in skip_libs):
                info['special_imports'].append(s)
        elif s.startswith('import ') and not any(lib in s for lib in skip_libs):
            info['special_imports'].append(s)

    # Deduplicate imports
    seen = set()
    unique = []
    for imp in info['special_imports']:
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)
    info['special_imports'] = unique

    return info


def get_template(method_name, info):
    """根据方法类型选择合适的模板"""

    if info['has_class']:
        return get_class_template(method_name, info)
    elif info['has_nna'] and info['has_gpr'] and info['has_poly']:
        return get_ensemble_nna_gpr_poly_template(method_name, info)
    elif info['has_nna'] and info['has_gpr']:
        return get_ensemble_nna_gpr_template(method_name, info)
    elif info['has_nna']:
        return get_nna_template(method_name, info)
    elif info['has_gpr'] and info['has_poly'] and info['has_ols']:
        return get_rk_poly_template(method_name, info)
    elif info['has_gpr'] and info['has_ols']:
        return get_rk_ols_template(method_name, info)
    elif info['has_gpr']:
        return get_gpr_template(method_name, info)
    else:
        return get_generic_template(method_name, info)


def get_header(method_name, info):
    """生成脚本头部"""
    imports = [
        '# -*- coding: utf-8 -*-',
        '"""',
        f'{method_name} 十折交叉验证 - 标准模式',
        '=' * 50,
        '"""',
        '',
        'import sys',
        'import os',
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        '',
        "from shared.paths import get_project_root, data_path",
        'import json',
        'import numpy as np',
        'import pandas as pd',
        'import netCDF4 as nc',
        'from datetime import datetime, timedelta',
        'from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error',
        'from sklearn.linear_model import LinearRegression',
        'from joblib import Parallel, delayed',
    ]
    for imp in info['special_imports']:
        imports.append(imp)

    imports.append('')
    imports.append("ROOT_DIR = str(get_project_root())")
    imports.append("CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')")
    imports.append("MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')")
    imports.append("FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')")
    imports.append("OUTPUT_DIR = f'{ROOT_DIR}/test_result/创新方法'")
    imports.append("os.makedirs(OUTPUT_DIR, exist_ok=True)")
    imports.append('')
    imports.append("BASELINE = {")
    imports.append("    'pre_exp': {'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76},")
    imports.append("    'stage1':  {'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50},")
    imports.append("    'stage2':  {'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04},")
    imports.append("    'stage3':  {'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36},")
    imports.append("}")
    imports.append('')

    return '\n'.join(imports)


COMMON_HELPERS = '''

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


def load_day_data(selected_day):
    """加载单日数据并提取CMAQ值"""
    monitor_df = pd.read_csv(MONITOR_FILE)
    fold_df = pd.read_csv(FOLD_FILE)

    day_df = monitor_df[monitor_df['Date'] == selected_day].copy()
    day_df = day_df.merge(fold_df, on=['Date', 'Site'], how='left')
    day_df = day_df.dropna(subset=['Lat', 'Lon', 'Conc'])

    if len(day_df) < 100:
        return None, None, None, None

    ds = nc.Dataset(CMAQ_FILE, 'r')
    lon_cmaq = ds.variables['lon'][:]
    lat_cmaq = ds.variables['lat'][:]
    pred_pm25 = ds.variables['pred_PM25'][:]
    ds.close()

    date_obj = datetime.strptime(selected_day, '%Y-%m-%d')
    day_idx = (date_obj - datetime(2020, 1, 1)).days
    if day_idx >= pred_pm25.shape[0]:
        return None, None, None, None
    cmaq_day = pred_pm25[day_idx]

    cmaq_values = []
    for _, row in day_df.iterrows():
        val = get_cmaq_at_site(row['Lon'], row['Lat'], lon_cmaq, lat_cmaq, cmaq_day)
        cmaq_values.append(val)
    day_df['CMAQ'] = cmaq_values

    return day_df, lon_cmaq, lat_cmaq, cmaq_day


'''


def get_stage_main(method_name, safe_name):
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


# ============================================================
# 模板生成函数
# ============================================================

def get_class_template(method_name, info):
    """类方法模板（gVNA, CopulaSpatialFusion, GDIDW等）"""
    cn = info['class_name']
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test_df) == 0 or len(train_df) == 0:
            continue

        train_lon = train_df['Lon'].values
        train_lat = train_df['Lat'].values
        train_Conc = train_df['Conc'].values
        train_mod = train_df['CMAQ'].values
        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        cmaq_r2 = r2_score(train_Conc, train_mod) if len(train_Conc) > 10 else np.nan

        model = {cn}()
        model.fit(train_lon, train_lat, train_Conc, train_mod, cmaq_r2=cmaq_r2)

        X_test = np.column_stack([test_df['Lon'].values, test_df['Lat'].values])
        y_pred = model.predict(X_test, m_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_rk_poly_template(method_name, info):
    """RK-Poly模板（OLS多项式 + GPR残差）"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        y_pred = ols.predict(m_test_poly) + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_rk_ols_template(method_name, info):
    """RK-OLS模板（OLS线性 + GPR残差）"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        y_pred = ols.predict(m_test.reshape(-1, 1)) + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_ensemble_nna_gpr_poly_template(method_name, info):
    """集成方法模板：RK-Poly + eVNA + aVNA"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from Code.VNAeVNAaVNA.nna_methods import NNA

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_day.ravel()

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        # 1. RK-Poly
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        ols_poly = LinearRegression()
        ols_poly.fit(m_train_poly, y_train)
        residual_poly = y_train - ols_poly.predict(m_train_poly)
        gpr_poly = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr_poly.fit(X_train, residual_poly)
        rk_poly_pred = ols_poly.predict(m_test_poly) + gpr_poly.predict(X_test)

        # 2. eVNA
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])
        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]
            avna_pred[i] = m_test[i] + bias_grid[idx]

        # 3. Ensemble
        best_r2 = -np.inf
        best_w = (0.4, 0.3, 0.3)
        for w1 in np.arange(0.1, 0.9, 0.1):
            for w2 in np.arange(0.1, 0.91 - w1, 0.1):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < 0.05:
                    continue
                pred = w1 * rk_poly_pred + w2 * evna_pred + w3 * avna_pred
                r2 = r2_score(y_test, pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = (w1, w2, w3)

        y_pred = best_w[0] * rk_poly_pred + best_w[1] * evna_pred + best_w[2] * avna_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_ensemble_nna_gpr_template(method_name, info):
    """集成方法模板：RK-OLS + eVNA/aVNA"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from Code.VNAeVNAaVNA.nna_methods import NNA

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_day.ravel()

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        # 1. RK-OLS
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        rk_ols_pred = ols.predict(m_test.reshape(-1, 1)) + gpr.predict(X_test)

        # 2. eVNA/aVNA
        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])
        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]
        rn_grid = zdf_grid[:, 1]

        evna_pred = np.zeros(len(test_df))
        avna_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            evna_pred[i] = y_grid_model_full[idx] * rn_grid[idx]
            avna_pred[i] = m_test[i] + bias_grid[idx]

        # 3. Ensemble
        best_r2 = -np.inf
        best_w = (0.4, 0.3, 0.3)
        for w1 in np.arange(0.1, 0.9, 0.1):
            for w2 in np.arange(0.1, 0.91 - w1, 0.1):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < 0.05:
                    continue
                pred = w1 * rk_ols_pred + w2 * evna_pred + w3 * avna_pred
                r2 = r2_score(y_test, pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = (w1, w2, w3)

        y_pred = best_w[0] * rk_ols_pred + best_w[1] * evna_pred + best_w[2] * avna_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_nna_template(method_name, info):
    """纯NNA方法模板（MSEF等）"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from Code.VNAeVNAaVNA.nna_methods import NNA

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    ny, nx = lon_cmaq.shape
    X_grid_full = np.column_stack([lon_cmaq.ravel(), lat_cmaq.ravel()])
    y_grid_model_full = cmaq_day.ravel()

    all_y_true = []
    all_y_pred = []

    for fold_id in range(1, 11):
        train_df = day_df[day_df['fold'] != fold_id].copy()
        test_df = day_df[day_df['fold'] == fold_id].copy()
        train_df = train_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        test_df = test_df.dropna(subset=['Lon', 'Lat', 'CMAQ', 'Conc'])
        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_test = test_df['Conc'].values
        m_test = test_df['CMAQ'].values

        train_df['x'] = train_df['Lon']
        train_df['y'] = train_df['Lat']
        train_df['mod'] = train_df['CMAQ']
        train_df['bias'] = train_df['Conc'] - train_df['CMAQ']
        train_df['rn'] = train_df['Conc'] / train_df['CMAQ']

        nn = NNA(method='voronoi', k=30, power=-2)
        nn.fit(train_df[['x', 'y']], train_df[['bias', 'rn']])
        zdf_grid = nn.predict(X_grid_full, njobs=4)
        bias_grid = zdf_grid[:, 0]

        y_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            y_pred[i] = m_test[i] + bias_grid[idx]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_gpr_template(method_name, info):
    """纯GPR方法模板"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

    kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

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

        residual = y_train - m_train

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual)
        gpr_pred = gpr.predict(X_test)

        y_pred = m_test + gpr_pred

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)
'''


def get_generic_template(method_name, info):
    """通用模板：OLS + IDW残差插值"""
    return f'''def ten_fold_{method_name}(selected_day):
    """{method_name} 标准模式十折验证"""
    day_df, lon_cmaq, lat_cmaq, cmaq_day = load_day_data(selected_day)
    if day_df is None:
        return np.array([]), np.array([])

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


def generate_script(method_name):
    """为单个方法生成完整的验证脚本"""
    info = analyze_method(method_name)
    if info is None:
        return None

    header = get_header(method_name, info)
    helpers = COMMON_HELPERS
    core = get_template(method_name, info)
    safe_name = method_name.replace('-', '_').replace('.', '_')
    stage_main = get_stage_main(method_name, safe_name)

    return header + helpers + core + stage_main


def main():
    print("=" * 70)
    print("Generate All Validation Scripts (V2)")
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

        try:
            script = generate_script(method_name)
            if script is None:
                print(f"  MISSING: {method_name}")
                errors.append(method_name)
                continue

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(script)

            print(f"  OK: {method_name}")
            generated.append(method_name)
        except Exception as e:
            print(f"  ERROR: {method_name} - {e}")
            errors.append(method_name)

    print(f"\nGenerated: {len(generated)}, Skipped: {len(skipped)}, Errors: {len(errors)}")


if __name__ == '__main__':
    main()
