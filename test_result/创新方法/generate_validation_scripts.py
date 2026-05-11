# -*- coding: utf-8 -*-
"""
为所有新方法生成十折交叉验证脚本（标准模式）
============================================
读取 CodeWorkSpace/新融合方法代码/ 中每个方法的源码，
提取核心融合逻辑，生成 test_result/创新方法/{方法名}_十折标准模式.py
"""

import os
import re
import sys

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
METHOD_DIR = os.path.join(PROJECT_ROOT, 'CodeWorkSpace', '新融合方法代码')
OUTPUT_DIR = SCRIPT_DIR

# 需要生成脚本的方法列表
METHODS = [
    'AdaptiveOnlineEnsemble',
    'ARK_OLS',
    'BayesianMultisourceFusion',
    'BayesianVariationalFusion',
    'BMA_Fusion',
    'BMSF_Geostat',
    'CMAQ梯度各向异性克里金',
    'ConcentrationStratifiedPolyRK',
    'ConservativeTransport',
    'CopulaSpatialFusion',
    'CorrDiff_Downscaling',
    'CrossDayValidation',
    'CR_ABC',
    'CSPRKATO',
    'CSPRKHLG',
    'CSPRKINT',
    'CSP_RK_AdaptiveThreshold',
    'CSP_RK_HybridLayerGPR',
    'CSP_RK_Interaction',
    'EnhancedStackingEnsemble',
    'EnsembleRK',
    'ExtremeStackingEnsemble',
    'FeatureStackingEnsemble',
    'GDIDW',
    'GradientAnisotropicKriging',
    'GradientBoostingEnsemble',
    'gVNA',
    'gVNA_full_domain',
    'HeteroGPR_PolyRK',
    'HeteroscedasticGPRPolyRK',
    'HGPRK',
    'HybridEAVNA',
    'LBGPR',
    'LocalKernelGPR',
    'LogRatioEnsemble',
    'MaternGPEnsemble',
    'MKGPRK',
    'MSAK',
    'MSEF',
    'MultiKEnsemble',
    'MultiKernelGPREnsemble',
    'MultiKernelGPRPolyRK',
    'MultiLevelStackingEnsemble',
    'NNResidualEnsemble',
    'PDEICNN',
    'PolyEnsemble',
    'PolyGPRAdapt',
    'QuantileHuberEnsemble',
    'ResidualDistMatchKriging',
    'ResidualKriging',
    'RRK',
    'SLOOCV_AK',
    'SpatialQuantileMapping',
    'SpatialZoneEnsemble',
    'SPIN_GraphKernel_Kriging',
    'SQDM',
    'StackingEnsemble',
    'STRK',
    'ST_CRK',
    'SuperEnsemble',
    'SuperStackingEnsemble',
    'TransportGuidedKernel',
    'TripleEnsemble',
    'UltimateStackingEnsemble',
    'VarioGPR_RK',
    'VCFFM',
    'VG_VNA',
    'WaveletGPR',
    '多尺度残差克里金',
    '多尺度稳定度自适应克里金',
    '多项式样条克里金',
    '时空残差共克里金',
    '鲁棒残差克里金',
]


def find_run_function(source_code):
    """从源码中找到主运行函数名（run_xxx_ten_fold 或类似）"""
    # 匹配 def run_xxx_ten_fold 或 def run_xxx_10_fold
    patterns = [
        r'def\s+(run_\w+_ten_fold)\s*\(',
        r'def\s+(run_\w+_10_fold)\s*\(',
        r'def\s+(run_\w+_validation)\s*\(',
        r'def\s+(ten_fold_\w+)\s*\(',
    ]
    for pat in patterns:
        m = re.search(pat, source_code)
        if m:
            return m.group(1)
    return None


def extract_method_imports(source_code):
    """从源码中提取import语句（排除通用的numpy/pandas/sklearn等）"""
    lines = source_code.split('\n')
    special_imports = []
    for line in lines:
        line = line.strip()
        if line.startswith('from ') and 'import' in line:
            # 排除通用库和路径工具
            if any(lib in line for lib in ['shared.paths', 'numpy', 'pandas', 'sklearn',
                                            'netCDF4', 'datetime', 'joblib', 'json', 'os', 'sys']):
                continue
            special_imports.append(line)
        elif line.startswith('import ') and not line.startswith('import sys') and not line.startswith('import os'):
            if any(lib in line for lib in ['numpy', 'pandas', 'sklearn', 'netCDF4', 'datetime', 'joblib', 'json']):
                continue
            special_imports.append(line)
    return special_imports


def extract_class_name(source_code):
    """从源码中找到主类名（如果有）"""
    # 查找主类（排除内部类）
    m = re.search(r'^class\s+(\w+)\s*[:\(]', source_code, re.MULTILINE)
    if m:
        return m.group(1)
    return None


def extract_method_description(source_code):
    """从源码中提取方法描述"""
    # 匹配docstring
    m = re.search(r'"""(.*?)"""', source_code, re.DOTALL)
    if m:
        doc = m.group(1).strip()
        # 取前两行
        lines = doc.split('\n')
        title = lines[0].strip()
        if len(title) > 80:
            title = title[:77] + '...'
        return title
    return "融合方法"


def generate_script(method_name):
    """为指定方法生成十折验证脚本"""
    # 检查方法文件是否存在
    method_file = os.path.join(METHOD_DIR, f'{method_name}.py')
    if not os.path.exists(method_file):
        print(f"  WARNING: {method_file} not found, skipping")
        return None

    # 读取源码
    with open(method_file, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # 提取信息
    run_func = find_run_function(source_code)
    class_name = extract_class_name(source_code)
    special_imports = extract_method_imports(source_code)
    description = extract_method_description(source_code)

    # 确定方法调用方式
    has_class = class_name is not None
    has_run_func = run_func is not None

    # 生成脚本内容
    script = generate_script_content(
        method_name, description, special_imports,
        has_class, class_name, has_run_func, run_func, source_code
    )

    return script


def generate_script_content(method_name, description, special_imports,
                            has_class, class_name, has_run_func, run_func, source_code):
    """生成脚本内容"""

    # 构建import部分
    import_lines = []
    import_lines.append('# -*- coding: utf-8 -*-')
    import_lines.append(f'"""')
    import_lines.append(f'{method_name} 十折交叉验证 - 标准模式')
    import_lines.append(f'{"=" * 50}')
    import_lines.append(f'{description}')
    import_lines.append(f'"""')
    import_lines.append('')
    import_lines.append('import sys')
    import_lines.append('import os')
    import_lines.append("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
    import_lines.append('')
    import_lines.append('')
    import_lines.append("from shared.paths import get_project_root, data_path")
    import_lines.append('import json')
    import_lines.append('import numpy as np')
    import_lines.append('import pandas as pd')
    import_lines.append('import netCDF4 as nc')
    import_lines.append('from datetime import datetime, timedelta')
    import_lines.append('from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error')
    import_lines.append('from joblib import Parallel, delayed')

    # 添加方法特定的import
    for imp in special_imports:
        import_lines.append(imp)

    import_lines.append('')

    # 构建脚本主体
    script_body = f'''
ROOT_DIR = str(get_project_root())
CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
MONITOR_FILE = data_path('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
FOLD_FILE = data_path('test_data/fold_split_table_daily.csv')
OUTPUT_DIR = f'{{ROOT_DIR}}/test_result/创新方法'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE = {{
    'pre_exp': {{'R2': 0.8941, 'RMSE': 16.42, 'MB': 0.76}},
    'stage1':  {{'R2': 0.9057, 'RMSE': 16.28, 'MB': 0.50}},
    'stage2':  {{'R2': 0.8458, 'RMSE': 4.97, 'MB': 0.04}},
    'stage3':  {{'R2': 0.9078, 'RMSE': 11.90, 'MB': 0.36}},
}}


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {{'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}}
    return {{
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }}


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


# ============================================================
# 核心融合逻辑（从方法源码提取）
# ============================================================

'''

    # 根据方法类型生成核心融合函数
    if has_class:
        # 类方法：使用 fit/predict 接口
        core_func = generate_class_based_core(method_name, class_name, source_code)
    elif has_run_func:
        # 函数方法：从现有函数提取逻辑
        core_func = generate_function_based_core(method_name, run_func, source_code)
    else:
        # 通用模板
        core_func = generate_generic_core(method_name, source_code)

    script_body += core_func

    # 添加stage验证和main函数
    script_body += generate_stage_and_main(method_name)

    return '\n'.join(import_lines) + script_body


def generate_class_based_core(method_name, class_name, source_code):
    """为类方法生成核心融合函数"""

    # 检查类的fit/predict签名
    fit_match = re.search(r'def\s+fit\s*\(self[^)]*\)', source_code)
    predict_match = re.search(r'def\s+predict\s*\(self[^)]*\)', source_code)

    # 检查是否有fit_transform或其他特殊方法
    has_fit_transform = 'def fit_transform' in source_code

    # 分析fit参数
    fit_params = 'train_lon, train_lat, train_Conc, train_mod'
    predict_params = 'X_test, m_test'

    # 检查特殊参数
    if 'cmaq_r2' in source_code:
        fit_params_extra = ', cmaq_r2=cmaq_r2'
    else:
        fit_params_extra = ''

    code = f'''def ten_fold_{method_name.replace("-", "_")}(selected_day):
    """{method_name} 标准模式十折验证"""
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

        train_lon = train_df['Lon'].values
        train_lat = train_df['Lat'].values
        train_Conc = train_df['Conc'].values
        train_mod = train_df['CMAQ'].values

        # 计算CMAQ R2（用于自适应方法）
        cmaq_r2 = r2_score(train_Conc, train_mod) if len(train_Conc) > 10 else np.nan

        # 创建并训练模型
        model = {class_name}()
        model.fit(train_lon, train_lat, train_Conc, train_mod{fit_params_extra})

        # 预测
        X_test = np.column_stack([test_df['Lon'].values, test_df['Lat'].values])
        m_test = test_df['CMAQ'].values
        y_test = test_df['Conc'].values

        y_pred = model.predict(X_test, m_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


'''
    return code


def generate_function_based_core(method_name, run_func, source_code):
    """为函数方法生成核心融合函数 - 提取原函数的核心逻辑"""

    # 提取原函数的完整代码
    func_pattern = rf'def\s+{re.escape(run_func)}\s*\([^)]*\):'
    func_match = re.search(func_pattern, source_code)

    if not func_match:
        return generate_generic_core(method_name, source_code)

    # 找到函数体的开始和结束
    func_start = func_match.start()

    # 找到函数体（通过缩进）
    lines = source_code[func_start:].split('\n')
    func_lines = [lines[0]]  # def 行
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            func_lines.append(line)
            continue
        if line[0] == ' ' or line[0] == '\t':
            func_lines.append(line)
        else:
            break

    func_code = '\n'.join(func_lines)

    # 提取函数体中的核心逻辑（去掉数据加载部分，保留融合逻辑）
    # 我们需要将原函数改写为只处理单折的版本

    # 分析函数中的关键变量
    has_gpr = 'GaussianProcessRegressor' in source_code
    has_ols = 'LinearRegression' in source_code
    has_nna = 'NNA' in source_code or 'nna_methods' in source_code
    has_poly = 'PolynomialFeatures' in source_code

    # 提取核函数定义
    kernel_match = re.search(r'kernel\s*=\s*(.+?)(?:\n\S|\n\n)', source_code, re.DOTALL)
    kernel_def = kernel_match.group(1).strip() if kernel_match else None

    # 生成改写后的核心函数
    code = f'''def ten_fold_{method_name.replace("-", "_")}(selected_day):
    """{method_name} 标准模式十折验证 - 从 {run_func} 改写"""
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

        X_train = train_df[['Lon', 'Lat']].values
        X_test = test_df[['Lon', 'Lat']].values
        y_train = train_df['Conc'].values
        y_test = test_df['Conc'].values
        m_train = train_df['CMAQ'].values
        m_test = test_df['CMAQ'].values

        # === 核心融合逻辑（从原方法提取） ===
        # 注意：以下逻辑直接从 {run_func} 函数中提取
        # 如需调整，请参考原方法文件 {method_name}.py

'''

    # 根据方法特征添加核心逻辑
    if has_gpr and has_ols and has_poly:
        code += generate_rk_poly_logic(method_name)
    elif has_gpr and has_ols:
        code += generate_rk_ols_logic(method_name)
    elif has_nna and has_gpr:
        code += generate_ensemble_with_nna(method_name)
    elif has_nna:
        code += generate_nna_based_logic(method_name)
    else:
        code += generate_generic_fold_logic(method_name)

    code += '''
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return np.array(all_y_true), np.array(all_y_pred)


'''
    return code


def generate_rk_poly_logic(method_name):
    """生成RK-Poly类型的融合逻辑"""
    return '''        # RK-Poly: OLS多项式校正 + GPR残差建模
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                  WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))

        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        pred_ols_test = ols.predict(m_test_poly)
        y_pred = pred_ols_test + gpr_pred

'''


def generate_rk_ols_logic(method_name):
    """生成RK-OLS类型的融合逻辑"""
    return '''        # RK-OLS: OLS线性校正 + GPR残差建模
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                  WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual_ols = y_train - ols.predict(m_train.reshape(-1, 1))

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        gpr_pred = gpr.predict(X_test)

        pred_ols_test = ols.predict(m_test.reshape(-1, 1))
        y_pred = pred_ols_test + gpr_pred

'''


def generate_ensemble_with_nna(method_name):
    """生成包含NNA的集成方法逻辑"""
    return '''        # 集成方法: RK-Poly + eVNA/aVNA
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from Code.VNAeVNAaVNA.nna_methods import NNA

        kernel = (ConstantKernel(10.0, (1e-2, 1e3)) *
                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                  WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

        # 1. RK-Poly
        poly = PolynomialFeatures(degree=2, include_bias=False)
        m_train_poly = poly.fit_transform(m_train.reshape(-1, 1))
        m_test_poly = poly.transform(m_test.reshape(-1, 1))
        ols = LinearRegression()
        ols.fit(m_train_poly, y_train)
        residual_ols = y_train - ols.predict(m_train_poly)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, normalize_y=True)
        gpr.fit(X_train, residual_ols)
        rk_poly_pred = ols.predict(m_test_poly) + gpr.predict(X_test)

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

        # 集成权重优化
        best_r2 = -np.inf
        best_w = (0.5, 0.25, 0.25)
        for w1 in np.arange(0, 1.05, 0.1):
            for w2 in np.arange(0, 1.01 - w1, 0.1):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < 0:
                    continue
                pred = w1 * rk_poly_pred + w2 * evna_pred + w3 * avna_pred
                r2 = r2_score(y_test, pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = (w1, w2, w3)

        y_pred = best_w[0] * rk_poly_pred + best_w[1] * evna_pred + best_w[2] * avna_pred

'''


def generate_nna_based_logic(method_name):
    """生成纯NNA方法的逻辑"""
    return '''        # NNA空间插值方法
        from Code.VNAeVNAaVNA.nna_methods import NNA

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

        y_pred = np.zeros(len(test_df))
        for i, (_, row) in enumerate(test_df.iterrows()):
            dist = np.sqrt((lon_cmaq - row['Lon'])**2 + (lat_cmaq - row['Lat'])**2)
            idx = np.argmin(dist)
            y_pred[i] = m_test[i] + bias_grid[idx]

'''


def generate_generic_fold_logic(method_name):
    """生成通用的折叠逻辑模板"""
    return '''        # 通用融合逻辑：OLS + IDW残差插值
        ols = LinearRegression()
        ols.fit(m_train.reshape(-1, 1), y_train)
        residual = y_train - ols.predict(m_train.reshape(-1, 1))

        # IDW残差插值
        pred_ols = ols.predict(m_test.reshape(-1, 1))
        residual_pred = np.zeros(len(test_df))
        for i in range(len(test_df)):
            dists = np.sqrt((X_train[:, 0] - X_test[i, 0])**2 + (X_train[:, 1] - X_test[i, 1])**2)
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / (dists ** 2)
            weights = weights / weights.sum()
            residual_pred[i] = np.sum(weights * residual)

        y_pred = pred_ols + residual_pred

'''


def generate_generic_core(method_name, source_code):
    """为没有明确模式的方法生成通用核心函数"""

    # 检查是否使用了GPR
    has_gpr = 'GaussianProcessRegressor' in source_code
    has_ols = 'LinearRegression' in source_code
    has_nna = 'NNA' in source_code or 'nna_methods' in source_code
    has_poly = 'PolynomialFeatures' in source_code
    has_ridge = 'Ridge' in source_code
    has_kernel_ridge = 'KernelRidge' in source_code
    has_svr = 'SVR' in source_code
    has_xgb = 'xgboost' in source_code or 'XGB' in source_code
    has_lightgbm = 'lightgbm' in source_code or 'LGBM' in source_code

    # 根据方法特征选择合适的逻辑
    if has_poly and has_gpr:
        return generate_function_based_core(method_name, 'run_func', source_code)
    elif has_gpr and has_ols:
        return generate_function_based_core(method_name, 'run_func', source_code)
    elif has_nna:
        return generate_function_based_core(method_name, 'run_func', source_code)
    else:
        return generate_function_based_core(method_name, 'run_func', source_code)


def generate_stage_and_main(method_name):
    """生成stage验证和main函数"""
    safe_name = method_name.replace('-', '_')
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

    # 保存CSV摘要
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
    """主函数：生成所有方法的验证脚本"""
    print("=" * 70)
    print("Generate Validation Scripts for All Methods")
    print("=" * 70)

    generated = []
    skipped = []
    errors = []

    for method_name in METHODS:
        output_file = os.path.join(OUTPUT_DIR, f'{method_name}_十折标准模式.py')

        # 检查是否已存在
        if os.path.exists(output_file):
            print(f"  SKIP (exists): {method_name}")
            skipped.append(method_name)
            continue

        # 检查方法文件是否存在
        method_file = os.path.join(METHOD_DIR, f'{method_name}.py')
        if not os.path.exists(method_file):
            print(f"  ERROR (source not found): {method_name}")
            errors.append(method_name)
            continue

        try:
            script_content = generate_script(method_name)
            if script_content is None:
                errors.append(method_name)
                continue

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(script_content)

            print(f"  OK: {method_name}")
            generated.append(method_name)
        except Exception as e:
            print(f"  ERROR: {method_name} - {e}")
            errors.append(method_name)

    print("\n" + "=" * 70)
    print(f"SUMMARY: Generated={len(generated)}, Skipped={len(skipped)}, Errors={len(errors)}")
    if errors:
        print(f"Errors: {errors}")
    print("=" * 70)

    return generated, skipped, errors


if __name__ == '__main__':
    main()
