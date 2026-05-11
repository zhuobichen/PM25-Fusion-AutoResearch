# -*- coding: utf-8 -*-
"""Analyze method files to understand their structure"""
import re
import os

method_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'CodeWorkSpace', '新融合方法代码')
method_dir = os.path.abspath(method_dir)

methods = [
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

results = []
for m in methods:
    fpath = os.path.join(method_dir, m + '.py')
    if not os.path.exists(fpath):
        results.append(f'MISSING: {m}')
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        code = f.read()

    run_funcs = re.findall(r'def\s+(run_\w+|ten_fold_\w+)\s*\(', code)
    classes = re.findall(r'^class\s+(\w+)\s*[:\(]', code, re.MULTILINE)
    has_nna = 'NNA' in code or 'nna_methods' in code
    has_gpr = 'GaussianProcessRegressor' in code
    has_poly = 'PolynomialFeatures' in code
    has_ols = 'LinearRegression' in code

    results.append(f'{m}|funcs={run_funcs}|classes={classes}|nna={has_nna}|gpr={has_gpr}|poly={has_poly}|ols={has_ols}')

for r in results:
    print(r)
