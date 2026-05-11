# -*- coding: utf-8 -*-
"""Delete V1 generated scripts so V2 can regenerate them"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

deleted = 0
for m in METHODS:
    f = os.path.join(SCRIPT_DIR, f'{m}_十折标准模式.py')
    if os.path.exists(f):
        os.remove(f)
        deleted += 1
        print(f"  Deleted: {m}")

print(f"\nDeleted {deleted} scripts")
