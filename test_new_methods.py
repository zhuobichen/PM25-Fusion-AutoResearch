#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试新创建的方法
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_data_utils():
    """测试数据工具模块"""
    print("=" * 60)
    print("测试 1: 数据工具模块")
    print("=" * 60)
    
    try:
        from shared.data_utils import get_project_paths, load_daily_data
        print("✓ 成功导入 data_utils")
        
        paths = get_project_paths()
        print(f"✓ 项目根路径: {paths['root']}")
        print(f"✓ CMAQ文件: {paths['cmaq_file']}")
        print(f"✓ 输出目录: {paths['output_dir']}")
        
        # 测试加载数据（如果数据存在）
        if os.path.exists(paths['cmaq_file']) and os.path.exists(paths['monitor_file']):
            day_df, lon_cmaq, lat_cmaq, pred_day = load_daily_data('2020-01-01', paths)
            if day_df is not None:
                print(f"✓ 成功加载数据: {len(day_df)} 条记录")
            else:
                print("⚠ 数据加载返回None（可能数据不存在）")
        else:
            print("⚠ 数据文件不存在，跳过数据加载测试")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_csprk_adaptive():
    """测试CSPRK自适应版本"""
    print("\n" + "=" * 60)
    print("测试 2: CSPRK_Adaptive 模块")
    print("=" * 60)
    
    try:
        from CodeWorkSpace.新融合方法代码.CSPRK_Adaptive import CSPRK_Adaptive, get_adaptive_thresholds
        print("✓ 成功导入 CSPRK_Adaptive")
        
        # 测试阈值计算函数
        import numpy as np
        test_data = np.random.normal(50, 20, 100)
        thresholds = get_adaptive_thresholds(test_data, n_layers=3)
        print(f"✓ 阈值计算正常: {thresholds}")
        
        # 测试模型初始化
        model = CSPRK_Adaptive(poly_degree=2, n_layers=3)
        print(f"✓ 模型初始化成功")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csp_advanced_rk():
    """测试CSP-AdvancedRK融合方法"""
    print("\n" + "=" * 60)
    print("测试 3: CSP_AdvancedRK 模块")
    print("=" * 60)
    
    try:
        from CodeWorkSpace.新融合方法代码.CSP_AdvancedRK import CSP_AdvancedRK
        print("✓ 成功导入 CSP_AdvancedRK")
        
        # 测试模型初始化
        model = CSP_AdvancedRK(poly_degree=2, n_layers=3, matern_nu=1.5)
        print(f"✓ 模型初始化成功")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """测试评估指标模块"""
    print("\n" + "=" * 60)
    print("测试 4: 评估指标模块")
    print("=" * 60)
    
    try:
        from shared.metrics import compute_metrics
        print("✓ 成功导入 compute_metrics")
        
        # 测试指标计算
        import numpy as np
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = compute_metrics(y_true, y_pred)
        print(f"✓ 指标计算: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("PM2.5 数据融合项目 - 新方法测试")
    print("=" * 60)
    
    results = {}
    
    # 运行所有测试
    results['data_utils'] = test_data_utils()
    results['metrics'] = test_metrics()
    results['csprk_adaptive'] = test_csprk_adaptive()
    results['csp_advanced_rk'] = test_csp_advanced_rk()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查")
        return 1


if __name__ == '__main__':
    sys.exit(main())
