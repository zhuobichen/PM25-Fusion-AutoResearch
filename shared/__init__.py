# -*- coding: utf-8 -*-
"""
共享工具模块
============
提供项目中所有脚本共用的工具函数，消除代码重复。

模块：
- paths: 项目路径解析
- metrics: 评估指标计算
- geo_utils: 地理空间工具函数
"""

from shared.paths import get_project_root, data_path
from shared.metrics import compute_metrics
from shared.geo_utils import (
    get_cmaq_at_site,
    get_cmaq_grid_coord,
    haversine_distance,
    idw_interpolate,
)
