# -*- coding: utf-8 -*-
"""
项目路径工具
============
基于 __file__ 的相对路径解析，替代所有硬编码绝对路径。

使用方式：
    from shared.paths import get_project_root, data_path

    ROOT_DIR = str(get_project_root())
    CMAQ_FILE = data_path('test_data/raw/CMAQ/2020_PM25.nc')
"""

from pathlib import Path as _Path

# 项目根目录 = shared/ 的父目录
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent


def get_project_root() -> _Path:
    """返回项目根目录的 Path 对象"""
    return _PROJECT_ROOT


def data_path(relative_path: str) -> str:
    """
    将相对于项目根目录的路径转换为绝对路径。

    参数:
        relative_path: 相对于项目根目录的路径，如 'test_data/fold_split_table_daily.csv'

    返回:
        str: 绝对路径
    """
    return str(_PROJECT_ROOT / relative_path)
