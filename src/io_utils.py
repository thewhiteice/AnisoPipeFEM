"""
读取设置及保存结果工具文件
    _build_C_from_yaml()    根据YAML设置构建刚度矩阵
    load_config()           读取单个 YAML 配置文件
    make_result_dir()       创建结果文件夹
    save_json()             保存JSON
    save_fig()              保存图片
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.material_utils import build_stiffness


def _build_C_from_yaml(cfg):
    """根据 YAML 中的 stiffness 配置构建刚度矩阵 (6,6)，单位 Pa。"""
    typ = cfg["type"]

    if typ == "matrix":
        C = np.array(cfg["C"], dtype=float)
        return C * cfg.get("scale", 1.0)

    if typ == "engineering":
        E = np.array(cfg["E"], dtype=float)
        nu = np.array(cfg["nu"], dtype=float)

        G = []
        for i, g in enumerate(cfg["G"]):
            if g == "auto":
                G.append(E[1] / (2 * (1 + nu[2])))
            else:
                G.append(float(g))
        G = np.array(G, dtype=float)

        return build_stiffness(E, nu, G)

    raise ValueError(f"Unknown stiffness type: {typ}")


def load_config(path):
    """
    读取单个 YAML 配置文件，展开铺层。

    返回:
        C_list:           (n, 6, 6) 刚度矩阵
        r_interface_list: (n+1,)   界面半径
        theta_rad_list:   (n,)     铺层角度 (rad)
        failure:          list[dict]，每层失效参数
        name:             str      显示名
        slug:             str      机器名（用于目录）
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    slug = cfg.get("slug") or name

    materials = cfg["materials"]

    angles_deg = []
    thicknesses = []
    mats = []

    # 展开段式铺层
    for seg in cfg["layup"]:
        ang = seg["angles_deg"]
        if isinstance(ang, (int, float)):
            ang = [ang]

        rep = seg.get("repeat", 1)
        th = seg["thickness"]
        mat = seg["material"]

        for _ in range(rep):
            angles_deg.extend(ang)
            thicknesses.extend([th] * len(ang))
            mats.extend([mat] * len(ang))

    C_list = []
    failure = []
    for m in mats:
        mat_cfg = materials[m]
        C_list.append(_build_C_from_yaml(mat_cfg["stiffness"]))
        failure.append(copy.deepcopy(mat_cfg["failure"]))

    C_list = np.array(C_list)
    theta_rad_list = np.deg2rad(angles_deg)

    r_i = cfg["geometry"]["inner_radius"]
    r_interface_list = np.concatenate(([r_i], r_i + np.cumsum(thicknesses)))

    return C_list, r_interface_list, theta_rad_list, failure, name, slug


def make_result_dir(root, slug, subdir=""):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(root) / subdir / f"{slug}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(out_dir, data, filename="result.json"):
    with open(Path(out_dir) / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_fig(out_dir, fig, filename):
    fig.savefig(Path(out_dir) / filename, dpi=300, bbox_inches="tight")
