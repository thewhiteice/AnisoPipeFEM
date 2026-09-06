"""
set_config() 函数
"""

import numpy as np


def setup_config1():
    """
    作用:
        返回工况1几何和材料参数
        层数3层:
            1. 层厚10mm, 铺层角90deg
            2. 层厚4mm, 铺层角12deg
            3. 层厚3mm, 铺层角67deg
        半径191mm
    输出:
        C_list:  刚度矩阵 (3, 6, 6)
        C_edit_list: 含体积分数刚度矩阵 (3, 6, 6)
        r_interface_list:   界面半径位置 (1,) (m)
        theta:  铺层角度 (1,) (rad)
        failure: 材料失效参数 dict
        name: 'config1'
    """

    C = (
        np.array(
            [
                [159.9998, 3.3195, 3.8702, 0, 0, 0],
                [3.3195, 10.4519, 2.5826, 0, 0, 0],
                [3.8702, 2.5826, 10.4767, 0, 0, 0],
                [0, 0, 0, 5.25, 0, 0],
                [0, 0, 0, 0, 3.05, 0],
                [0, 0, 0, 0, 0, 5.25],
            ]
        )
        * 1e9
    )  # (Pa)

    C_edit = (
        np.array(
            [
                [95.613, 13.712, 13.712, 0, 9.6658e-5, 0],
                [13.712, 20.114, 12.266, 0, 1.1571e-4, 0],
                [13.712, 12.266, 20.113, 0, 9.7739e-5, 0],
                [0.0, 0.0, 0.0, 7.8198, 0.0, 4.8020e-5],
                [9.6658e-5, 1.1571e-4, 9.7739e-5, 0, 5.2973, 0],
                [0.0, 0.0, 0.0, 4.8020e-5, 0, 7.8198],
            ]
        )
        * 1e9
    )  # (Pa)

    C_list = np.array([C, C, C])
    C_edit_list = np.array([C_edit, C_edit, C_edit])

    r_i = 328.0e-3  # (m)
    th = np.array([10.0e-3, 4.0e-3, 3.0e-3])  # (m)
    r_interface_list = np.concatenate(([r_i], r_i + np.cumsum(th)))  # (m)

    theta_deg_list = np.array([90, 12, 67])
    theta_rad_list = np.deg2rad(theta_deg_list)

    failure = {
        "Xt": 2860.0e6,  # 纵向拉伸强度 pa
        "Xc": 1450.0e6,  # 纵向压缩强度 pa
        "Yt": 81.0e6,  # 横向拉伸强度 pa
        "Yc": 170.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    return C_list, C_edit_list, r_interface_list, theta_rad_list, failure, "config1"


def setup_config2():
    """
    作用:
        返回工况2几何和材料参数
        层数1层, 层厚30mm, 铺层角12deg
        半径191mm
    输出:
        C:  刚度矩阵 (6, 6)
        C_edit: 含体积分数刚度矩阵 (6, 6)
        r_interface_list:   界面半径位置 (2,) (m)
        theta:  铺层角度 (1,) (rad)
        failure: 材料失效参数 dict
        name: 'config2'
    """

    C = (
        np.array(
            [
                [159.9998, 3.3195, 3.8702, 0, 0, 0],
                [3.3195, 10.4519, 2.5826, 0, 0, 0],
                [3.8702, 2.5826, 10.4767, 0, 0, 0],
                [0, 0, 0, 5.25, 0, 0],
                [0, 0, 0, 0, 3.05, 0],
                [0, 0, 0, 0, 0, 5.25],
            ]
        )
        * 1e9
    )  # (Pa)

    C_edit = (
        np.array(
            [
                [95.613, 13.712, 13.712, 0, 9.6658e-5, 0],
                [13.712, 20.114, 12.266, 0, 1.1571e-4, 0],
                [13.712, 12.266, 20.113, 0, 9.7739e-5, 0],
                [0.0, 0.0, 0.0, 7.8198, 0.0, 4.8020e-5],
                [9.6658e-5, 1.1571e-4, 9.7739e-5, 0, 5.2973, 0],
                [0.0, 0.0, 0.0, 4.8020e-5, 0, 7.8198],
            ]
        )
        * 1e9
    )  # (Pa)

    r_interface_list = np.array([191.0e-3, 191.0e-3 + 30.0e-3])  # (m)
    theta = np.deg2rad(12)

    failure = {
        "Xt": 2860.0e6,  # 纵向拉伸强度 pa
        "Xc": 1450.0e6,  # 纵向压缩强度 pa
        "Yt": 81.0e6,  # 横向拉伸强度 pa
        "Yc": 170.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    return C, C_edit, r_interface_list, theta, failure, "config2"


def setup_config3():
    """
    作用:
        返回工况3几何和材料参数
        层数1层, 层厚30mm, 铺层角90deg
        半径191mm
    输出:
        C:  刚度矩阵 (6, 6)
        C_edit: 含体积分数刚度矩阵 (6, 6)
        r_interface_list:   界面半径位置 (2,) (m)
        theta:  铺层角度 (1,) (rad)
        failure: 材料失效参数 dict
        name: 'config3'
    """
    C = (
        np.array(
            [
                [159.9998, 3.3195, 3.8702, 0, 0, 0],
                [3.3195, 10.4519, 2.5826, 0, 0, 0],
                [3.8702, 2.5826, 10.4767, 0, 0, 0],
                [0, 0, 0, 5.25, 0, 0],
                [0, 0, 0, 0, 3.05, 0],
                [0, 0, 0, 0, 0, 5.25],
            ]
        )
        * 1e9
    )  # (Pa)

    C_edit = (
        np.array(
            [
                [95.613, 13.712, 13.712, 0, 9.6658e-5, 0],
                [13.712, 20.114, 12.266, 0, 1.1571e-4, 0],
                [13.712, 12.266, 20.113, 0, 9.7739e-5, 0],
                [0.0, 0.0, 0.0, 7.8198, 0.0, 4.8020e-5],
                [9.6658e-5, 1.1571e-4, 9.7739e-5, 0, 5.2973, 0],
                [0.0, 0.0, 0.0, 4.8020e-5, 0, 7.8198],
            ]
        )
        * 1e9
    )  # (Pa)

    r_interface_list = np.array([191.0e-3, 191.0e-3 + 30.0e-3])  # (m)
    theta = np.deg2rad(90)

    failure = {
        "Xt": 2860.0e6,  # 纵向拉伸强度 pa
        "Xc": 1450.0e6,  # 纵向压缩强度 pa
        "Yt": 81.0e6,  # 横向拉伸强度 pa
        "Yc": 170.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    return C, C_edit, r_interface_list, theta, failure, "config3"


def setup_config4():
    """
    作用:
        返回工况4几何和材料参数
        层数8层:
            1. 层厚10mm, 铺层角90deg
            2. 层厚4mm, 铺层角12deg
            3. 层厚3mm, 铺层角67deg
        内半径191mm
    输出:
        C_list:  刚度矩阵 (6, 6)
        r_interface_list:   界面半径位置 (2,) (m)
        theta:  铺层角度 (1,) (rad)
        failure: 材料失效参数 dict
        name: 'config4'
    """

    from src.material_utils import build_stiffness

    r_i = 179.5e-3 / 2  # (m)
    r_o = 215.5e-3 / 2  # (m)
    num_layers = 8
    r_interface_list = np.linspace(r_i, r_o, num_layers + 1)

    theta_deg_list = np.array([-45.0, 45.0, 45.0, -45.0, -45.0, 45.0, 45.0, -45.0])
    theta_rad_list = np.deg2rad(theta_deg_list)

    E_list = np.array([138.9e9, 9.86e9, 9.86e9])
    nu_list = np.array([0.3, 0.3, 0.3])
    G23 = E_list[1] / (2 * (1 + nu_list[2]))
    G_list = np.array([5.24e9, 5.24e9, G23])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.full(len(theta_rad_list), C)

    failure = {
        "Xt": 2326.0e6,  # 纵向拉伸强度 pa
        "Xc": -1236.0e6,  # 纵向压缩强度 pa
        "Yt": 51.0e6,  # 横向拉伸强度 pa
        "Yc": -209.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 87.9e6,  # 面内剪切强度 pa
        "S13": 87.9e6,  # 横向剪切强度 pa  不能确定
        "S23": 99.2e6,  # 横向剪切强度 pa
    }

    return C_list, r_interface_list, theta_rad_list, failure, "config4"
