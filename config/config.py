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
        "Xt": 2180.0e6,  # 纵向拉伸强度 pa 2180.0e6
        "Xc": 1200.0e6,  # 纵向压缩强度 pa 1200.0e6
        "Yt": 60.0e6,  # 横向拉伸强度 pa 60.0e6
        "Yc": 140.0e6,  # 横向压缩强度 pa  不能确定 doubao 140.0e6
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    failure = [failure, failure, failure]
    # failure = [failure.copy() for _ in range(3)]

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
    theta = np.atleast_1d(theta)

    failure = {
        "Xt": 2180.0e6,  # 纵向拉伸强度 pa 2180.0e6
        "Xc": 1200.0e6,  # 纵向压缩强度 pa 1200.0e6
        "Yt": 60.0e6,  # 横向拉伸强度 pa 60.0e6
        "Yc": 140.0e6,  # 横向压缩强度 pa  不能确定 doubao 140.0e6
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    return (
        C[None, :, :],
        C_edit[None, :, :],
        r_interface_list,
        theta,
        [failure],
        "config2",
    )


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
    theta = np.atleast_1d(theta)

    failure = {
        "Xt": 2180.0e6,  # 纵向拉伸强度 pa 2180.0e6
        "Xc": 1200.0e6,  # 纵向压缩强度 pa 1200.0e6
        "Yt": 60.0e6,  # 横向拉伸强度 pa 60.0e6
        "Yc": 140.0e6,  # 横向压缩强度 pa  不能确定 doubao 140.0e6
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }

    return (
        C[None, :, :],
        C_edit[None, :, :],
        r_interface_list,
        theta,
        [failure],
        "config3",
    )


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
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 2326.0e6,  # 纵向拉伸强度 pa
        "Xc": 1236.0e6,  # 纵向压缩强度 pa
        "Yt": 51.0e6,  # 横向拉伸强度 pa
        "Yc": 209.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 87.9e6,  # 面内剪切强度 pa
        "S13": 87.9e6,  # 横向剪切强度 pa  不能确定
        "S23": 99.2e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config4"


def setup_config5():
    """2022 康凯 铝合金内胆碳纤维缠绕气瓶爆破与疲劳性能研究"""
    from src.material_utils import build_stiffness

    ply_scheme = (
        [11.0, -11.0] * 2
        + [90.0] * 6
        + [11.0, -11.0] * 2
        + [90.0] * 6
        + [11.0, -11.0]
        + [90.0] * 2
    )
    theta_deg_list = np.array(ply_scheme)
    theta_rad_list = np.deg2rad(theta_deg_list)

    num_layers = len(theta_deg_list)
    r_i = 163.0e-3 / 2  # (m)
    r_o = r_i + num_layers * 0.2e-3  # (m)
    r_interface_list = np.linspace(r_i, r_o, num_layers + 1)

    E_list = np.array([134.0e9, 7.42e9, 7.42e9])
    nu_list = np.array([0.28, 0.28, 0.3])
    G_list = np.array([3.71e9, 3.71e9, 4.79e9])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 1950.0e6,  # 纵向拉伸强度 pa
        "Xc": 1250.0e6,  # 纵向压缩强度 pa
        "Yt": 74.0e6,  # 横向拉伸强度 pa
        "Yc": 180.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 50.0e6,  # 面内剪切强度 pa
        "S13": 50.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 50.0e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config5"


def setup_config6():
    """2025 娄雪莹 IV型储氢瓶塑料内胆坍塌失稳模拟及缠绕层渐进损伤分析"""
    from src.material_utils import build_stiffness

    ply_scheme = (
        [55.0, 50.0, 45.0, 38.913, 35.0, 31.087, 30.0]
        + [28.043, 26.087, 25.0]
        + [90.0] * 3
        + [21.087]
        + [90.0] * 3
        + [15.217]
        + [90.0] * 3
        + [18.043]
        + [90.0] * 3
        + [20.0]
        + [90.0] * 3
        + [15.0]
        + [90.0] * 3
        + [24.130, 22.826]
        + [15.217] * 5
        + [17.174, 20.0]
        + [15.217] * 3
        + [90.0]
    )
    theta_deg_list = np.array(ply_scheme)
    theta_rad_list = np.deg2rad(theta_deg_list)

    num_layers = len(theta_deg_list)
    r_i = 380.0e-3 / 2  # (m)
    r_o = r_i + num_layers * 0.625e-3  # (m)
    r_interface_list = np.linspace(r_i, r_o, num_layers + 1)

    E_list = np.array([160.0e9, 7.42e9, 7.42e9])
    nu_list = np.array([0.28, 0.28, 0.3])
    G_list = np.array([3.71e9, 3.71e9, 4.79e9])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 2500.0e6,  # 纵向拉伸强度 pa
        "Xc": 1250.0e6,  # 纵向压缩强度 pa
        "Yt": 74.0e6,  # 横向拉伸强度 pa
        "Yc": 180.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 50.0e6,  # 面内剪切强度 pa
        "S13": 90.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 90.0e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config6"


def setup_config7():
    """2025 李瑞奇 基于纤维强度折减效应的IV型气瓶爆破失效分析方法研究"""
    from src.material_utils import build_stiffness

    ply_scheme = (
        [14.0, -14.0]
        + [90.0, -90.0]
        + [90.0, -90.0]
        + [25.0, -25.0]
        + [90.0, -90.0]
        + [35.0, -35.0]
        + [90.0, 90.0]
        + [90.0, -90.0]
    )
    theta_deg_list = np.array(ply_scheme)
    theta_rad_list = np.deg2rad(theta_deg_list)

    num_layers = len(theta_deg_list)
    r_i = 164.0e-3 / 2  # (m)
    r_o = r_i + num_layers * 0.25e-3  # (m)
    r_interface_list = np.linspace(r_i, r_o, num_layers + 1)

    E_list = np.array([154.0e9, 11.4e9, 11.4e9])
    nu_list = np.array([0.3, 0.3, 0.33])
    G_list = np.array([4.8e9, 4.8e9, 3.8e9])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 2500.0e6,  # 纵向拉伸强度 pa
        "Xc": 1200.0e6,  # 纵向压缩强度 pa
        "Yt": 70.0e6,  # 横向拉伸强度 pa
        "Yc": 180.0e6,  # 横向压缩强度 pa  不能确定 doubao
        "S12": 50.0e6,  # 面内剪切强度 pa
        "S13": 90.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 90.0e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config7"


def setup_config8():
    """本文IV气瓶"""
    from src.material_utils import build_stiffness

    ply_scheme = (
        [89.0] * 5
        + [12.0, -12.0] * 2
        + [89.0] * 19
        + [12.0, -12.0] * 5
        + [89.0] * 19
        + [12.0, -12.0] * 5
        + [89.0] * 6
        + [12.0, -12.0] * 17
    )
    theta_deg_list = np.array(ply_scheme)
    theta_rad_list = np.deg2rad(theta_deg_list)

    num_layers = len(theta_deg_list)
    r_i = 366.0e-3 / 2  # (m)
    r_o = r_i + num_layers * 0.3128e-3  # (m)
    r_interface_list = np.linspace(r_i, r_o, num_layers + 1)

    E_list = np.array([158.0e9, 9.78e9, 9.78e9])
    nu_list = np.array([0.241, 0.310, 0.241])
    G_list = np.array([5.25e9, 3.05e9, 5.25e9])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 2180.0e6,  # 纵向拉伸强度 pa 2180.0e6
        "Xc": 1200.0e6,  # 纵向压缩强度 pa 1200.0e6
        "Yt": 60.0e6,  # 横向拉伸强度 pa 60.0e6
        "Yc": 140.0e6,  # 横向压缩强度 pa  不能确定 doubao 140.0e6
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config8"


def setup_config9():
    """本文IV气瓶(简并铺层后)"""
    from src.material_utils import build_stiffness

    ply_scheme = [89.0] + [12.0] + [89.0] + [12.0] + [89.0] + [12.0] + [89.0] + [12.0]
    theta_deg_list = np.array(ply_scheme)
    theta_rad_list = np.deg2rad(theta_deg_list)

    num_layers = len(theta_deg_list)
    th_scheme = (
        [1.564]
        + [0.3128 * 4]
        + [5.9432]
        + [0.3128 * 10]
        + [5.9432]
        + [0.3128 * 10]
        + [1.8768]
        + [0.3128 * 34]
    )
    th = np.array(th_scheme) * 1.0e-3
    r_i = 366.0e-3 / 2  # (m)
    r_interface_list = np.concatenate(([r_i], r_i + np.cumsum(th)))  # (m)

    E_list = np.array([158.0e9, 9.78e9, 9.78e9])
    nu_list = np.array([0.241, 0.310, 0.241])
    G_list = np.array([5.25e9, 3.05e9, 5.25e9])

    C = build_stiffness(E_list, nu_list, G_list)
    C_list = np.array([C] * num_layers)

    failure = {
        "Xt": 2180.0e6,  # 纵向拉伸强度 pa 2180.0e6
        "Xc": 1200.0e6,  # 纵向压缩强度 pa 1200.0e6
        "Yt": 60.0e6,  # 横向拉伸强度 pa 60.0e6
        "Yc": 140.0e6,  # 横向压缩强度 pa  不能确定 doubao 140.0e6
        "S12": 136.0e6,  # 面内剪切强度 pa
        "S13": 136.0e6,  # 横向剪切强度 pa  不能确定
        "S23": 86.9e6,  # 横向剪切强度 pa
    }
    failure = [failure.copy() for _ in range(num_layers)]

    return C_list, C_list, r_interface_list, theta_rad_list, failure, "config9"


if __name__ == "__main__":
    C, _, r_i_list, theta, failure, name = setup_config8()
    print(f"Num of Layers = {len(theta)}")
    print(r_i_list)
