"""
材料属性工具文件
    build_stiffness() 计算刚度矩阵
    stiffness_to_properties() 刚度矩阵计算材料属性函数
    bond_transform() 刚度矩阵 Bond 变换
    condense_stiffness() 静力凝聚函数 
    cyl2mat_stress() 柱坐标系下应力转换到材料坐标系
"""

import numpy as np

def build_stiffness(E_list, nu_list, G_list):
    E1, E2, E3 = E_list[0], E_list[1], E_list[2]
    nu12, nu13 ,nu23 = nu_list[0], nu_list[1], nu_list[2]
    G12, G13, G23 = G_list[0], G_list[1], G_list[2]

    S = np.array(
        [
            [1.0 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
            [-nu12 / E1, 1.0 / E2, -nu23 / E2, 0, 0, 0],
            [-nu13 / E1, -nu23 / E2, 1.0 / E3, 0, 0, 0],
            [0, 0, 0, 1.0 / G23, 0, 0],
            [0, 0, 0, 0, 1.0 / G13, 0],
            [0, 0, 0, 0, 0, 1.0 / G12],
        ]
    )

    C = np.linalg.inv(S)
    return C


def stiffness_to_properties(C):
    """
    从正交各向异性刚度矩阵(6x6)恢复工程弹性常数。
    
    输入:
        C : np.ndarray, shape (6,6), 正交各向异性刚度矩阵
    
    输出:
        E_list : 
            E1, E2, E3
        nu_list: 
            nu12, nu13, nu23
        G_list:
            G12, G13, G23
    """
    C = np.asarray(C, dtype=float)
    # 求柔度矩阵 S = C^{-1}
    S = np.linalg.inv(C)
    
    # 弹性模量
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    E3 = 1.0 / S[2, 2]
    
    # 泊松比（注意定义：nu12 = -S12 * E1）
    nu12 = -S[0, 1] * E1
    nu13 = -S[0, 2] * E1
    nu23 = -S[1, 2] * E2
    
    # 剪切模量
    G23 = 1.0 / S[3, 3]
    G13 = 1.0 / S[4, 4]
    G12 = 1.0 / S[5, 5]

    E_list = np.array([E1, E2, E3])
    nu_list = np.array([nu12, nu13, nu23])
    G_list = np.array([G12, G13, G23])
    
    return E_list, nu_list, G_list


def bond_transform(G, phi):
    """
    计算 bond 变换矩阵
        G: 输入矩阵 (..., L1, 6, 6)
        phi: 旋转角度 (弧度) (L1,)
    返回:
        G1 = M * G * M^T (..., L1, 6, 6)
    """

    phi = np.asarray(phi)
    G = np.asarray(G)

    # 检查批量维度是否可广播
    try:
        _ = np.broadcast_shapes(phi.shape, G.shape[:-2])
    except ValueError:
        raise ValueError(
            f"phi 形状 {phi.shape} 与 G 的前导维度 {G.shape[:-2]} 无法广播。"
        )

    c = np.cos(phi)
    s = np.sin(phi)
    c2 = np.cos(2*phi)   # 等价于 c^2 - s^2
    cs = c * s

    # 构建变换矩阵 M (6x6)
    z = np.zeros_like(c)
    o = np.ones_like(c)

    row0 = np.stack([c**2,   s**2,   z,   z,   z,   2*cs], axis=-1)
    row1 = np.stack([s**2,   c**2,   z,   z,   z,  -2*cs], axis=-1)
    row2 = np.stack([z,      z,      o,   z,   z,   z],    axis=-1)
    row3 = np.stack([z,      z,      z,   c,  -s,   z],    axis=-1)
    row4 = np.stack([z,      z,      z,   s,   c,   z],    axis=-1)
    row5 = np.stack([-cs,    cs,     z,   z,   z,   c2],   axis=-1)

    M = np.stack([row0, row1, row2, row3, row4, row5], axis=-2)

    # 计算 G1 = M * G * M^T
    G1 = M @ G @ M.swapaxes(-1, -2)
    return G1


def condense_stiffness(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    静力凝聚
    输入:
        C 刚度矩阵 (6, 6)
    输出:
        Q 静力凝聚矩阵 (3, 3) Constant
        C 刚度矩阵 Constant
    """
    Q_full = C - np.outer(C[:, 4], C[:, 4]) / C[4, 4]
    Q = Q_full[:3, :3]
    Q = np.ascontiguousarray(Q, dtype=np.float64)  # 强制转换为 C 连续
    Q.setflags(write=False)  # 强制设置为只读
    return Q, C


def cyl2mat_stress(sigma_cyl, theta_rad):
    """
    将圆柱坐标系下的应力张量转换到材料主方向（纤维、横向、径向）。
    输入:
        sigma_cyl : np.ndarray, shape (N, 6) or (6,)
            圆柱坐标系下的应力分量，顺序为：
            [sigma_r, sigma_theta, sigma_z, tau_thetaz, tau_rz, tau_rtheta]
            其中 N 为半径上的点数。
        theta_rad : float
            铺层角（纤维与 z 轴的夹角），单位：弧度。
    输出:
        np.ndarray, shape (N, 6) or (6,)
            材料主方向下的应力分量，顺序为：
            [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
    """
    original_shape = sigma_cyl.shape
    # 确保最后一维是 6
    if original_shape[-1] != 6:
        raise ValueError("最后一维必须是 6")
    # 展平为 (-1, 6)
    sigma_flat = sigma_cyl.reshape(-1, 6)

    # 以下是原有逻辑，但改为对每一行操作
    # theta = np.radians(theta_deg)
    theta = theta_rad
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [0, s, c],
            [0, -c, s],  # 原为[0, c, -s], 建议改为 [0, -c, s]
            [1, 0, 0],
        ]
    )

    # 预分配结果数组
    N = sigma_flat.shape[0]
    sigma_loc_flat = np.zeros_like(sigma_flat)

    for i in range(N):
        sr, st, sz, ttz, trz, trt = sigma_flat[i, :]
        S_glob = np.array([[sr, trt, trz], [trt, st, ttz], [trz, ttz, sz]])
        S_loc = R @ S_glob @ R.T
        sigma_loc_flat[i, :] = [
            S_loc[0, 0],
            S_loc[1, 1],
            S_loc[2, 2],
            S_loc[1, 2],
            S_loc[0, 2],
            S_loc[0, 1],
        ]

    # 恢复原始形状
    sigma_loc = sigma_loc_flat.reshape(original_shape)
    return sigma_loc
