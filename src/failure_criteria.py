import numpy as np


def tsai_wu(stress_tensors: np.ndarray, criteria: dict) -> np.ndarray:
    """
    三维 tsai_wu 失效准则

    表达式:
      F_i * s_i + F_{ij} * s_i * s_j = 1
      Zt, Zc = Yt, Yc

    参数:
      应力分量 $sigma$ = [s1, s2, s3, t23, t13, t12]
    输出:
      失效判断
    """

    Xt = criteria["Xt"]  # 拉伸强度 (正)
    Xc = criteria["Xc"]  # 压缩强度 (正)
    Yt = criteria["Yt"]  # 拉伸强度 (正)
    Yc = criteria["Yc"]  # 压缩强度 (正)
    Zt, Zc = Yt, Yc  # 假设横向同性

    S12 = criteria["S12"]  # 面内剪切
    S13 = criteria["S13"]  # 1-3 剪切
    S23 = criteria["S23"]  # 2-3 剪切

    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F3 = 1.0 / Zt - 1.0 / Zc

    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F33 = 1.0 / (Zt * Zc)

    F44 = 1.0 / (S23**2)
    F55 = 1.0 / (S13**2)
    F66 = 1.0 / (S12**2)

    F12 = -0.5 * np.sqrt(F11 * F22)
    F23 = -0.5 * np.sqrt(F22 * F33)
    F13 = -0.5 * np.sqrt(F33 * F11)

    s_1 = stress_tensors[..., 0]
    s_2 = stress_tensors[..., 1]
    s_3 = stress_tensors[..., 2]
    tau_23 = stress_tensors[..., 3]
    tau_13 = stress_tensors[..., 4]
    tau_12 = stress_tensors[..., 5]

    L = (
        F1 * s_1
        + F2 * s_2
        + F3 * s_3
        + F11 * s_1**2
        + F22 * s_2**2
        + F33 * s_3**2
        + F44 * tau_23**2
        + F55 * tau_13**2
        + F66 * tau_12**2
        + 2 * (F12 * s_1 * s_2 + F23 * s_2 * s_3 + F13 * s_3 * s_1)
    )

    return L >= 1.0


def hashin(stress_tensors: np.ndarray, criteria: dict) -> np.ndarray:
    """
    三维 Hashin 应力失效准则 (纤维/基体分离模式)

    表达式:
      ...
      Zt, Zc = Yt, Yc

    参数:
      应力分量 $sigma$ = [s1, s2, s3, t23, t13, t12]
    输出:
      失效判断
    """

    Xt = criteria["Xt"]  # 拉伸强度 (正)
    Xc = criteria["Xc"]  # 压缩强度 (正)
    Yt = criteria["Yt"]  # 拉伸强度 (正)
    Yc = criteria["Yc"]  # 压缩强度 (正)
    Zt, Zc = Yt, Yc  # 假设横向同性

    S12 = criteria["S12"]  # 面内剪切
    S13 = criteria["S13"]  # 1-3 剪切
    S23 = criteria["S23"]  # 2-3 剪切

    s_1 = stress_tensors[..., 0]
    s_2 = stress_tensors[..., 1]
    s_3 = stress_tensors[..., 2]
    t_23 = stress_tensors[..., 3]
    t_13 = stress_tensors[..., 4]
    t_12 = stress_tensors[..., 5]

    # 纤维拉伸失效 s_11 >= 0
    L1 = (s_1 / Xt) ** 2 + (t_12 / S12) ** 2 + (t_13 / S13) ** 2

    # 纤维压缩失效 s_11 < 0
    L2 = (s_1 / Xc) ** 2

    # 基体拉伸失效 s_22 + s_33 >= 0
    L3 = (
        (s_2 + s_3) ** 2 / Yt**2
        + (t_23**2 - s_2 * s_3) / S23**2
        + (t_12 / S12) ** 2
        + (t_13 / S13) ** 2
    )

    # 基体压缩失效 s_22 + s_33 < 0
    L4 = (
        ((Yc / (2 * S23)) ** 2 - 1) * (s_2 + s_3) / Yc
        + (s_2 + s_3) ** 2 / (4 * S23**2)
        + (t_23**2 - s_2 * s_3) / S23**2
        + (t_12 / S12) ** 2
        + (t_13 / S13) ** 2
    )

    L = np.stack([L1, L2, L3, L4], axis=-1)

    return L >= 1.0
