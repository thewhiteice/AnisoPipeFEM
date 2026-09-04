"""
单一工况应力求解文件
    solve_iso_pipe()            单层各向同性圆管
    solve_aniso_pipe()          单层正交各向异性圆管
    solve_aniso_layered_pipe()  多层正交各向异性圆管
    solve_lame_stress()         厚壁均匀圆管闭口状态
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI


def solve_iso_pipe(r_in, r_out, E, nu, p_in_list, p_out_list, nx=100):
    """
    各向同性圆管径向应力分布求解器

    求解径向一维广义平面应变问题的应力分量分布
    已与 lame 解析解结果成功对照

    输入:
        r_in:   内壁半径 m
        r_out:  外壁半径 m
        E:      各向同性杨氏模量 Pa
        nu:     各向同性泊松比
        p_in:   内壁压强 Pa
        p_out:  外壁压强 Pa
        nx:     径向划分单元数 int 默认100
    输出:
        r_vals: 径向插值坐标 (nx,)
        u_vals: 插值径向位移 (nx,)
        sigma_vec_vals: 柱坐标系下插值应力分量 (nx,3)
            - 索引0: sigma_r
            - 索引1: sigma_θ
            - 索引2: sigma_z
    """


def solve_aniso_pipe(r_interface_list, C_basic, theta_list, p_in_list, p_out_list, nx=100):
    """
    各向异性圆管径向应力分布求解器

    求解径向一维广义平面应变问题的应力分量分布
    不支持多铺层嵌套圆管, 不支持获得材料坐标系下应力分量

    输入:
        r_interface_list: 界面半径数组 (layers_num+1,)
        C_basic:    材料坐标系下原始刚度矩阵数组 (6, 6)
        theta_list: 铺层角度数组 rad (layers_num,)
        p_i_list:   内壁压强数组 Pa (...,)
        p_o_list:   外壁压强数组 Pa (...,)
        nx:         网格单元数及插值输出数 默认=200 int
    输出:
        r_vals: 径向插值坐标 (nx,)
        u_vals: 插值径向位移 (nx,)
        sigma_vec_vals: 柱坐标系下插值应力分量 (nx,6)
            - 索引0: sigma_r
            - 索引1: sigma_θ
            - 索引2: sigma_z
            - 索引3: sigma_θz
            - 索引4: sigma_rz
            - 索引5: sigma_rθ
    """


def solver_aniso_layered_pipe():
    pass


def solve_lame_stress(r, r_in, r_out, p_in, p_out):
    """
    本函数给出厚壁均匀圆管(闭口状态)在受内外压力下的lame应力分布解析解
    解遵循 A ± B/r**2 的简洁形式
    """

    p1 = p_in
    p2 = p_out
    r1 = r_in
    r2 = r_out

    A = (r1**2 * p1 - r2**2 * p2) / (r2**2 - r1**2)
    B = (r1**2 * r2**2 * (p1 - p2)) / (r2**2 - r1**2)

    s_r = A - B / (r**2)
    s_t = A + B / (r**2)
    s_z = A

    return s_r, s_t, s_z
