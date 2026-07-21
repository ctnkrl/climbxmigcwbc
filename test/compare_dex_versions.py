#!/usr/bin/env python3
"""
测试脚本：比较 func_sp_trans_dex.py 和 func_sp_trans_dex_optimized.py 的运行结果一致性
"""
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.func_sp_trans_dex import FuncSPTrans as DexOriginal
from common.func_sp_trans_dex_opt import OptimizedFuncSPTrans as DexOptimized


def test_parallel_to_serial():
    """
    测试并联到串联转换的一致性
    """
    print("=" * 80)
    print("测试并联到串联转换的一致性")
    print("=" * 80)
    
    # 初始化两个版本
    dex_orig = DexOriginal()
    dex_opt = DexOptimized()
    
    # 准备测试数据
    q_a_p = np.array([0.1, 0.2, -0.1, -0.2])
    qdot_a_p = np.array([0.01, 0.02, -0.01, -0.02])
    tor_a_p = np.array([1.0, 2.0, -1.0, -2.0])
    
    print(f"输入数据:")
    print(f"  q_a_p: {q_a_p}")
    print(f"  qdot_a_p: {qdot_a_p}")
    print(f"  tor_a_p: {tor_a_p}")
    print()
    
    # 原始版本计算
    dex_orig.setPEst(q_a_p, qdot_a_p, tor_a_p)
    dex_orig.calcFK()
    dex_orig.calcIK()
    q_a_s_orig, qdot_a_s_orig, tor_a_s_orig = dex_orig.getSState()
    
    # 优化版本计算
    dex_opt.set_p_est(q_a_p, qdot_a_p, tor_a_p)
    dex_opt.calc_fk()
    dex_opt.calc_ik()
    q_a_s_opt = np.zeros(4)
    qdot_a_s_opt = np.zeros(4)
    tor_a_s_opt = np.zeros(4)
    dex_opt.get_s_state(q_a_s_opt, qdot_a_s_opt, tor_a_s_opt)
    
    # 输出结果
    print(f"原始版本输出数据:")
    print(f"  q_a_s: {q_a_s_orig}")
    print(f"  qdot_a_s: {qdot_a_s_orig}")
    print(f"  tor_a_s: {tor_a_s_orig}")
    print()
    
    print(f"优化版本输出数据:")
    print(f"  q_a_s: {q_a_s_opt}")
    print(f"  qdot_a_s: {qdot_a_s_opt}")
    print(f"  tor_a_s: {tor_a_s_opt}")
    print()
    
    # 计算差异
    q_diff = np.abs(q_a_s_orig - q_a_s_opt)
    qdot_diff = np.abs(qdot_a_s_orig - qdot_a_s_opt)
    tor_diff = np.abs(tor_a_s_orig - tor_a_s_opt)
    
    print(f"结果差异:")
    print(f"  q_a_s 差异: {q_diff}")
    print(f"  qdot_a_s 差异: {qdot_diff}")
    print(f"  tor_a_s 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()


def test_serial_to_parallel():
    """
    测试串联到并联转换的一致性
    """
    print("=" * 80)
    print("测试串联到并联转换的一致性")
    print("=" * 80)
    
    # 初始化两个版本
    dex_orig = DexOriginal()
    dex_opt = DexOptimized()
    
    # 准备测试数据
    q_d_s = np.array([0.1 + 0.09765, 0.2, -0.1 + 0.09765, -0.2])  # 注意偏移
    qdot_d_s = np.array([0.01, 0.02, -0.01, -0.02])
    tor_d_s = np.array([1.0, 2.0, -1.0, -2.0])
    
    print(f"输入数据:")
    print(f"  q_d_s: {q_d_s}")
    print(f"  qdot_d_s: {qdot_d_s}")
    print(f"  tor_d_s: {tor_d_s}")
    print()
    
    # 原始版本计算
    dex_orig.setSDes(q_d_s, qdot_d_s, tor_d_s)
    dex_orig.calcJointPosRef()
    dex_orig.calcJointTorDes()
    q_d_p_orig, qdot_d_p_orig, tor_d_p_orig = dex_orig.getPDes()
    
    # 优化版本计算
    dex_opt.set_s_des(q_d_s, qdot_d_s, tor_d_s)
    dex_opt.calc_joint_pos_ref()
    dex_opt.calc_joint_tor_des()
    q_d_p_opt = np.zeros(4)
    qdot_d_p_opt = np.zeros(4)
    tor_d_p_opt = np.zeros(4)
    dex_opt.get_p_des(q_d_p_opt, qdot_d_p_opt, tor_d_p_opt)
    
    # 输出结果
    print(f"原始版本输出数据:")
    print(f"  q_d_p: {q_d_p_orig}")
    print(f"  qdot_d_p: {qdot_d_p_orig}")
    print(f"  tor_d_p: {tor_d_p_orig}")
    print()
    
    print(f"优化版本输出数据:")
    print(f"  q_d_p: {q_d_p_opt}")
    print(f"  qdot_d_p: {qdot_d_p_opt}")
    print(f"  tor_d_p: {tor_d_p_opt}")
    print()
    
    # 计算差异
    q_diff = np.abs(q_d_p_orig - q_d_p_opt)
    qdot_diff = np.abs(qdot_d_p_orig - qdot_d_p_opt)
    tor_diff = np.abs(tor_d_p_orig - tor_d_p_opt)
    
    print(f"结果差异:")
    print(f"  q_d_p 差异: {q_diff}")
    print(f"  qdot_d_p 差异: {qdot_diff}")
    print(f"  tor_d_p 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()


def test_neural_network():
    """
    单独测试神经网络计算的一致性
    """
    print("=" * 80)
    print("测试神经网络计算的一致性")
    print("=" * 80)
    
    # 初始化两个版本
    dex_orig = DexOriginal()
    dex_opt = DexOptimized()
    
    # 设置相同的输入
    alpha1 = 0.1
    alpha2 = 0.2
    dex_orig.alpha1_l_est = alpha1
    dex_orig.alpha2_l_est = alpha2
    dex_opt.alpha1_l_est = alpha1
    dex_opt.alpha2_l_est = alpha2
    
    # 原始版本计算
    dex_orig.calcAnkleEstLeft()
    roll_orig = dex_orig.roll_l_est
    pitch_orig = dex_orig.pitch_l_est
    
    # 优化版本计算
    dex_opt._calc_ankle_est_left()
    roll_opt = dex_opt.roll_l_est
    pitch_opt = dex_opt.pitch_l_est
    
    print(f"原始版本输出:")
    print(f"  roll_l_est: {roll_orig}")
    print(f"  pitch_l_est: {pitch_orig}")
    print()
    
    print(f"优化版本输出:")
    print(f"  roll_l_est: {roll_opt}")
    print(f"  pitch_l_est: {pitch_opt}")
    print()
    
    # 计算差异
    roll_diff = abs(roll_orig - roll_opt)
    pitch_diff = abs(pitch_orig - pitch_opt)
    
    print(f"结果差异:")
    print(f"  roll_l_est 差异: {roll_diff}")
    print(f"  pitch_l_est 差异: {pitch_diff}")
    print(f"  最大差异: {max(roll_diff, pitch_diff)}")
    print()


def main():
    """
    主测试函数
    """
    print("DEX版本一致性测试")
    print("比较 func_sp_trans_dex.py 和 func_sp_trans_dex_optimized.py")
    print("开始测试...\n")
    
    # 测试神经网络计算
    test_neural_network()
    
    # 测试并联到串联转换
    test_parallel_to_serial()
    
    # 测试串联到并联转换
    test_serial_to_parallel()
    
    print("=" * 80)
    print("所有一致性测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()