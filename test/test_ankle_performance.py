#!/usr/bin/env python3
"""
测试脚本：比较原始 FuncSPTrans 和优化版 OptimizedFuncSPTrans 的性能和结果
"""
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.func_sp_trans import FuncSPTrans
from common.func_sp_trans_new import OptimizedFuncSPTrans

def compare_parallel_to_serial():
    """
    比较并联到串联转换的性能和结果
    """
    print("=" * 80)
    print("比较 ankle_parallel_to_serial 转换")
    print("=" * 80)
    
    # 初始化转换器
    original = FuncSPTrans()
    optimized = OptimizedFuncSPTrans()
    
    # 准备测试数据 (并联关节数据)
    q_a_p = np.array([0.1, 0.2, -0.1, -0.2])  # 左右脚踝的alpha角度
    qdot_a_p = np.array([0.01, 0.02, -0.01, -0.02])
    tor_a_p = np.array([1.0, 2.0, -1.0, -2.0])
    
    print(f"输入数据:")
    print(f"  q_a_p: {q_a_p}")
    print(f"  qdot_a_p: {qdot_a_p}")
    print(f"  tor_a_p: {tor_a_p}")
    print()
    
    # 预热运行
    for _ in range(10):
        # 原始版本
        original.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        original.calc_fk()
        original.calcIK()
        
        q_a_s_orig = np.zeros(4)
        qdot_a_s_orig = np.zeros(4)
        tor_a_s_orig = np.zeros(4)
        original.get_s_state(q_a_s_orig, qdot_a_s_orig, tor_a_s_orig)
        
        # 优化版本
        optimized.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        optimized.calc_fk()
        optimized.calc_ik()
        
        q_a_s_opt = np.zeros(4)
        qdot_a_s_opt = np.zeros(4)
        tor_a_s_opt = np.zeros(4)
        optimized.get_s_state(q_a_s_opt, qdot_a_s_opt, tor_a_s_opt)
    
    # 性能测试
    num_runs = 1000
    orig_times = []
    opt_times = []
    
    # 原始版本性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        original.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        original.calc_fk()
        original.calcIK()
        
        q_a_s_orig = np.zeros(4)
        qdot_a_s_orig = np.zeros(4)
        tor_a_s_orig = np.zeros(4)
        original.get_s_state(q_a_s_orig, qdot_a_s_orig, tor_a_s_orig)
        
        end_time = time.perf_counter()
        orig_times.append(end_time - start_time)
    
    # 优化版本性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        optimized.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        optimized.calc_fk()
        optimized.calc_ik()
        
        q_a_s_opt = np.zeros(4)
        qdot_a_s_opt = np.zeros(4)
        tor_a_s_opt = np.zeros(4)
        optimized.get_s_state(q_a_s_opt, qdot_a_s_opt, tor_a_s_opt)
        
        end_time = time.perf_counter()
        opt_times.append(end_time - start_time)
    
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
    
    # 计算结果差异
    q_diff = np.abs(q_a_s_orig - q_a_s_opt)
    qdot_diff = np.abs(qdot_a_s_orig - qdot_a_s_opt)
    tor_diff = np.abs(tor_a_s_orig - tor_a_s_opt)
    
    print(f"结果差异:")
    print(f"  q_a_s 差异: {q_diff}")
    print(f"  qdot_a_s 差异: {qdot_diff}")
    print(f"  tor_a_s 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()
    
    # 性能统计
    orig_avg_time = np.mean(orig_times)
    opt_avg_time = np.mean(opt_times)
    speedup = orig_avg_time / opt_avg_time
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  原始版本平均时间: {orig_avg_time*1000:.4f} ms")
    print(f"  优化版本平均时间: {opt_avg_time*1000:.4f} ms")
    print(f"  性能提升: {speedup:.2f}x")
    print(f"  原始版本平均频率: {1.0/orig_avg_time:.2f} Hz")
    print(f"  优化版本平均频率: {1.0/opt_avg_time:.2f} Hz")
    print()

def compare_serial_to_parallel():
    """
    比较串联到并联转换的性能和结果
    """
    print("=" * 80)
    print("比较 ankle_serial_to_parallel 转换")
    print("=" * 80)
    
    # 初始化转换器
    original = FuncSPTrans()
    optimized = OptimizedFuncSPTrans()
    
    # 准备测试数据 (串联关节数据)
    q_d_s = np.array([0.1, 0.2, -0.1, -0.2])  # 左右脚踝的pitch和roll
    qdot_d_s = np.array([0.01, 0.02, -0.01, -0.02])
    tor_d_s = np.array([1.0, 2.0, -1.0, -2.0])
    
    print(f"输入数据:")
    print(f"  q_d_s: {q_d_s}")
    print(f"  qdot_d_s: {qdot_d_s}")
    print(f"  tor_d_s: {tor_d_s}")
    print()
    
    # 预热运行
    for _ in range(10):
        # 原始版本
        original.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        original.calc_joint_pos_ref()
        original.calc_joint_tor_des()
        
        q_d_p_orig = np.zeros(4)
        qdot_d_p_orig = np.zeros(4)
        tor_d_p_orig = np.zeros(4)
        original.get_p_des(q_d_p_orig, qdot_d_p_orig, tor_d_p_orig)
        
        # 优化版本
        optimized.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        optimized.calc_joint_pos_ref()
        optimized.calc_joint_tor_des()
        
        q_d_p_opt = np.zeros(4)
        qdot_d_p_opt = np.zeros(4)
        tor_d_p_opt = np.zeros(4)
        optimized.get_p_des(q_d_p_opt, qdot_d_p_opt, tor_d_p_opt)
    
    # 性能测试
    num_runs = 1000
    orig_times = []
    opt_times = []
    
    # 原始版本性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        original.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        original.calc_joint_pos_ref()
        original.calc_joint_tor_des()
        
        q_d_p_orig = np.zeros(4)
        qdot_d_p_orig = np.zeros(4)
        tor_d_p_orig = np.zeros(4)
        original.get_p_des(q_d_p_orig, qdot_d_p_orig, tor_d_p_orig)
        
        end_time = time.perf_counter()
        orig_times.append(end_time - start_time)
    
    # 优化版本性能测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        optimized.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        optimized.calc_joint_pos_ref()
        optimized.calc_joint_tor_des()
        
        q_d_p_opt = np.zeros(4)
        qdot_d_p_opt = np.zeros(4)
        tor_d_p_opt = np.zeros(4)
        optimized.get_p_des(q_d_p_opt, qdot_d_p_opt, tor_d_p_opt)
        
        end_time = time.perf_counter()
        opt_times.append(end_time - start_time)
    
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
    
    # 计算结果差异
    q_diff = np.abs(q_d_p_orig - q_d_p_opt)
    qdot_diff = np.abs(qdot_d_p_orig - qdot_d_p_opt)
    tor_diff = np.abs(tor_d_p_orig - tor_d_p_opt)
    
    print(f"结果差异:")
    print(f"  q_d_p 差异: {q_diff}")
    print(f"  qdot_d_p 差异: {qdot_diff}")
    print(f"  tor_d_p 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()
    
    # 性能统计
    orig_avg_time = np.mean(orig_times)
    opt_avg_time = np.mean(opt_times)
    speedup = orig_avg_time / opt_avg_time
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  原始版本平均时间: {orig_avg_time*1000:.4f} ms")
    print(f"  优化版本平均时间: {opt_avg_time*1000:.4f} ms")
    print(f"  性能提升: {speedup:.2f}x")
    print(f"  原始版本平均频率: {1.0/orig_avg_time:.2f} Hz")
    print(f"  优化版本平均频率: {1.0/opt_avg_time:.2f} Hz")
    print()

def compare_round_trip_conversion():
    """
    比较往返转换的精度
    """
    print("=" * 80)
    print("比较往返转换精度")
    print("=" * 80)
    
    original1 = FuncSPTrans()
    original2 = FuncSPTrans()
    optimized1 = OptimizedFuncSPTrans()
    optimized2 = OptimizedFuncSPTrans()
    
    # 原始并联数据
    original_q_p = np.array([0.1, 0.2, -0.1, -0.2])
    original_qdot_p = np.array([0.01, 0.02, -0.01, -0.02])
    original_tor_p = np.array([1.0, 2.0, -1.0, -2.0])
    
    print("原始并联数据:")
    print(f"  q_p: {original_q_p}")
    print(f"  qdot_p: {original_qdot_p}")
    print(f"  tor_p: {original_tor_p}")
    print()
    
    # 原始版本往返转换
    original1.set_p_est(original_q_p, original_qdot_p, original_tor_p)
    original1.calc_fk()
    original1.calcIK()
    
    q_s_orig = np.zeros(4)
    qdot_s_orig = np.zeros(4)
    tor_s_orig = np.zeros(4)
    original1.get_s_state(q_s_orig, qdot_s_orig, tor_s_orig)
    
    original2.set_s_des(q_s_orig, qdot_s_orig, tor_s_orig)
    original2.calc_joint_pos_ref()
    original2.calc_joint_tor_des()
    
    recovered_q_p_orig = np.zeros(4)
    recovered_qdot_p_orig = np.zeros(4)
    recovered_tor_p_orig = np.zeros(4)
    original2.get_p_des(recovered_q_p_orig, recovered_qdot_p_orig, recovered_tor_p_orig)
    
    print("原始版本往返转换结果:")
    print(f"  恢复的q_p: {recovered_q_p_orig}")
    print(f"  恢复的qdot_p: {recovered_qdot_p_orig}")
    print(f"  恢复的tor_p: {recovered_tor_p_orig}")
    print()
    
    # 优化版本往返转换
    optimized1.set_p_est(original_q_p, original_qdot_p, original_tor_p)
    optimized1.calc_fk()
    optimized1.calc_ik()
    
    q_s_opt = np.zeros(4)
    qdot_s_opt = np.zeros(4)
    tor_s_opt = np.zeros(4)
    optimized1.get_s_state(q_s_opt, qdot_s_opt, tor_s_opt)
    
    optimized2.set_s_des(q_s_opt, qdot_s_opt, tor_s_opt)
    optimized2.calc_joint_pos_ref()
    optimized2.calc_joint_tor_des()
    
    recovered_q_p_opt = np.zeros(4)
    recovered_qdot_p_opt = np.zeros(4)
    recovered_tor_p_opt = np.zeros(4)
    optimized2.get_p_des(recovered_q_p_opt, recovered_qdot_p_opt, recovered_tor_p_opt)
    
    print("优化版本往返转换结果:")
    print(f"  恢复的q_p: {recovered_q_p_opt}")
    print(f"  恢复的qdot_p: {recovered_qdot_p_opt}")
    print(f"  恢复的tor_p: {recovered_tor_p_opt}")
    print()
    
    # 计算版本间差异
    q_diff = np.abs(recovered_q_p_orig - recovered_q_p_opt)
    qdot_diff = np.abs(recovered_qdot_p_orig - recovered_qdot_p_opt)
    tor_diff = np.abs(recovered_tor_p_orig - recovered_tor_p_opt)
    
    print("版本间往返转换差异:")
    print(f"  q_p 差异: {q_diff}")
    print(f"  qdot_p 差异: {qdot_diff}")
    print(f"  tor_p 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()

def main():
    """
    主测试函数
    """
    print("踝关节串并联转换对比测试")
    print("比较原始 FuncSPTrans 和优化版 OptimizedFuncSPTrans")
    print("开始测试...\n")
    
    # 测试并联到串联转换
    compare_parallel_to_serial()
    
    # 测试串联到并联转换
    compare_serial_to_parallel()
    
    # 测试往返转换精度
    compare_round_trip_conversion()
    
    print("=" * 80)
    print("所有对比测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()