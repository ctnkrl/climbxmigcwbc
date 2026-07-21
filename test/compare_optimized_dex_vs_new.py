#!/usr/bin/env python3
"""
性能测试脚本：比较优化后的DEX版本和NEW版本的性能
"""
import numpy as np
import time
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.func_sp_trans_dex_opt import OptimizedFuncSPTrans as OptimizedFuncSPTransDex
from common.func_sp_trans_opt import OptimizedFuncSPTrans as OptimizedFuncSPTransNew


def test_parallel_to_serial_performance():
    """
    测试并联到串联转换的性能
    """
    print("=" * 80)
    print("测试并联到串联转换性能")
    print("=" * 80)
    
    # 初始化转换器
    dex_opt = OptimizedFuncSPTransDex()
    new_opt = OptimizedFuncSPTransNew()
    
    # 准备测试数据 (并联关节数据)
    # 使用与dex版本匹配的参数
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
        # DEX优化版本
        dex_opt.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        dex_opt.calc_fk()
        dex_opt.calcIK()
        
        # NEW优化版本
        new_opt.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        new_opt.calc_fk()
        new_opt.calcIK()
    
    # 性能测试
    num_runs = 10000
    dex_times = []
    new_times = []
    
    # DEX优化版本性能测试
    print("测试优化后的DEX版本性能...")
    start_time = time.perf_counter()
    
    for i in range(num_runs):
        inner_start = time.perf_counter()
        
        dex_opt.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        dex_opt.calc_fk()
        dex_opt.calcIK()
        
        q_a_s_dex = np.zeros(4)
        qdot_a_s_dex = np.zeros(4)
        tor_a_s_dex = np.zeros(4)
        dex_opt.get_s_state(q_a_s_dex, qdot_a_s_dex, tor_a_s_dex)
        
        inner_end = time.perf_counter()
        dex_times.append(inner_end - inner_start)
    
    dex_total_time = time.perf_counter() - start_time
    
    # NEW优化版本性能测试
    print("测试NEW优化版本性能...")
    start_time = time.perf_counter()
    
    for i in range(num_runs):
        inner_start = time.perf_counter()
        
        new_opt.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        new_opt.calc_fk()
        new_opt.calcIK()
        
        q_a_s_new = np.zeros(4)
        qdot_a_s_new = np.zeros(4)
        tor_a_s_new = np.zeros(4)
        new_opt.get_s_state(q_a_s_new, qdot_a_s_new, tor_a_s_new)
        
        inner_end = time.perf_counter()
        new_times.append(inner_end - inner_start)
    
    new_total_time = time.perf_counter() - start_time
    
    # 输出结果
    print(f"优化后的DEX版本输出数据:")
    print(f"  q_a_s: {q_a_s_dex}")
    print(f"  qdot_a_s: {qdot_a_s_dex}")
    print(f"  tor_a_s: {tor_a_s_dex}")
    print()
    
    print(f"NEW优化版本输出数据:")
    print(f"  q_a_s: {q_a_s_new}")
    print(f"  qdot_a_s: {qdot_a_s_new}")
    print(f"  tor_a_s: {tor_a_s_new}")
    print()
    
    # 计算结果差异
    q_diff = np.abs(q_a_s_dex - q_a_s_new)
    qdot_diff = np.abs(qdot_a_s_dex - qdot_a_s_new)
    tor_diff = np.abs(tor_a_s_dex - tor_a_s_new)
    
    print(f"结果差异:")
    print(f"  q_a_s 差异: {q_diff}")
    print(f"  qdot_a_s 差异: {qdot_diff}")
    print(f"  tor_a_s 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()
    
    # 性能统计
    dex_avg_time = np.mean(dex_times)
    new_avg_time = np.mean(new_times)
    speedup = new_avg_time / dex_avg_time if dex_avg_time > 0 else float('inf')
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  优化后的DEX版本总时间: {dex_total_time:.4f} 秒")
    print(f"  NEW优化版本总时间: {new_total_time:.4f} 秒")
    print(f"  优化后的DEX版本平均时间: {dex_avg_time*1000:.6f} ms")
    print(f"  NEW优化版本平均时间: {new_avg_time*1000:.6f} ms")
    print(f"  性能提升 (DEX vs NEW): {speedup:.2f}x")
    print(f"  优化后的DEX版本平均频率: {1.0/dex_avg_time:.2f} Hz")
    print(f"  NEW优化版本平均频率: {1.0/new_avg_time:.2f} Hz")
    print()


def test_serial_to_parallel_performance():
    """
    测试串联到并联转换的性能
    """
    print("=" * 80)
    print("测试串联到并联转换性能")
    print("=" * 80)
    
    # 初始化转换器
    dex_opt = OptimizedFuncSPTransDex()
    new_opt = OptimizedFuncSPTransNew()
    
    # 准备测试数据 (串联关节数据)
    # 使用与dex版本匹配的参数
    q_d_s = np.array([0.1 + 0.09765, 0.2, -0.1 + 0.09765, -0.2])  # 左右脚踝的pitch和roll
    qdot_d_s = np.array([0.01, 0.02, -0.01, -0.02])
    tor_d_s = np.array([1.0, 2.0, -1.0, -2.0])
    
    print(f"输入数据:")
    print(f"  q_d_s: {q_d_s}")
    print(f"  qdot_d_s: {qdot_d_s}")
    print(f"  tor_d_s: {tor_d_s}")
    print()
    
    # 预热运行
    for _ in range(10):
        # DEX优化版本
        dex_opt.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        dex_opt.calc_joint_pos_ref()
        dex_opt.calc_joint_tor_des()
        
        # NEW优化版本
        new_opt.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        new_opt.calc_joint_pos_ref()
        new_opt.calc_joint_tor_des()
    
    # 性能测试
    num_runs = 10000
    dex_times = []
    new_times = []
    
    # DEX优化版本性能测试
    print("测试优化后的DEX版本性能...")
    start_time = time.perf_counter()
    
    for i in range(num_runs):
        inner_start = time.perf_counter()
        
        dex_opt.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        dex_opt.calc_joint_pos_ref()
        dex_opt.calc_joint_tor_des()
        
        q_d_p_dex = np.zeros(4)
        qdot_d_p_dex = np.zeros(4)
        tor_d_p_dex = np.zeros(4)
        dex_opt.get_p_des(q_d_p_dex, qdot_d_p_dex, tor_d_p_dex)
        
        inner_end = time.perf_counter()
        dex_times.append(inner_end - inner_start)
    
    dex_total_time = time.perf_counter() - start_time
    
    # NEW优化版本性能测试
    print("测试NEW优化版本性能...")
    start_time = time.perf_counter()
    
    for i in range(num_runs):
        inner_start = time.perf_counter()
        
        new_opt.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        new_opt.calc_joint_pos_ref()
        new_opt.calc_joint_tor_des()
        
        q_d_p_new = np.zeros(4)
        qdot_d_p_new = np.zeros(4)
        tor_d_p_new = np.zeros(4)
        new_opt.get_p_des(q_d_p_new, qdot_d_p_new, tor_d_p_new)
        
        inner_end = time.perf_counter()
        new_times.append(inner_end - inner_start)
    
    new_total_time = time.perf_counter() - start_time
    
    # 输出结果
    print(f"优化后的DEX版本输出数据:")
    print(f"  q_d_p: {q_d_p_dex}")
    print(f"  qdot_d_p: {qdot_d_p_dex}")
    print(f"  tor_d_p: {tor_d_p_dex}")
    print()
    
    print(f"NEW优化版本输出数据:")
    print(f"  q_d_p: {q_d_p_new}")
    print(f"  qdot_d_p: {qdot_d_p_new}")
    print(f"  tor_d_p: {tor_d_p_new}")
    print()
    
    # 计算结果差异
    q_diff = np.abs(q_d_p_dex - q_d_p_new)
    qdot_diff = np.abs(qdot_d_p_dex - qdot_d_p_new)
    tor_diff = np.abs(tor_d_p_dex - tor_d_p_new)
    
    print(f"结果差异:")
    print(f"  q_d_p 差异: {q_diff}")
    print(f"  qdot_d_p 差异: {qdot_diff}")
    print(f"  tor_d_p 差异: {tor_diff}")
    print(f"  最大差异: {np.max([np.max(q_diff), np.max(qdot_diff), np.max(tor_diff)])}")
    print()
    
    # 性能统计
    dex_avg_time = np.mean(dex_times)
    new_avg_time = np.mean(new_times)
    speedup = new_avg_time / dex_avg_time if dex_avg_time > 0 else float('inf')
    
    print(f"性能统计 (运行 {num_runs} 次):")
    print(f"  优化后的DEX版本总时间: {dex_total_time:.4f} 秒")
    print(f"  NEW优化版本总时间: {new_total_time:.4f} 秒")
    print(f"  优化后的DEX版本平均时间: {dex_avg_time*1000:.6f} ms")
    print(f"  NEW优化版本平均时间: {new_avg_time*1000:.6f} ms")
    print(f"  性能提升 (DEX vs NEW): {speedup:.2f}x")
    print(f"  优化后的DEX版本平均频率: {1.0/dex_avg_time:.2f} Hz")
    print(f"  NEW优化版本平均频率: {1.0/new_avg_time:.2f} Hz")
    print()


def main():
    """
    主测试函数
    """
    print("优化版 FuncSPTrans 性能对比测试")
    print("比较优化后的 DEX 版本和 NEW 版本")
    print("开始测试...\n")
    
    # 测试并联到串联转换
    test_parallel_to_serial_performance()
    
    # 测试串联到并联转换
    test_serial_to_parallel_performance()
    
    print("=" * 80)
    print("所有性能对比测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()