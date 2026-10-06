# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 17:21:44 2025

@author: uqpfan
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import cwt, morlet2

def generate_sample_trajectory():
    """
    生成一个72秒的、包含多个加减速过程的示例速度轨迹（单位：ft/s）
    采样频率为10Hz (0.1s间隔)，总长度720个点
    """
    t = np.linspace(0, 72, 720)  # 72秒，0.1秒间隔
    speed = np.zeros_like(t)
    
    # 基础速度
    base_speed = 60.0  # 基础速度 60 ft/s
    
    # 添加多个加减速过程
    # 第一个减速-加速过程
    mask1 = (t >= 8) & (t <= 12)
    speed[mask1] = base_speed - 30 * np.sin(np.pi * (t[mask1] - 8) / 4)**2
    
    # 第二个减速-加速过程
    mask2 = (t >= 20) & (t <= 25)
    speed[mask2] = base_speed - 25 * np.sin(np.pi * (t[mask2] - 20) / 5)**2
    
    # 第三个减速-加速过程
    mask3 = (t >= 35) & (t <= 42)
    speed[mask3] = base_speed - 35 * np.sin(np.pi * (t[mask3] - 35) / 7)**2
    
    # 第四个减速-加速过程
    mask4 = (t >= 50) & (t <= 55)
    speed[mask4] = base_speed - 20 * np.sin(np.pi * (t[mask4] - 50) / 5)**2
    
    # 第五个减速-加速过程
    mask5 = (t >= 62) & (t <= 68)
    speed[mask5] = base_speed - 28 * np.sin(np.pi * (t[mask5] - 62) / 6)**2
    
    # 其他时间段保持基础速度
    speed[(t < 8) | ((t > 12) & (t < 20)) | ((t > 25) & (t < 35)) | 
          ((t > 42) & (t < 50)) | ((t > 55) & (t < 62)) | (t > 68)] = base_speed
    
    # 添加一些噪声使数据更真实
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(t))
    speed = speed + noise
    
    return t, speed

def mexican_hat_wavelet(t, a=1.0, b=0.0):
    """
    墨西哥帽小波函数（论文公式4）
    w((t-b)/a) = (1 - ((t-b)/a)^2) * exp(-((t-b)/a)^2 / 2)
    """
    x = (t - b) / a
    return (1 - x**2) * np.exp(-x**2 / 2)

def compute_cwt_with_mexican_hat(signal, scales, dt=0.1):
    """
    使用墨西哥帽小波计算连续小波变换
    根据论文公式3: T(a,b) = 1/sqrt(a) * ∫ x(t) * w((t-b)/a) dt
    """
    n = len(signal)
    cwt_matrix = np.zeros((len(scales), n), dtype=np.float64)
    
    t = np.arange(n) * dt  # 时间轴
    
    for i, a in enumerate(scales):
        if a <= 0:
            continue
        for j in range(n):
            b = t[j]
            # 计算小波函数在当前尺度和位置的值
            wavelet_values = mexican_hat_wavelet(t, a, b)
            # 计算连续小波变换系数（数值Integral turns to be as these is no fucntion）
            cwt_matrix[i, j] = (1/np.sqrt(a)) * np.sum(signal * wavelet_values) * dt
    
    return cwt_matrix

def compute_wavelet_energy(cwt_coeffs, scales):
    """
    计算平均小波能量（论文公式6）
    Eb = 1/max(a) * ∫ |T(a,b)|^2 da
    """
    n_time_points = cwt_coeffs.shape[1]
    energy = np.zeros(n_time_points)
    
    max_a = np.max(scales)
    
    for b_idx in range(n_time_points):
        # 对所有尺度a计算 |T(a,b)|^2 的积分
        energy_integrand = np.abs(cwt_coeffs[:, b_idx])**2
        # 使用梯形法则进行数值积分
        energy[b_idx] = (1.0 / max_a) * np.trapz(energy_integrand, scales)
    
    return energy

# 主程序
if __name__ == "__main__":
    # 生成示例速度轨迹
    t, speed = generate_sample_trajectory()
    
    # 设置参数
    dt = 0.1  # 时间步长（秒）
    scales = np.arange(1, 5, 1)  # 尺度从1到64（论文Fig.2使用1-64） #but too large..
    
    # 计算连续小波变换
    print("计算连续小波变换...")
    cwt_matrix = compute_cwt_with_mexican_hat(speed, scales, dt)
    
    # 计算小波能量
    print("计算小波能量...")
    wavelet_energy = compute_wavelet_energy(cwt_matrix, scales)
    
    # 创建图形，复现论文Fig.2的4个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wavelet Transform Analysis (Reproducing Fig.2 from Paper)', fontsize=16)
    
    # 子图(a): 速度时间序列与叠加的墨西哥帽小波
    ax1 = axes[0, 0]
    ax1.plot(t, speed, 'b-', linewidth=2, label='Speed')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Speed (ft/s)')
    ax1.set_title('(a) Speed time-series with superimposed Mexican hat wavelet')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 在t=8.267s处叠加一个a=32的墨西哥帽小波（论文中的示例点）
    example_time = 8.267
    example_scale = 2
    time_idx = int(example_time / dt)
    
    # 绘制叠加的小波（仅在示例点附近）
    t_local = np.linspace(example_time - 5, example_time + 5, 200)
    wavelet_local = mexican_hat_wavelet(t_local, example_scale, example_time)
    # 缩放小波使其在图上可见
    wavelet_scaled = wavelet_local * 10 + speed[time_idx]  # 调整幅度和偏移
    ax1.plot(t_local, wavelet_scaled, 'r--', linewidth=2, label=f'Mexican hat (a={example_scale})')
    ax1.axvline(x=example_time, color='k', linestyle=':', alpha=0.7)
    ax1.legend()
    
    # 标记点A
    ax1.plot(example_time, speed[time_idx], 'ro', markersize=8, label='Point A')
    ax1.annotate('A', xy=(example_time, speed[time_idx]), xytext=(example_time+0.5, speed[time_idx]+5),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 子图(b): 在特定尺度a=32的WT系数
    ax2 = axes[0, 1]
    scale_idx = np.where(scales == example_scale)[0][0]
    wt_coeffs_at_scale = cwt_matrix[scale_idx, :]
    ax2.plot(t, wt_coeffs_at_scale, 'g-', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('WT Coefficient T(a,b)')
    ax2.set_title(f'(b) WT coefficients, T(a,b), at scale a={example_scale}')
    ax2.grid(True, alpha=0.3)
    
    # 标记对应点A'
    ax2.plot(example_time, wt_coeffs_at_scale[time_idx], 'go', markersize=8, label="Point A'")
    ax2.annotate("A'", xy=(example_time, wt_coeffs_at_scale[time_idx]), 
                xytext=(example_time+0.5, wt_coeffs_at_scale[time_idx]+5),
                arrowprops=dict(arrowstyle='->', color='green'))
    ax2.legend()
    
    # 子图(c): |T(a,b)|的等高线图（尺度1-64）
    ax3 = axes[1, 0]
    # 创建尺度和时间的网格
    T, S = np.meshgrid(t, scales)
    contour = ax3.contourf(T, S, np.abs(cwt_matrix), levels=50, cmap='viridis')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Scale, a')
    ax3.set_title('(c) Contour of |T(a,b)| from scale 1-64')
    plt.colorbar(contour, ax=ax3, label='|T(a,b)|')
    ax3.grid(True, alpha=0.3)
    
    # 标记点A''
    ax3.plot(example_time, example_scale, 'wo', markersize=8, markeredgecolor='black', label="Point A''")
    ax3.annotate("A''", xy=(example_time, example_scale), 
                xytext=(example_time+2, example_scale+10),
                arrowprops=dict(arrowstyle='->', color='white'))
    ax3.legend()
    
    # 子图(d): 平均小波能量分布
    ax4 = axes[1, 1]
    ax4.plot(t, wavelet_energy, 'm-', linewidth=2)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Average Wavelet-based Energy')
    ax4.set_title('(d) Temporal distribution of average wavelet-based energy across scales')
    ax4.grid(True, alpha=0.3)
    
    # 标记点A'''
    energy_at_point = wavelet_energy[time_idx]
    ax4.plot(example_time, energy_at_point, 'mo', markersize=8, label="Point A'''")
    ax4.annotate("A'''", xy=(example_time, energy_at_point), 
                xytext=(example_time+0.5, energy_at_point+0.5),
                arrowprops=dict(arrowstyle='->', color='magenta'))
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 验证计算：打印示例点的值
    print(f"\n验证计算结果（在t={example_time}s, a={example_scale}处）:")
    print(f"速度值: {speed[time_idx]:.4f} ft/s")
    print(f"WT系数 T(a,b): {wt_coeffs_at_scale[time_idx]:.4f}")
    print(f"|T(a,b)|: {np.abs(wt_coeffs_at_scale[time_idx]):.4f}")
    print(f"小波能量 Eb: {energy_at_point:.4f}")