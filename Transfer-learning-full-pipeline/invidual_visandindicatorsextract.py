# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:48:55 2025

@author: uqpfan
I aim to use this manuscript to create individual trajectories plots and 
doing wavelet transform to analyse,
automaticlly changed the 
focus on two indiicators : ampitude and 
"""

import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import datetime, time
import pandas as pd
import tqdm
from tqdm import trange
import numpy as np
import os
import sys
import math
import shutil
import matplotlib.pyplot as plt
import pywt
from scipy.signal import cwt, morlet2
from WT_util import compute_cwt_with_mexican_hat, compute_wavelet_energy
# In[1]
#specify the datasourece
number_sampleGENERATE = 500 #used to be 50 and increase to 500
gen_data_path = r'D:\VT_GENERATOR\2025WEEK1013\TL_pre\tramsfer-learningFROMQWEN\500samplegenerate.npy' #this specify the generated trajectories.
gen_data = np.load(gen_data_path)  # Shape: [N, T] 500*301
gen_save_dir = './gen30Vis'
# In[1-1]
#directly first visluate the 30s existing trajectories.
'''
for i in trange(gen_data.shape[0]):
    name = 'traj_{}'.format(i)
    #run_save_dir = os.path.join(gen_save_dir, name) # 构建保存结果的目录路径
    os.makedirs(gen_save_dir, exist_ok=True) # 创建目录，如果已存在则不报错
    y = gen_data [i,:]#this signify the generated trajectory...
    plt.figure(figsize=(15, 5)) # 创建一个新的图形窗口，并设置大小
    timeplot =[ round(0.1 * x,1) for x in list(range(gen_data.shape[1]))]
    plt.plot(timeplot,y) # 绘制第 i 个样本的轨迹
    plt.xlabel('Time (s)') # 设置 x 轴标签
    plt.ylabel('Speed (m/s)') # 设置 y 轴标签
    plt.title('Generated Vehicle Trajectories' + name) # 设置图形标题
    #plt.legend() # 显示图例
    plot_path = os.path.join(gen_save_dir, (name + ".png")) # 构建 PNG 图像的保存路径
    plt.savefig(plot_path) # 保存图像到文件
    plt.close() # 关闭图形窗口以释放内存
    print(f"Sample plot saved to {plot_path}") # 打印图像保存路径
'''
# In[2]
#using certain method to add a 2-3 second's extention.
#in case I use the linear regression;
# Apply the extension function and already test;
#extended_time, extended_velocity, extension_samples = extend_velocity_sequence( original_sequence = gen_data[1,:], time_step=0.1, extension_duration = 2.0)

Pad_width = 20 # 每边扩展3个点
extend_gendata = np.zeros([gen_data.shape[0],(Pad_width + gen_data.shape[1]+ Pad_width)])
for i in trange(gen_data.shape[0]):
    '''
    extended_time, extended_velocity, extension_samples = extend_velocity_sequence_smooth( 
        original_sequence = gen_data[i,:],
           time_step=0.1, extension_duration = 2.0) #index is i 
    '''
    extended_velocity =  pywt.pad(x = gen_data[i,:], 
                                  pad_widths = Pad_width ,
                                  mode='smooth') #THIS IS a mode I think RELATIVEly resonable;
    #let me try 
    #only the extended_velocity is needed,
    extend_gendata[i,:] = extended_velocity
    #breakpoint()
del i
# In[3]
#save the extend data as excel for sourcing corresponding speed;
extenddata_path = "extend_gendata.xlsx" #save to the path that
ID_col = np.arange(0,extend_gendata.shape[0]) 
ID_col = ID_col[:, np.newaxis]
savedata = np.concatenate( (ID_col,extend_gendata),axis= 1)
df = pd.DataFrame(savedata)
# ---------- ③ 保存为 CSV 文件 ----------
df.to_excel(extenddata_path )

# In[4]
#following codes are not used after comparing.
#start using the wavelet as the syntehtic data case..
#here I copy the previous codes
#I do not care the absoulute timestep values, only starting
for i in trange(extend_gendata.shape[0]):
    vid = i
    denoise_speed = extend_gendata[i,:]

    # 时间步长固定为 0.1s
    dt = 0.1
    
    # 定义尺度范围 1-64; used to 65
    scales = np.arange(0.1, 6.5, 0.1) #shoudl be used for the paper
    
    #figure out CWT
    
    cwt_matrix = compute_cwt_with_mexican_hat(denoise_speed, scales, dt)
    
    # e average wavelet-based energy at b over time domain
    wavelet_energy = compute_wavelet_energy(cwt_matrix, scales)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Wavelet Transform of generated CAR{}'.format(vid), fontsize=14)
    
    #for following, I NEED edit set the true zeros points
    hours_since_midnight = [ round(0.1 * x - 0.1*Pad_width,
                                   1) for x in list(range(extend_gendata.shape[1]))] #smaller least 
    hours_since_midnight = np.array(hours_since_midnight) #otherwise AttributeError: 'list' object has no attribute 'min'
    rel_time = hours_since_midnight #this is a compromise
    #hours_since_midnight = [round(x - Pad_width*0.1,1 )for x in hours_since_midnight ]
    # 上子图：速度随时间变化
    ax1 = axes[0]
    ax1.plot(hours_since_midnight, denoise_speed, 'b-', 
             linewidth=1.5, label='generated Speed')
    ax1.set_ylabel('Speed (m/s)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 在 [] 时间范围内找到速度最低点
    #mask = (hours_since_midnight >= 7.84) & (hours_since_midnight <= 7.87)
    mask = (hours_since_midnight >= hours_since_midnight.min()) & (hours_since_midnight <= hours_since_midnight.max())
    if np.any(mask):
        min_idx = np.argmin(denoise_speed[mask])
        actual_min_idx = np.where(mask)[0][min_idx]
        min_rel_time = rel_time[actual_min_idx]
        min_hours = hours_since_midnight[actual_min_idx]
        min_speed = denoise_speed[actual_min_idx]
        #breakpoint()
        # 用红圈标记最低点
        ax1.plot(min_hours, min_speed, 'ro', markersize=8, label=f'Min @ {min_rel_time:.3f}s')
        ax1.annotate(f'{min_rel_time:.3f}s', 
                    xy = (min_hours, min_speed), 
                    xytext = (min_hours+0.001, min_speed +1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'))
        ax1.legend(loc='upper right')
        
    # 创建上子图的辅助x轴（使用 rel_time，单位为秒）
    ax1_top = ax1.twiny()
    # 设置辅助x轴的范围与主x轴对应
    ax1_top.set_xlim(ax1.get_xlim())
    # 获取当前x轴范围对应的 rel_time 值
    x_min, x_max = ax1.get_xlim()
    # 找到对应的 rel_time 范围
    rel_time_min = rel_time[0]  # 或者根据 hours_since_midnight 找到对应值
    rel_time_max = rel_time[-1]
    
    # 由于 hours_since_midnight 和 rel_time 是一一对应的，我们可以设置辅助x轴的刻度
    # 计算辅助x轴的刻度位置（在 hours_since_midnight 坐标系中）
    n_ticks = 10  # 设置刻度数量
    tick_positions_hours = np.linspace(x_min, x_max, n_ticks)
    # 找到这些位置对应的 rel_time 值（线性插值）
    tick_positions_rel_time = np.interp(tick_positions_hours, hours_since_midnight, rel_time)
    
    # 设置辅助x轴的刻度和标签
    ax1_top.set_xticks(tick_positions_hours)
    ax1_top.set_xticklabels([f'{t:.1f}' for t in tick_positions_rel_time])
    ax1_top.set_xlabel('Relative Time (seconds)', fontsize=12)    
            
    #change as the oscillation changes----------------------------------------------------------    
    #there is no cerain range.. or use a large.
    # 根据轨迹位置区间 ，找到对应的 hours_since_midnight 的 low 和 high
    time_interval = (0, 30) #by doing a eye inspection;
    
    mask_position = (hours_since_midnight >= time_interval[0]) & (hours_since_midnight <= time_interval[1])
    
    if np.any(mask_position):
        low = hours_since_midnight[mask_position].min()
        high = hours_since_midnight[mask_position].max()
    else:
        #raise 
        print("cterian apce range is not fund,mightbe too short for ID:", vid,"\n")
        low = hours_since_midnight.min()
        high = hours_since_midnight.max()
    
    #low subplot：WT-energy
    ax2 = axes[1]
    ax2.plot(hours_since_midnight, wavelet_energy, 'm-', linewidth=1.5, label='Wavelet Energy')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Average Wavelet-based Energy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 设置分析区间背景为浅灰色
    ax2.axvspan(low, high, alpha=0.3, color='gray', label='precursor stage spatial Analysis Interval')
    
    # 找到所有局部极大值（波峰） - 不能只在分析区间内寻找
    def find_peaks(data, min_distance=1):
        peaks = []
        n = len(data)
        for i in range(1, n-1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # 确保不是噪声引起的微小波动
                if len(peaks) == 0 or (i - peaks[-1] >= min_distance):
                    peaks.append(i)
        return peaks
    
    # 获取分析区间内的索引， if use this one , then only use peak within the range ..
    #analysis_mask = (hours_since_midnight >= low) & (hours_since_midnight <= high)
    analysis_mask = (hours_since_midnight >= hours_since_midnight.min()) & (hours_since_midnight <= hours_since_midnight.max())
    analysis_indices = np.where(analysis_mask)[0]
    
    # 如果分析区间内有数据，找到该区间内的波峰
    if len(analysis_indices) > 0:
        # 只在分析区间内寻找波峰
        local_wavelet_energy = wavelet_energy[analysis_mask]
        local_peaks = find_peaks(local_wavelet_energy, min_distance=2)
        
        # 转换回全局索引
        peak_indices = analysis_indices[local_peaks] if len(local_peaks) > 0 else []
    else:
        peak_indices = []
    
    # 用红色虚线竖线标记每个波峰，并标注时间戳
    for i, peak_idx in enumerate(peak_indices):
        peak_time = hours_since_midnight[peak_idx]
        peak_rel_time = round(rel_time[peak_idx],2) #reduce "0";
        ax2.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7)
        # 在图上标注时间戳
        ax2.annotate(f'{peak_rel_time:.3f}', 
                    xy=(peak_time, wavelet_energy[peak_idx]), 
                    xytext=(peak_time + 0.001/6, wavelet_energy[peak_idx] + 0.1*np.ptp(wavelet_energy)),
                    rotation=90, fontsize=8, ha='center', va='bottom',
                    color='red')
    
    # 在图例中添加分析区间
    ax2.legend(loc='upper right')
    
    # 设置x轴格式
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # 调整布局
    plt.tight_layout()
    savepth = './gen30Vis_Syn' #this can be fixed
    os.makedirs(savepth, exist_ok = True)
    # 保存图像
    #breakpoint()
    try:
        plt.savefig(savepth + "\\" + 'Synthe_ID{}.PNG'.format(vid), dpi=600, bbox_inches='tight')
    except OSError:
        pass
        print("it already generate for{}".format(vid))

    plt.close()











# In[7]    
def extend_velocity_sequence_smooth(original_sequence, time_step=0.1, extension_duration=2.0, smooth_window=5):
    """
    Extend velocity sequence with smooth transition at connection points to avoid discontinuities.
    
    Parameters:
    -----------
    original_sequence : array-like
        Original velocity sequence with 301 elements
    time_step : float
        Time step between samples (default: 0.1 seconds)
    extension_duration : float
        Duration to extend at each end in seconds (default: 2.0 seconds)
    smooth_window : int
        Number of samples for smoothing transition at connection points
    
    Returns:
    --------
    extended_time : numpy array
        Time axis for extended sequence
    extended_sequence : numpy array
        Extended velocity sequence with smooth transitions
    extension_samples : int
        Number of samples added at each end
    """
    
    # Convert to numpy array
    original_sequence = np.array(original_sequence)
    n_original = len(original_sequence)
    
    # Calculate extension samples
    extension_samples = int(extension_duration / time_step)
    
    # Create original time axis
    original_time = np.arange(0, n_original * time_step, time_step)
    
    # Use larger window for more stable trend estimation (2 seconds)
    trend_window = int(2.0 / time_step)  # 20 samples for 2-second window
    trend_window = min(trend_window, n_original // 3)  # Ensure window is not too large
    
    # 1. Front-end extension with smooth connection
    front_trend_data = original_sequence[:trend_window]
    front_trend_time = original_time[:trend_window].reshape(-1, 1)
    
    # Fit linear regression for front trend
    front_model = LinearRegression()
    front_model.fit(front_trend_time, front_trend_data)
    front_slope = front_model.coef_[0]
    
    # Generate front extension time points
    front_extension_time = np.arange(-extension_duration, 0, time_step)
    
    # Calculate extension using the slope but adjust intercept for smooth connection
    # Use the value at first point minus slope*time_step to start extension
    front_intercept_adjusted = original_sequence[0] - front_slope * time_step
    
    front_extension_sequence = front_slope * front_extension_time + front_intercept_adjusted
    
    # Apply smoothing to front extension
    if len(front_extension_sequence) > 5:
        front_extension_sequence = np.convolve(
            front_extension_sequence, 
            np.ones(5)/5, 
            mode='same'
        )
    
    # 2. Back-end extension with smooth connection
    back_trend_data = original_sequence[-trend_window:]
    back_trend_time = original_time[-trend_window:].reshape(-1, 1)
    
    # Fit linear regression for back trend
    back_model = LinearRegression()
    back_model.fit(back_trend_time, back_trend_data)
    back_slope = back_model.coef_[0]
    
    # Generate back extension time points
    back_extension_time = np.arange(
        original_time[-1] + time_step, 
        original_time[-1] + extension_duration + time_step, 
        time_step
    )
    
    # Adjust intercept to ensure smooth connection at the end
    back_intercept_adjusted = original_sequence[-1] + back_slope * time_step
    back_extension_sequence = back_slope * back_extension_time + back_intercept_adjusted
    
    # Apply smoothing to back extension
    if len(back_extension_sequence) > 5:
        back_extension_sequence = np.convolve(
            back_extension_sequence, 
            np.ones(5)/5, 
            mode='same'
        )
    
    # 3. Smooth transition at connection points
    # For front connection: blend last few points of front extension with first few points of original
    transition_points = min(smooth_window, len(front_extension_sequence), len(original_sequence)//10)
    
    if transition_points > 0:
        # Create smooth transition weights
        front_weights = np.linspace(1, 0, transition_points)
        original_weights = np.linspace(0, 1, transition_points)
        
        # Adjust the last few points of front extension
        front_extension_sequence[-transition_points:] = (
            front_extension_sequence[-transition_points:] * front_weights + 
            original_sequence[:transition_points] * original_weights
        )
    
    # For back connection: blend last few points of original with first few points of back extension
    if transition_points > 0:
        back_weights = np.linspace(0, 1, transition_points)
        original_end_weights = np.linspace(1, 0, transition_points)
        
        # Adjust the first few points of back extension
        back_extension_sequence[:transition_points] = (
            original_sequence[-transition_points:] * original_end_weights + 
            back_extension_sequence[:transition_points] * back_weights
        )
    
    # 4. Combine all segments
    extended_time = np.concatenate([
        front_extension_time,
        original_time,
        back_extension_time
    ])
    
    extended_sequence = np.concatenate([
        front_extension_sequence,
        original_sequence,
        back_extension_sequence
    ])
    
    return extended_time, extended_sequence, extension_samples










