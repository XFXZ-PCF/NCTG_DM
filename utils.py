# utils.py

"""
Utility functions for the DDPM process, including diffusion scheduling,
time embedding generation, and sampling logic.
Also includes helper functions for data visualization and saving.
Updated to work with argparse config.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 用于绘图
import os # 导入 os 模块用于文件和目录操作

# Note: We no longer import a global config instance here.
# Parameters will be passed explicitly or accessed via the config object passed from the main script.

class DiffusionUtils:
    """
    Handles the core mathematics of the forward and reverse diffusion processes.
    """
    def __init__(self, config):
        """
        Initializes the diffusion schedule using values from the config object.

        Args:
            config (argparse.Namespace): Configuration object containing diffusion parameters.
        """
        self.timesteps = config.diffusion_timesteps # 从配置中获取扩散总步数 T
        self.beta_start = config.beta_start # 从配置中获取 beta 调度的起始值
        self.beta_end = config.beta_end # 从配置中获取 beta 调度的结束值
        self.device = config.device # 从配置中获取计算设备 (e.g., 'cpu' or 'cuda')

        # Create the noise schedule (variances)
        self.beta = self.prepare_noise_schedule().to(self.device) # Shape: [T] # 生成并移动 beta 调度到指定设备
        self.alpha = 1. - self.beta # Shape: [T] # 计算 alpha_t = 1 - beta_t
        # Precompute cumulative product of alphas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Shape: [T] # 预计算 alpha_hat_t = prod(alpha_1, ..., alpha_t)

    def prepare_noise_schedule(self):
        """
        Prepares the beta (variance) schedule.
        Uses a simple linear schedule.

        Returns:
            torch.Tensor: Tensor of betas for each timestep. Shape: [T]
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps) # 使用线性插值生成 T 个 beta 值

    def noise_images(self, x, t):
        """
        Applies the forward diffusion process: adds noise to the data `x` at timestep `t`.

        Args:
            x (torch.Tensor): Original clean data. Shape: [B, C, T].
            t (torch.Tensor): Timestep indices. Shape: [B].

        Returns:
            tuple: (noisy_x, noise)
                - noisy_x (torch.Tensor): Data with noise added. Shape: [B, C, T].
                - noise (torch.Tensor): The noise that was added. Shape: [B, C, T].
        """
        # Get precomputed values for timestep t
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None] # Shape: [B, 1, 1] # 获取 sqrt(alpha_hat_t) 并调整形状以进行广播
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None] # Shape: [B, 1, 1] # 获取 sqrt(1 - alpha_hat_t) 并调整形状
        
        # Generate random noise
        noise = torch.randn_like(x) # Shape: [B, C, T] # 生成与输入数据 x 形状相同的随机高斯噪声
        
        # Apply forward diffusion formula: x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise
        noisy_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise # 根据公式计算加噪后的数据 x_t
        return noisy_x, noise # 返回加噪数据和添加的噪声

    def sample_timesteps(self, n):
        """
        Randomly samples timesteps for a batch of data.

        Args:
            n (int): Batch size.

        Returns:
            torch.Tensor: Tensor of randomly sampled timesteps. Shape: [n].
        """
        # Sample timesteps uniformly from 1 to T-1 (avoids t=0 issues)
        return torch.randint(low=1, high=self.timesteps, size=(n,)) # 从 1 到 T-1 均匀随机采样 n 个时间步

    def sample(self, model, config, n):
        """
        Generates new samples using the reverse diffusion (sampling or generating) process.

        Args:
            model (torch.nn.Module): The trained denoising U-Net model.
            config (argparse.Namespace): Configuration object containing model/data parameters.
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples. Shape: [n, channels, seq_length].
        """
        model.eval() # 将模型设置为评估模式，关闭如 dropout 等训练时特有的行为
        with torch.no_grad(): # 在此代码块内禁用梯度计算，节省内存并加快速度
            # 1. Start from pure noise
            x = torch.randn((n, config.trajectory_dim, config.seq_length)).to(self.device) # Shape: [n, C, T] # 从标准正态分布初始化纯噪声图像 x_T
            
            # 2. Iteratively denoise from T to 1
            for i in reversed(range(1, self.timesteps)): # 从 T-1 到 1 反向迭代时间步
                t = (torch.ones(n) * i).long().to(self.device) # Timestep tensor [n] # 创建一个形状为 [n] 的张量，所有元素都是当前时间步 i

                # 3. Model prediction (predict noise)
                # Prepare time embedding for the model
                t_emb = get_time_embedding(t, embedding_dim=config.time_embedding_dim) # Shape: [n, TIME_EMBEDDING_DIM] # 为当前时间步 t 生成时间嵌入
                t_emb_channel = torch.nn.functional.interpolate(
                    t_emb[:, None, :], size=config.seq_length, mode='linear', align_corners=False
                ) # Shape: [n, 1, T] # 将时间嵌入插值到与数据序列相同的长度，并增加通道维度
                model_input = torch.cat([x, t_emb_channel], dim=1) # Shape: [n, 2, T] # 将噪声数据 x 和时间嵌入在通道维度上拼接作为模型输入

                predicted_noise = model(model_input) # Shape: [n, C, T] # 使用模型预测当前步骤的噪声

                # 4. Get precomputed values for timestep t
                alpha = self.alpha[t][:, None, None] # Shape: [n, 1, 1] # 获取当前时间步的 alpha_t
                alpha_hat = self.alpha_hat[t][:, None, None] # Shape: [n, 1, 1] # 获取当前时间步的 alpha_hat_t
                beta = self.beta[t][:, None, None] # Shape: [n, 1, 1] # 获取当前时间步的 beta_t

                # 5. Add noise during sampling (except for the last step)
                if i > 1: # 如果不是最后一步 (t > 1)
                    noise = torch.randn_like(x) # Shape: [n, C, T] # 采样随机噪声
                else: # 如果是最后一步 (t = 1)
                    noise = torch.zeros_like(x) # Shape: [n, C, T] # 不添加噪声

                # 6. Reverse diffusion step (remove estimated noise)
                # Formula: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_hat_t) * pred_noise) + sqrt(beta_t) * noise
                mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) # 计算去噪步骤的均值部分
                x = mean + torch.sqrt(beta) * noise # 更新 x_{t-1}

        model.train() # 将模型恢复为训练模式
        return x # Shape: [n, C, T] # 返回最终生成的样本


def get_time_embedding(timesteps, embedding_dim):
    """
    Generates sinusoidal time embeddings for a given timestep.
    This provides the model with information about the current timestep in the diffusion process.

    Args:
        timesteps (torch.Tensor): Tensor of timesteps. Shape: [B].
        embedding_dim (int): Dimension of the time embedding.

    Returns:
        torch.Tensor: Time embedding tensor. Shape: [B, embedding_dim].
    """
    half_dim = embedding_dim // 2 # 计算嵌入维度的一半
    # Calculate frequencies
    emb = np.log(10000) / (half_dim - 1) # 计算频率对数的基
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb) # 生成频率向量
    # Multiply timesteps by frequencies
    emb = timesteps.float()[:, None] * emb[None, :] # Shape: [B, half_dim] # 将时间步与频率相乘
    # Apply sin and cos
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # Shape: [B, embedding_dim] # 对前半部分和后半部分分别应用 sin 和 cos 并拼接
    # Pad if embedding_dim is odd
    if embedding_dim % 2 == 1: # 如果嵌入维度是奇数
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0)) # 在最后一个维度填充一个0
    return emb.to(timesteps.device) # 将嵌入移动到与 timesteps 相同的设备并返回


def save_and_visualize_samples(sampled_trajectories_tensor, config, ckpt_epoch, num_to_plot=5):
    """
    Saves generated samples to disk as a .npy file and plots a few of them.

    Args:
        sampled_trajectories_tensor (torch.Tensor): Tensor of generated samples. Shape: [N, C, T].
        config (argparse.Namespace): Configuration object containing paths and run name.
        ckpt_epoch (int): Epoch of the model used for sampling.
        num_to_plot (int): Number of samples to plot and save as a PNG.
    """
    # 1. Convert PyTorch tensor to NumPy array
    # Squeeze the channel dimension to get [N, T]
    sampled_trajectories_np = sampled_trajectories_tensor.cpu().numpy().squeeze(1) # 将张量移到 CPU，转为 NumPy 数组，并移除通道维度 (假设 C=1)
    print(f"Generated samples shape: {sampled_trajectories_np.shape}") # 打印生成样本的形状

    # 2. Save samples as .npy file
    run_save_dir = os.path.join(config.results_dir, config.run_name) # 构建保存结果的目录路径
    os.makedirs(run_save_dir, exist_ok=True) # 创建目录，如果已存在则不报错
    save_path = os.path.join(run_save_dir, f"samples_epoch_{ckpt_epoch}.npy") # 构建 .npy 文件的保存路径
    np.save(save_path, sampled_trajectories_np) # 保存 NumPy 数组到 .npy 文件
    print(f"Samples saved to {save_path}") # 打印保存路径

    # 3. Plot and save a few samples as PNG
    num_samples = sampled_trajectories_np.shape[0] # 获取生成样本的总数量 N
    num_plots = min(num_to_plot, num_samples) # 确定实际要绘制的样本数量（不超过总数）
    
    if num_plots > 0: # 如果有样本需要绘制
        plt.figure(figsize=(15, 5)) # 创建一个新的图形窗口，并设置大小
        for i in range(num_plots): # 遍历要绘制的样本
            plt.plot(sampled_trajectories_np[i], label=f'Sample {i+1}') # 绘制第 i 个样本的轨迹
        plt.xlabel('Time Step') # 设置 x 轴标签
        plt.ylabel('Speed (m/s)') # 设置 y 轴标签
        plt.title(f'Generated Vehicle Trajectories (Epoch {ckpt_epoch})') # 设置图形标题
        plt.legend() # 显示图例
        plot_path = os.path.join(run_save_dir, f"samples_plot_epoch_{ckpt_epoch}.png") # 构建 PNG 图像的保存路径
        plt.savefig(plot_path) # 保存图像到文件
        plt.close() # 关闭图形窗口以释放内存
        print(f"Sample plot saved to {plot_path}") # 打印图像保存路径
    else: # 如果没有样本可绘制
        print("No samples to plot.") # 打印提示信息




