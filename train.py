# train.py

"""
Main training script for the DDPM Trajectory model.
Updated to work with argparse config.
"""

import torch
import torch.nn as nn # 导入 PyTorch 的神经网络模块
import os # 导入操作系统接口模块
from tqdm import tqdm # 导入 tqdm 库用于显示进度条
# Import project modules
from model import UNet1D # 从 model.py 文件导入 UNet1D 模型
from utils import DiffusionUtils, get_time_embedding # Import get_time_embedding # 从 utils.py 导入 DiffusionUtils 类和 get_time_embedding 函数
from dataset import get_dataloader # 从 dataset.py 导入 get_dataloader 函数
# Import config
from config import get_config # 从 config.py 导入 get_config 函数

def save_checkpoint(state, filepath):
    """Saves model and optimizer state to a checkpoint file.""" # 保存模型和优化器状态到检查点文件
    torch.save(state, filepath) # 使用 PyTorch 的 save 函数保存状态字典到指定文件路径
    print(f"Checkpoint saved to {filepath}") # 打印保存成功的消息

def main():
    """Main training function.""" # 主训练函数
    #breakpoint()
    # --- 1. Parse Configuration ---
    config = get_config() # 调用 get_config 函数获取解析后的配置参数
    print(f"Starting training run: {config.run_name}") # 打印当前训练运行的名称
    print(f"Using device: {config.device}") # 打印正在使用的计算设备 (CPU/GPU)

    # --- 2. Setup ---
    os.makedirs(config.checkpoint_dir, exist_ok=True) # 创建用于保存检查点的根目录，如果已存在则不报错
    checkpoint_run_dir = os.path.join(config.checkpoint_dir, config.run_name) # 将运行名称附加到检查点根目录，形成特定运行的检查点目录路径
    os.makedirs(checkpoint_run_dir, exist_ok=True) # 创建特定运行的检查点目录

    # --- 3. Data Loading ---
    print("Loading data...") # 打印数据加载开始的信息
    dataloader = get_dataloader(config, train=True) # 调用 get_dataloader 函数，传入配置和 train=True 标志，获取训练数据加载器
    print(f"Training DataLoader created with {len(dataloader)} batches.") # 打印创建的训练 DataLoader 的批次数量

    # --- 4. Model, Diffusion, Optimizer ---
    print("Initializing model, diffusion, and optimizer...") # 打印模型、扩散工具和优化器初始化开始的信息
    model = UNet1D(config).to(config.device) # Pass config to model # 实例化 UNet1D 模型，并传入配置，然后将其移动到指定设备
    diffusion = DiffusionUtils(config) # Pass config to diffusion # 实例化 DiffusionUtils 类，并传入配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 使用 Adam 优化器，传入模型参数和学习率
    mse_loss = nn.MSELoss() # 定义均方误差 (MSE) 损失函数
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.") # 计算并打印模型中可训练参数的总数

    # --- 5. Training Loop ---
    print("Starting training loop...") # 打印训练循环开始的信息
    model.train() # 将模型设置为训练模式
    for epoch in range(config.num_epochs): # 遍历配置中指定的训练轮数
        print(f"\nEpoch {epoch+1}/{config.num_epochs}") # 打印当前轮次信息
        epoch_loss = 0.0 # 初始化当前轮次的总损失
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}") # 创建一个进度条，包装数据加载器，用于显示当前轮次的进度

        for batch_idx, batch_data in enumerate(progress_bar): # 遍历数据加载器中的每个批次
            # Move data to device
            batch_data = batch_data.to(config.device) # Shape: [B, C, T] # 将当前批次的数据移动到指定设备

            # Sample random timesteps
            t = diffusion.sample_timesteps(batch_data.shape[0]).to(config.device) # Shape: [B] # 为当前批次的每个样本随机采样一个时间步，并移动到设备

            # Apply forward diffusion to get noisy data and true noise
            noisy_data, true_noise = diffusion.noise_images(batch_data, t) # Shapes: [B, C, T] # 对原始数据应用前向扩散过程，得到加噪数据和真实的噪声

            # Generate time embedding and prepare model input
            t_emb = get_time_embedding(t, embedding_dim=config.time_embedding_dim) # Shape: [B, TIME_EMBEDDING_DIM] # 为采样的时间步生成时间嵌入
            t_emb_channel = torch.nn.functional.interpolate(
                t_emb[:, None, :], size=config.seq_length, mode='linear', align_corners=False
            ) # Shape: [B, 1, T] # 将时间嵌入在第二个维度增加一个维度，并线性插值到序列长度，以匹配数据维度
            model_input = torch.cat([noisy_data, t_emb_channel], dim=1) # Shape: [B, 2, T] # 在通道维度上拼接加噪数据和时间嵌入，作为模型的输入

            # Forward pass: predict noise
            predicted_noise = model(model_input) # Shape: [B, C, T] # 将拼接后的输入传递给模型，得到预测的噪声

            # Calculate loss
            loss = mse_loss(predicted_noise, true_noise) # 计算预测噪声和真实噪声之间的 MSE 损失
            #breakpoint()
            # Backward pass and optimization
            optimizer.zero_grad() # 清除优化器中所有参数的梯度缓存
            loss.backward() # 对损失进行反向传播，计算梯度
            optimizer.step() # 根据计算出的梯度更新模型参数

            epoch_loss += loss.item() # 累加当前批次的损失到轮次总损失
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}) # 在进度条后显示当前批次的损失

            # Log periodically
            if batch_idx % config.log_interval == 0: # 如果当前批次索引是日志间隔的倍数
                 print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}") # 打印当前批次的索引和损失

        avg_epoch_loss = epoch_loss / len(dataloader) # 计算当前轮次的平均损失
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}") # 打印轮次完成信息和平均损失

        # --- 6. Save Checkpoint Periodically ---
        if (epoch + 1) % config.save_ckpt_interval == 0 or epoch == config.num_epochs - 1: # 如果当前轮次是保存间隔的倍数，或者已经是最后一个轮次
            ckpt_path = os.path.join(checkpoint_run_dir, f"model_epoch_{epoch}.pth") # 构建检查点文件的保存路径
            save_checkpoint({
                'epoch': epoch, # 保存当前轮次
                'model_state_dict': model.state_dict(), # 保存模型的状态字典
                'optimizer_state_dict': optimizer.state_dict(), # 保存优化器的状态字典
                'loss': avg_epoch_loss, # 保存轮次平均损失
            }, ckpt_path) # 调用 save_checkpoint 函数保存检查点
            print(f"Checkpoint saved at epoch {epoch + 1}") # 打印检查点保存信息

    print("Training finished.") # 打印训练完成信息

if __name__ == "__main__": # 如果此脚本是作为主程序运行
    main() # 调用 main 函数开始训练




