# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 21:57:18 2025

@author: uqpfan
"""

# train.py

"""
Main training script for DDPM Trajectory model.
- Fine-tunes only decoder (upsampling blocks + output conv)
- Uses ReduceLROnPlateau scheduler
- Plots and saves training loss curve
"""
# In[0]
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet1D
from utils import DiffusionUtils, get_time_embedding
from dataset import get_dataloader
from config import get_config


# In[1]
def freeze_encoder(model, freeze_bottleneck= False):
    """
    冻结UNet1D模型中的编码器层（downs 和 pool），可选是否冻结瓶颈层。

    Args:
        model (UNet1D): 已实例化的UNet1D模型。
        freeze_bottleneck (bool): 是否冻结瓶颈层。默认True。
    """
    # 冻结编码器部分
    
    for param in model.downs.parameters():
        param.requires_grad = False

    # 冻结下采样层
    for param in model.pool.parameters():
        param.requires_grad = False
    #捕获了最高级别的抽象特征。​​建议解冻瓶颈层
    # 可选冻结瓶颈层
    if freeze_bottleneck:
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    print("✅ Encoder layers have been frozen. "
          f"Bottleneck frozen: {freeze_bottleneck}")


def save_checkpoint(state, filepath):
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_loss_curve(train_losses, config, epoch):
    run_dir = os.path.join(config.results_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'loss_curve_epoch_{epoch}.png'))
    plt.close()


def main():
    config = get_config()
    print(f"Starting fine-tuning run: {config.run_name}")
    print(f"Using device: {config.device}")

    # Setup directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_run_dir = os.path.join(config.checkpoint_dir, config.run_name)
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, config.run_name), exist_ok=True)

    # Data
    print("Loading real trajectory data...")
    dataloader = get_dataloader(config, train=True)
    print(f"Training DataLoader created with {len(dataloader)} batches.")

    # Model
    model = UNet1D(config).to(config.device)

    # Load pretrained weights and set a ERROR if not read
    if config.pretrained_model_path and os.path.exists(config.pretrained_model_path):
        print(f"Loading pretrained weights from {config.pretrained_model_path}...")
        checkpoint = torch.load(config.pretrained_model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Pretrained weights loaded.")
    else:
        raise FileNotFoundError(f"Pretrained model not found at {config.pretrained_model_path}")

    # 🔒 Freeze encoder
    freeze_encoder(model,freeze_bottleneck=False) #I set as False.

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}, Trainable params: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")

    # Diffusion & Loss
    diffusion = DiffusionUtils(config)
    criterion = nn.MSELoss()

    # Optimizer: only trainable params
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=4e-5  # Start with 1e-5 as requested
    )

    # Scheduler: ReduceLROnPlateau on epoch loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    train_losses = []
    model.train() #aginst vs model.eval()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
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
            breakpoint() #x in wrong dimension before..
            model_input = torch.cat([noisy_data, t_emb_channel], dim=1) # Shape: [B, 2, T] # 在通道维度上拼接加噪数据和时间嵌入，作为模型的输入

            # Forward pass: predict noise
            predicted_noise = model(model_input) # Shape: [B, C, T] # 将拼接后的输入传递给模型，得到预测的噪声

            # Calculate loss
            loss = criterion(predicted_noise, true_noise)
            #loss = mse_loss(predicted_noise, true_noise) # 计算预测噪声和真实噪声之间的 MSE 损失
            #breakpoint()
            # Backward pass and optimization
            optimizer.zero_grad() # 清除优化器中所有参数的梯度缓存
            loss.backward() # 对损失进行反向传播，计算梯度
            optimizer.step() # 根据计算出的梯度更新模型参数

            epoch_loss += loss.item() # 累加当前批次的损失到轮次总损失
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}) # 在进度条后显示当前批次的损失
        
        '''
        for batch_data in progress_bar:
            batch_data = batch_data.to(config.device)  # [B, 1, T]

            t = diffusion.sample_timesteps(batch_data.shape[0]).to(config.device)
            noisy_data, true_noise = diffusion.noise_images(batch_data, t)

            t_emb = get_time_embedding(t, embedding_dim=config.time_embedding_dim)
            t_emb_channel = torch.nn.functional.interpolate(
                t_emb[:, None, :], size=config.seq_length, mode='linear', align_corners=False
            )
            model_input = torch.cat([noisy_data, t_emb_channel], dim=1)  # [B, 2, T]

            predicted_noise = model(model_input)
            loss = criterion(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        '''
        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Step scheduler based on epoch loss
        scheduler.step(avg_epoch_loss)

        # Save checkpoint every N epochs or at last epoch
        if (epoch + 1) % config.save_ckpt_interval == 0 or epoch == config.num_epochs - 1:
            ckpt_path = os.path.join(checkpoint_run_dir, f"model_epoch_{epoch}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'config': vars(config),
            }, ckpt_path)

        # Plot loss curve every 40 epochs (or always update final plot)
        if (epoch + 1) % 40 == 0 or epoch == config.num_epochs - 1:
            plot_loss_curve(train_losses, config, epoch)

    # Final loss curve
    plot_loss_curve(train_losses, config, config.num_epochs - 1)
    print("Fine-tuning as transfer learning completed.")


if __name__ == "__main__":
    main()