# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:37:16 2025

@author: uqpfan
"""

# config.py

import argparse
import torch
def get_config():
    parser = argparse.ArgumentParser(description="fine-tuned DDPM Trajectory Training Configuration")

    # --- basic environment ---
    parser.add_argument('--run_name', type=str, default="ddpm_trajectory_finetune_real", help="Name of the experiment run")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    # ---input data path ---
    parser.add_argument("--seq_length", type=int, default=301,help="Length of each trajectory sequence (T)")
    parser.add_argument('--data_path', type=str, default="DA_Empri.npy", help="Path to the real trajectory data (.npy)")
    parser.add_argument('--pretrained_model_path', type=str, default="../QWEN_DDPMV1/checkpoints/ddpm_trajectory_run_1/model_epoch_199.pth", help="Path to pretrained model checkpoint")
    parser.add_argument("--train_split", type=float, default=0.90,
                        help="Proportion of data for training (between 0 and 1)") #not useful actually.
    # --- model para ---
    parser.add_argument("--trajectory_dim", type=int, default=1,
                        help="Dimension of the trajectory data (e.g., 1 for speed)")
    
    parser.add_argument("--time_embedding_dim", type=int, default=128,
                        help="Dimension of the sinusoidal time embedding")
    
    parser.add_argument("--unet_features", type=int, nargs='+', default=[64, 128, 256, 512],
                        help="Feature sizes for the U-Net layers")

    # --- diffusion parameters ---
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument('--beta_start', type=float, default=1e-4, help="Start value of beta schedule")
    parser.add_argument('--beta_end', type=float, default=0.02, help="End value of beta schedule")

    # --- training parameters ---
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of fine-tuning epochs")
    parser.add_argument('--batch_size', type=int, default = 32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=4e-5, help="Learning rate for fine-tuning (smaller than pretrain!)")
    parser.add_argument('--log_interval', type=int, default=20, help="Log loss every N batches")
    parser.add_argument('--save_ckpt_interval', type=int, default=50, help="Save checkpoint every N epochs")

    # --- save path设置 ---
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--results_dir', type=str, default="results", help="Directory to save results and plots")

    # --- Inference Configuration ---
    parser.add_argument("--num_samples_to_generate", type=int, default=10,
                        help="Number of new trajectories to generate during inference")
    #！！！！！！！！！！this is where your model .pth file is stored
    parser.add_argument("--ckpt_path_for_inference", type=str,
                        default="checkpoints/ddpm_trajectory_finetune_real/model_epoch_199.pth", #to be added 
                        help="Path to the trained model checkpoint for sampling")


    args = parser.parse_args()
    return args

# Example of how to use it in a script:
if __name__ == "__main__":
    
    config = get_config()
    print(config.data_path)
    print(config.learning_rate)
