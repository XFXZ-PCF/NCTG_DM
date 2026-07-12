# config.py

"""
Configuration file for the DDPM Trajectory project using argparse.
Defines all hyperparameters and paths, allowing command-line overrides.
"""

import argparse
import torch

def get_config():
    """
    Parses command line arguments and returns a configuration object.
    Default values are set here.
    """
    parser = argparse.ArgumentParser(description="DDPM for Vehicle Trajectory Generation")

    # --- Data Configuration ---
    parser.add_argument("--data_path", type=str, default="DFA_syndata_nosie.npy",
                        help="Path to the preprocessed .npy file containing speed sequences [N, T]") #changed
    
    parser.add_argument("--seq_length", type=int, default=301,
                        help="Length of each trajectory sequence (T)")
    
    parser.add_argument("--train_split", type=float, default=0.90,
                        help="Proportion of data for training (between 0 and 1)")

    # --- Model Configuration ---
    parser.add_argument("--trajectory_dim", type=int, default=1,
                        help="Dimension of the trajectory data (e.g., 1 for speed)")
    
    parser.add_argument("--time_embedding_dim", type=int, default=128,
                        help="Dimension of the sinusoidal time embedding")
    
    parser.add_argument("--unet_features", type=int, nargs='+', default=[64, 128, 256, 512],
                        help="Feature sizes for the U-Net layers")

    # --- Diffusion Configuration ---
    parser.add_argument("--diffusion_timesteps", type=int, default=1000,
                        help="Total number of diffusion timesteps (T)")
    
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Starting beta value for the noise schedule")
    
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="Ending beta value for the noise schedule")

    # --- Training Configuration ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on ('cuda' or 'cpu')")
    
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for the optimizer")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log training stats every N batches")
    
    parser.add_argument("--save_ckpt_interval", type=int, default=5,
                        help="Save model checkpoint every N epochs")
    
    parser.add_argument("--run_name", type=str, default="ddpm_trajectory_run_1",
                        help="Identifier for this training run")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save generated samples and plots")

    # --- Inference Configuration ---
    parser.add_argument("--num_samples_to_generate", type=int, default=10,
                        help="Number of new trajectories to generate during inference")
    
    parser.add_argument("--ckpt_path_for_inference", type=str,
                        default="checkpoints/ddpm_trajectory_run_1/model_epoch_199.pth",
                        help="Path to the trained model checkpoint for sampling")

    args = parser.parse_args()
    return args

# Example of how to use it in a script:
if __name__ == "__main__":
    
    config = get_config()
    print(config.data_path)
    print(config.learning_rate)

