# inference.py

"""
Main inference script for generating new vehicle trajectories using a trained DDPM model.
Updated to work with argparse config.
"""

import torch
import os
# Import project modules
from model import UNet1D
from utils import DiffusionUtils, save_and_visualize_samples
# Import config
from config import get_config

def load_model_for_inference(model, checkpoint_path, device):
    """Loads a trained model checkpoint."""
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained until epoch {checkpoint['epoch']}).")
    return model

def main():
    """Main inference function."""
    # --- 1. Parse Configuration ---
    config = get_config()
    print(f"Starting inference run for: {config.run_name}")
    print(f"Using device: {config.device}")

    # --- 2. Model and Diffusion Setup ---
    print("Initializing model and diffusion utilities...")
    model = UNet1D(config).to(config.device) # Pass config to model
    diffusion = DiffusionUtils(config) # Pass config to diffusion
    print("Model and diffusion utilities initialized.")

    # --- 3. Load Trained Model ---
    if not os.path.exists(config.ckpt_path_for_inference):
        raise FileNotFoundError(f"Checkpoint file not found at {config.ckpt_path_for_inference}. Please train the model first.")
    model = load_model_for_inference(model, config.ckpt_path_for_inference, config.device)
    model.eval() # Set model to evaluation mode for sampling

    # --- 4. Sampling ---
    print(f"Generating {config.num_samples_to_generate} new trajectories...")
    try:
        # Use the sample method from DiffusionUtils
        generated_samples = diffusion.sample(
            model=model,
            config=config, # Pass config
            n=config.num_samples_to_generate
        ) # Output shape: [N, C, T]
        print(f"Sampling completed. Generated samples shape: {generated_samples.shape}")

        # --- 5. Save and Visualize ---
        # Extract epoch number from checkpoint path for saving results
        ckpt_filename = os.path.basename(config.ckpt_path_for_inference)
        ckpt_epoch = int(ckpt_filename.split('_')[-1].split('.')[0]) # Assumes format "model_epoch_X.pth"

        save_and_visualize_samples(
            sampled_trajectories_tensor=generated_samples,
            config=config, # Pass config
            ckpt_epoch=ckpt_epoch,
            num_to_plot=5 # Plot the first 5 samples
        )
        print("Inference completed successfully.")

    except Exception as e:
        print(f"An error occurred during sampling or saving: {e}")
        raise

if __name__ == "__main__":
    main()

