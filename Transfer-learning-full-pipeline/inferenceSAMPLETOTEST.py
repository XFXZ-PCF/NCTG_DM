# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:26:24 2025

@author: uqpfan
"""
"""
Main inference script for generating 
MORE trajectory
"""
# In[1]
import torch
import os
# Import project modules
from model import UNet1D
from utils import DiffusionUtils, save_and_visualize_samples
# Import config
from config import get_config
import numpy as np
number_sampleGENERATE = 500 #used to be 50 and increase to 500
# In[2]
def load_model_for_inference(model, checkpoint_path, device):
    """Loads a trained model checkpoint."""
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained until epoch {checkpoint['epoch']}).")
    return model

# --- 1. Parse Configuration ---
config = get_config()
print(f"Starting inference run for: {config.run_name}")
print(f"Using device: {config.device}")

# --- 2. Model and Diffusion Setup ---
print("Initializing model and diffusion utilities with fine-tuned model\n...")
model = UNet1D(config).to(config.device) # Pass config to model
diffusion = DiffusionUtils(config) # Pass config to diffusion
print("Model and diffusion utilities initialized.")

# --- 3. Load Trained Model ---
if not os.path.exists(config.ckpt_path_for_inference):
    raise FileNotFoundError(f"Checkpoint file not found at {config.ckpt_path_for_inference}. Please train the model first.")
model = load_model_for_inference(model, config.ckpt_path_for_inference, config.device)
model.eval() # Set model to evaluation mode for sampling

# --- 4. Sampling ---
print(f"Generating {number_sampleGENERATE} new trajectories...")
try:
    # Use the sample method from DiffusionUtils
    generated_samples = diffusion.sample(
        model = model,
        config=config, # Pass config
        n = number_sampleGENERATE                         #config.num_samples_to_generate
    ) # Output shape: [N, C, T]
    print(f"Sampling completed. Generated samples shape: {generated_samples.shape}")

except Exception as e:
    print(f"An error occurred during sampling or saving: {e}")
    raise
    
# In[3]
#SAVE THE Generate result;
generated_samples_array = generated_samples.numpy().reshape([-1,301])

np.save('{}samplegenerate.npy'.format(number_sampleGENERATE), generated_samples_array)
# In[4]
save_and_visualize_samples(sampled_trajectories_tensor = generated_samples,
                           config = config,
                           ckpt_epoch = 199, 
                           num_to_plot=10)
"""
Saves generated samples to disk as a .npy file and plots a few of them.

Args:
    sampled_trajectories_tensor (torch.Tensor): Tensor of generated samples. Shape: [N, C, T].
    config (argparse.Namespace): Configuration object containing paths and run name.
    ckpt_epoch (int): Epoch of the model used for sampling.
    num_to_plot (int): Number of samples to plot and save as a PNG.
"""