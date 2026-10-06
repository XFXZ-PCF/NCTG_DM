# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 21:43:20 2025

@author: uqpfan
"""
#Support realtraj.npy
# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TrajectoryDataset(Dataset):
    def __init__(self, config, train=True):
        """
        Load trajectory data from .npy file.
        For transfer learning, we use real data for both 'train' and 'val' since we only have 500 samples.
        self.train = True # 标记当前是训练集还是验证集, used to be train #not used.
        self.train_split = config.train_split # 从配置中获取训练集所占比例
        """
        self.data_path = config.data_path
        self.seq_length = config.seq_length


        print(f"Loading data from {self.data_path}...")
        raw_data = np.load(self.data_path)  # Shape: [N, T] or [N, 1, T] or [N, T, 1]
        #to trucate out\drop the first two columns as indexes!!!!!!!!!!!!!!
        raw_data = raw_data[:,2:]

        # Ensure shape is [N, T] where three dimnsion are imported 
        if len(raw_data.shape) == 3:
            if raw_data.shape[1] == 1:
                raw_data = raw_data.squeeze(1)  # [N, 1, T] -> [N, T]
            elif raw_data.shape[2] == 1:
                raw_data = raw_data.squeeze(2)  # [N, T, 1] -> [N, T]
            else:
                raise ValueError(f"Unexpected data shape: {raw_data.shape}")

        # Ensure we have at least seq_length timesteps
        if raw_data.shape[1] < self.seq_length:
            raise ValueError(f"Data timesteps ({raw_data.shape[1]}) < config.seq_length ({self.seq_length})")
        elif raw_data.shape[1] > self.seq_length:
            raw_data = raw_data[:, :self.seq_length]  # Truncate if longer
            print(f"Truncated data to {self.seq_length} timesteps.")

        self.data = torch.from_numpy(raw_data).float()  # Shape: [N, T]
        self.data = self.data.unsqueeze(1)  # Add channel dim: [N, 1, T] for model input

        print(f"Dataset loaded. Shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Fetches a single sample.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            torch.Tensor: A single trajectory sequence. Shape: [C, T] where C is the channel dim (1 for speed).
        """
        # Get the sequence data
        sequence = self.data[index] # Shape: [T] # 根据索引获取一个序列样本，形状为 [T]
        # Add channel dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0) # Shape: [1, T] # 将 NumPy 数组转换为 FloatTensor 并增加一个通道维度，形状变为 [1, T]
        return sequence_tensor # Shape: [C, T] # 返回形状为 [通道数C, 序列长度T] 的张量


def get_dataloader(config, train=True):
    """
    Creates a DataLoader for the TrajectoryDataset using values from the config object.

    Args:
        config (argparse.Namespace): Configuration object containing data and training parameters.
        train (bool): Whether to create a train or validation loader.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    dataset = TrajectoryDataset(config, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size = config.batch_size,
        shuffle=True,  # Always shuffle for training
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    return dataloader