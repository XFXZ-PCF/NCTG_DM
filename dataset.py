# dataset.py

"""
Defines the PyTorch Dataset for loading vehicle trajectory data.
Updated to work with argparse config.
"""

import torch
from torch.utils.data import Dataset, DataLoader # 导入 PyTorch 的 Dataset 和 DataLoader 类
import numpy as np # 导入 NumPy 库用于处理数组

# Note: We no longer import a global config instance here.
# Parameters will be passed explicitly or accessed via the config object passed from the main script.

class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed vehicle trajectory data from a .npy file.
    """
    def __init__(self, config, train=True):
        """
        Initializes the dataset using values from the config object.

        Args:
            config (argparse.Namespace): Configuration object containing data parameters.
            train (bool): If True, loads the training split; otherwise, loads the validation split.
        """
        self.data_path = config.data_path # 从配置中获取数据文件路径
        self.seq_length = config.seq_length # 从配置中获取序列长度
        self.train = True # 标记当前是训练集还是验证集, used to be train
        self.train_split = config.train_split # 从配置中获取训练集所占比例

        # Load data from .npy file
        raw_data = np.load(self.data_path) # Expected shape: [N, T] # 从 .npy 文件加载数据，预期形状为 [样本数N, 序列长度T]
        print(f"Raw data loaded with shape: {raw_data.shape}") # 打印加载的数据形状

        # Validate sequence length
        if raw_data.shape[1] != self.seq_length: # 检查加载数据的序列长度是否与配置一致
            raise ValueError(f"Data sequence length {raw_data.shape[1]} does not match expected {self.seq_length}") # 如果不一致则抛出错误

        # Split data into train/val
        split_idx = int(len(raw_data) * self.train_split) # 计算训练集和验证集的分割索引
        if self.train: # 如果是训练集
            self.data = raw_data[:split_idx] # 取前 split_idx 个样本作为训练数据
        else: # 如果是验证集
            self.data = raw_data[split_idx:] # 取 split_idx 之后的样本作为验证数据
        
        print(f"{'Training' if self.train else 'Validation'} dataset created with {len(self.data)} samples.") # 打印创建的数据集信息

    def __len__(self):
        """Returns the total number of samples.""" # 返回数据集中的总样本数
        return len(self.data) # 返回 self.data 的长度

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

# --- Factory Function for DataLoaders ---
def get_dataloader(config, train=True):
    """
    Creates a DataLoader for the TrajectoryDataset using values from the config object.

    Args:
        config (argparse.Namespace): Configuration object containing data and training parameters.
        train (bool): Whether to create a train or validation loader.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    dataset = TrajectoryDataset(config=config, train=train) # 使用配置和 train 标志创建 TrajectoryDataset 实例
    shuffle = train # Shuffle only for training # 仅在训练时打乱数据
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle) # 使用指定的 dataset、batch_size 和 shuffle 创建 DataLoader
    return dataloader # 返回配置好的 DataLoader 实例




