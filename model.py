# model.py

"""
Defines the 1D U-Net architecture used as the denoising network in the DDPM model
for vehicle trajectory data. Updated to work with argparse config.
"""

import torch
import torch.nn as nn

# Note: We no longer import a global config instance here.
# Parameters will be passed explicitly or accessed via the config object passed from the main script.

class UNet1D(nn.Module):
    """
    A 1D U-Net for processing 1D time-series data like vehicle speed trajectories.
    
    The network takes a noisy input sequence and a time embedding and predicts the noise.
    Input shape: [B, C_in, T] where B is batch size, C_in is input channels (e.g., 2 for data+time), T is sequence length.
    Output shape: [B, C_out, T] where C_out is the number of output channels (e.g., 1 for predicting noise on speed).
    """
    def __init__(self, config):
        """
        Initializes the 1D U-Net using values from the config object.

        Args:
            config (argparse.Namespace): Configuration object containing model parameters.
        """
        super(UNet1D, self).__init__() # 调用父类 nn.Module 的初始化方法
        
        # Use config values for model definition
        self.in_channels = config.trajectory_dim + 1 # Data channel + Time embedding channel # 输入通道数：轨迹数据通道 + 时间嵌入通道
        self.out_channels = config.trajectory_dim # 输出通道数：预测轨迹数据的噪声
        self.features = config.unet_features # U-Net 各层的特征图数量列表

        self.ups = nn.ModuleList() # Decoder layers # 用于存储解码器（上采样路径）层的列表
        self.downs = nn.ModuleList() # Encoder layers # 用于存储编码器（下采样路径）层的列表
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # Downsampling # 下采样层：最大池化，核大小为2，步长为2

        # --- Encoder (Downsampling Path) ---
        in_channels = self.in_channels # 初始化编码器的输入通道数
        for feature in self.features: # 遍历 U-Net 特征列表中的每个特征数
            self.downs.append(DoubleConv1D(in_channels, feature)) # 添加一个 DoubleConv1D 模块到编码器列表
            in_channels = feature # 更新下一层的输入通道数为当前层的输出通道数

        # --- Bottleneck ---
        bottleneck_features = self.features[-1] * 2 # 瓶颈层的特征数通常是编码器最后一层的两倍
        self.bottleneck = DoubleConv1D(self.features[-1], bottleneck_features) # 定义瓶颈层

        # --- Decoder (Upsampling Path) ---
        # Note: features are iterated in reverse order # 注意：特征列表是反向迭代的
        for feature in reversed(self.features): # 反向遍历特征列表
            # Upsampling layer (transposed convolution) # 上采样层（转置卷积）
            self.ups.append(
                nn.ConvTranspose1d(
                    feature * 2, feature, kernel_size=2, stride=2, # 输入通道是 feature*2 因为会与跳跃连接拼接
                )
            )
            # Double convolution after upsampling and skip connection concatenation # 上采样和跳跃连接拼接后的双重卷积
            self.ups.append(DoubleConv1D(feature * 2, feature)) # 输入通道是 feature*2 因为拼接了来自跳跃连接的特征

        # --- Final Output Layer ---
        self.final_conv = nn.Conv1d(self.features[0], self.out_channels, kernel_size=1) # 最终输出层：1x1 卷积，将特征图数量减少到输出通道数

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, T].
        """
        skip_connections = [] # 初始化列表以存储跳跃连接（编码器各层的输出）

        # --- Encoder Forward Pass ---
        for down in self.downs: # 遍历编码器中的每一层
            x = down(x) # 通过当前的 DoubleConv1D 层
            skip_connections.append(x) # 保存输出用于跳跃连接
            x = self.pool(x) # 应用下采样

        # --- Bottleneck ---
        x = self.bottleneck(x) # 通过瓶颈层
        # Reverse skip connections for decoder # 反转跳跃连接列表，以便在解码器中从 deepest 开始使用
        skip_connections = skip_connections[::-1]

        # --- Decoder Forward Pass ---
        # Iterate through upsampling and double conv layers in pairs # 成对迭代上采样层和双重卷积层
        for idx in range(0, len(self.ups), 2): # 步长为2，每次处理一对（上采样 + 双重卷积）
            x = self.ups[idx](x) # 执行上采样（转置卷积）
            skip_connection = skip_connections[idx // 2] # 获取对应的跳跃连接

            # --- Dimension Matching ---
            # Ensure spatial dimensions match before concatenation
            # This can happen due to rounding in pooling/striding
            if x.shape[2] != skip_connection.shape[2]: # 检查上采样后的序列长度是否与跳跃连接的匹配
                # Interpolate x to match the spatial dimension of skip_connection # 如果不匹配，对 x 进行插值
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2], mode='linear', align_corners=False # 使用线性插值调整大小
                )

            # Concatenate along the channel dimension # 沿着通道维度拼接
            concat_skip = torch.cat((skip_connection, x), dim=1) # 将跳跃连接和上采样后的特征图拼接
            x = self.ups[idx + 1](concat_skip) # 对拼接后的结果应用双重卷积

        # --- Final Output ---
        return self.final_conv(x) # 通过最终的 1x1 卷积层得到输出


class DoubleConv1D(nn.Module):
    """
    A block consisting of two Conv1D -> BatchNorm1D -> ReLU layers.
    This is a standard building block for U-Nets.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DoubleConv1D block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.double_conv = nn.Sequential( # 定义一个按顺序执行的模块序列
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # 第一个 1D 卷积层，3x3 卷积，填充1，无偏置
            nn.BatchNorm1d(out_channels), # 第一个批归一化层
            nn.ReLU(inplace=True), # 第一个 ReLU 激活函数，inplace=True 节省内存
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 第二个 1D 卷积层，3x3 卷积，填充1，无偏置
            nn.BatchNorm1d(out_channels), # 第二个批归一化层
            nn.ReLU(inplace=True) # 第二个 ReLU 激活函数
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv1D block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x) # 按顺序执行序列中的所有层




