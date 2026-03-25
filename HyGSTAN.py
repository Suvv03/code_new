import torch
from torch import nn
from module import GSAM, GSTAM, SSFM


# =============================================================================
# HyGSTAN 主网络类
# 功能：实现基于超图和注意力机制的双时相高光谱图像变化检测
# 输入：两个时相的高光谱图像块 (x1, x2)
# 输出：变化检测的分类结果 (0-未变化, 1-变化)
# =============================================================================
class hygstan(nn.Module):
    # -------------------------------------------------------------------------
    # 初始化函数
    # 输入参数：
    #   - num_patches: 波段数量 (int)，即高光谱图像的光谱维度
    #   - image_size: 图像块大小 (int)，如 5 表示 5x5 的空间邻域
    #   - p: SSFM 模块参数 (int)，控制光谱融合的数量，默认13
    #   - d: 注意力机制维度 (int)，GSAM和GSTAM的隐藏维度，默认64
    # 输出：无（初始化网络层）
    # -------------------------------------------------------------------------
    def __init__(self, num_patches, image_size, p=13, d=64):
        super().__init__()
        self.p = p  # 光谱融合参数，用于SSFM模块
        self.d = d  # 注意力维度
        self.num_patches = num_patches  # 波段数
        # GSAM模块: 图光谱注意力模块，用于学习特征表示
        # 输入: (batch_size, num_patches, image_size^2 - p)
        # 输出: (batch_size, num_patches, image_size^2 - p), 注意力图Z
        self.gsam = GSAM((image_size ** 2)-p, self.num_patches, d=d)
        # GSTAM模块: 图光谱时序注意力模块，用于融合双时相信息
        # 输入: 特征张量x和来自另一个时相的注意力图Z2
        # 输出: 融合后的特征张量
        self.gstam = GSTAM(self.num_patches, d=d)
        # 全连接分类层
        # 输入: 展平后的特征 (batch_size, num_patches * image_size^2)
        # 输出: 2维分类结果 (batch_size, 2) - 未变化/变化的概率
        self.fc = nn.Sequential(
            nn.BatchNorm1d((num_patches) * (image_size ** 2)),  # 批归一化
            nn.LeakyReLU(inplace=True),  # 激活函数
            nn.Linear((num_patches) * (image_size ** 2), 2, bias=True))  # 线性分类
        # Softmax层：将输出转换为概率分布
        # 输入: 2维 logits (batch_size, 2)
        # 输出: 概率分布 (batch_size, 2)，每行和为1
        self.softmax = nn.Softmax(dim=-1)


    # -------------------------------------------------------------------------
    # 前向传播函数
    # 输入参数：
    #   - x1: 时相1的图像数据 (torch.Tensor)，形状 (batch_size, num_patches, image_size^2)
    #   - x2: 时相2的图像数据 (torch.Tensor)，形状 (batch_size, num_patches, image_size^2)
    # 输出：
    #   - out: 分类概率 (torch.Tensor)，形状 (batch_size, 2)
    #          第0列：未变化概率，第1列：变化概率
    # -------------------------------------------------------------------------
    def forward(self, x1, x2):
        # 调整维度顺序: (batch, seq_len, features) -> (batch, features, seq_len)
        # 便于后续处理光谱维度
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        # SSFM：光谱-空间特征融合模块
        # 生成融合函数和恢复函数
        # 输入: x1, x2; 输出: 融合函数 ssfm_c, 恢复函数 ssfm_r
        ssfm_c1, ssfm_r1 = SSFM(x1, p=self.p)
        ssfm_c2, ssfm_r2 = SSFM(x2, p=self.p)
        # 对两个时相的数据进行光谱融合
        # 输出形状: (batch_size, num_patches - p, image_size^2)
        x1 = ssfm_c1(x1)
        x2 = ssfm_c2(x2)
        # GSAM处理：提取特征并生成注意力图
        # x12, x22: 处理后的特征 (batch, num_patches-p, image_size^2-p)
        # z1, z2: 注意力图，用于时序交互
        x12, z1 = self.gsam(x1)
        x22, z2 = self.gsam(x2)
        # GSTAM处理：时序注意力融合
        # x12使用z2（时相2的注意力图），实现交叉时相注意力
        # x22使用z1（时相1的注意力图），实现双向信息交换
        src12 = self.gstam(x12, z2)
        src22 = self.gstam(x22, z1)
        # SSFM恢复：将融合后的特征恢复原始维度
        x11 = ssfm_r1(src12)
        x22 = ssfm_r2(src22)
        # 融合双时相特征并分类
        # 步骤1: 转置回原始维度 (batch, seq_len, features)
        # 步骤2: 在波段维度展平 (batch, num_patches * image_size^2)
        # 步骤3: 通过全连接层得到2维输出
        # 步骤4: Softmax转换为概率
        out = self.softmax(self.fc(torch.flatten((x11 + x22).transpose(1, 2), 1, 2)))
        return out
