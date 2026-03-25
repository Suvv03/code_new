from typing import Callable, Tuple
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

# =============================================================================
# SSFM (Spectral-Spatial Feature Merging) 光谱-空间特征融合模块
# 功能：通过光谱相似性进行自适应特征融合，减少光谱冗余
# =============================================================================
# 输入参数：
#   - x: 输入特征张量 (torch.Tensor)，形状 (batch_size, band, features)
#   - p: 融合参数 (int)，控制要融合的光谱波段数量
# 输出：
#   - ssfm_c: 压缩函数 (Callable)，用于融合光谱特征
#   - ssfm_r: 恢复函数 (Callable)，用于将融合后的特征恢复原始维度
# =============================================================================
def SSFM(
        x: torch.Tensor,
        p: int,
) -> Tuple[Callable, Callable]:
    # 获取输入的光谱波段数
    band = x.shape[1]
    # 限制 p 不超过波段数的一半，避免过度融合
    p = min(p, band // 2)
    # 在 no_grad 模式下计算光谱相似性（不计算梯度，节省内存）
    with torch.no_grad():
        # L2归一化：对每个特征向量进行归一化，便于计算余弦相似度
        x = x / x.norm(dim=-1, keepdim=True)
        # 将光谱波段分为偶数索引(x_1)和奇数索引(x_2)两组
        # 形状: (batch, band/2, features)
        x_1, x_2 = x[..., ::2, :], x[..., 1::2, :]
        # 计算两组之间的相似度矩阵
        # sim[b, i, j] 表示第b个样本中x_1的第i个波段与x_2的第j个波段的相似度
        sim = x_1 @ x_2.transpose(-1, -2)
        # 为每个x_1波段找到最相似的x_2波段索引
        _, top_indices = sim.max(dim=-1)
        # 按相似度降序排列，得到排序后的索引
        so_id = top_indices.argsort(dim=-1, descending=True)[..., None]
        # sk_id: 相似度较低的波段索引（保留这部分）
        sk_id = so_id[..., p:, :]
        # se_id: 相似度较高的波段索引（这部分将被融合）
        se_id = so_id[..., :p, :]
        # m: 映射矩阵，记录se_id对应的x_2中的位置
        m = top_indices[..., None].gather(dim=-2, index=se_id)

    # -------------------------------------------------------------------------
    # ssfm_c: 压缩/融合函数
    # 功能：将相似的光谱波段融合，减少冗余
    # 输入：
    #   - feature_data: 输入特征 (batch, band, features)
    #   - operation: 融合操作，默认为"mean"（平均融合）
    # 输出：
    #   - 融合后的特征 (batch, band-p, features)
    # -------------------------------------------------------------------------
    def ssfm_c(feature_data: torch.Tensor, operation="mean") -> torch.Tensor:
        # 同样分为偶数和奇数索引两组
        part_a, part_b = feature_data[..., ::2, :], feature_data[..., 1::2, :]
        N, _, C = part_a.shape  # N=batch_size, C=特征维度
        # sk: 保留的、相似度较低的光谱波段
        sk = part_a.gather(dim=-2, index=sk_id.expand(N, -1, C))
        # se_a: 需要融合的高相似度波段
        se_a = part_a.gather(dim=-2, index=se_id.expand(N, -1, C))
        # com_b: 将se_a融合到对应的part_b位置
        # scatter_reduce: 按照m指定的位置，将se_a的值散射到part_b并执行归约操作
        com_b = part_b.scatter_reduce(-2, m.expand(N, -1, C), se_a, reduce=operation)
        # 合并：保留部分 + 融合后的部分
        merged_parts = [sk, com_b]
        return torch.cat(merged_parts, dim=1)

    # -------------------------------------------------------------------------
    # ssfm_r: 恢复函数
    # 功能：将融合后的特征恢复为原始光谱维度（用于最终输出）
    # 输入：
    #   - data: 融合后的特征 (batch, band-p, features)
    # 输出：
    #   - 恢复后的特征 (batch, band, features)，与输入SSFM的x同形状
    # -------------------------------------------------------------------------
    def ssfm_r(data: torch.Tensor) -> torch.Tensor:
        sk_len = sk_id.shape[1]  # 保留的波段数量
        # 将输入分为保留部分(sk)和融合部分(com)
        sk, com = data.split([sk_len, data.shape[1] - sk_len], dim=-2)
        N, _, C = sk.shape
        # res_a: 从com中提取融合前的se_a部分
        res_a = com.gather(dim=-2, index=m.expand(-1, -1, C))
        # 初始化输出张量，形状与原始x相同
        output = torch.zeros_like(x).scatter_(-2, (2 * sk_id).expand(-1, -1, C), sk)
        # 将se_a放回偶数索引位置
        output = output.scatter_(-2, (2 * se_id).expand(-1, -1, C), res_a)
        # 将com放回奇数索引位置
        output[..., 1::2, :] = com
        return output

    return ssfm_c, ssfm_r



# =============================================================================
# GSAM (Graph Spectral Attention Module) 图光谱注意力模块
# 功能：使用注意力机制学习高光谱图像的特征表示，生成注意力图
# =============================================================================
# 输入参数：
#   - n: 空间位置数量 (int)，即 image_size^2 - p
#   - num_patches: 光谱波段数量 (int)
#   - d: 注意力维度 (int)，默认64
#   - eps: LayerNorm的epsilon值 (float)，默认0
# 输出：无（初始化时），forward返回特征和注意力图
# =============================================================================
class GSAM(nn.Module):
    def __init__(self, n, num_patches, d=64, eps=0):
        super(GSAM, self).__init__()
        self.n = n  # 空间位置数
        self.num_patches = num_patches  # 光谱波段数
        self.d = d  # 注意力维度
        # gamma和beta: 可学习的缩放和偏置参数，用于调制注意力输出
        # 形状: (2, d)，2对应query和key
        self.gamma = nn.Parameter(torch.rand((2, d)))
        self.beta = nn.Parameter(torch.rand((2, d)))
        # b: 可学习的偏置矩阵，用于注意力计算
        # 形状: (n, n)，n为空间位置数
        self.b = nn.Parameter(torch.rand((n, n)))
        # w1: 线性变换层，将输入映射到3个部分：p, v, o
        # 输入维度: num_patches，输出维度: 2*num_patches + d
        self.w1 = nn.Linear(num_patches, 2 * num_patches + d)
        # w2: 输出线性层，将注意力结果映射回原始维度
        self.w2 = nn.Linear(num_patches, num_patches)
        # LayerNorm: 层归一化，稳定训练
        self.LayerNorm = nn.LayerNorm(num_patches, eps=eps)
        # g: ReLU激活函数
        self.g = nn.ReLU()
        # drop: Dropout层，防止过拟合，丢弃率0.5
        self.drop = nn.Dropout(0.5)

    # -------------------------------------------------------------------------
    # forward: 前向传播
    # 输入：
    #   - x: 输入特征 (batch_size, num_patches, n)
    # 输出：
    #   - x: 处理后的特征 (batch_size, num_patches, n)
    #   - Z: 注意力图 (batch_size, n, n)，表示空间位置间的注意力关系
    # -------------------------------------------------------------------------
    def forward(self, x):
        x0, x = x, self.LayerNorm(x)  # 保留原始输入，对x进行归一化
        # w1: 将x从 (batch, num_patches, n) 映射到高维空间
        # split: 分割为 p, v, o 三部分
        # p: 投影特征 (batch, num_patches, n)
        # v: 值特征 (batch, num_patches, n)
        # o: 注意力调制向量 (batch, num_patches, d)
        p, v, o = torch.split(self.g(self.w1(x)), [self.num_patches, self.num_patches, self.d], dim=-1)
        # 调制o: 通过gamma缩放和beta偏置生成query和key
        # 输出形状: (batch, num_patches, 2, d) -> unbind为两个 (batch, num_patches, d)
        o = torch.einsum("...r, hr->...hr", o, self.gamma) + self.beta
        q, k = torch.unbind(o, dim=-2)
        # 计算注意力矩阵: q @ k^T / sqrt(d)
        # qk: (batch, n, n)，表示空间位置间的相似度
        qk = torch.einsum("bnd,bmd->bnm", q, self.drop(k))
        # Z: 最终注意力图，使用ReLU平方激活和可学习偏置b
        # Z[i,j] 表示位置i对位置j的注意力权重
        Z = torch.square(F.relu(qk / np.sqrt(self.d) + self.b))/self.n
        # 注意力加权：对v进行加权求和
        # x = p * (Z @ v)，元素乘法结合矩阵乘法
        x = p * torch.einsum("bnm, bme->bne", Z, v)
        # 输出变换: w2 + 残差连接 + dropout
        x = self.w2(x) + self.drop(x0)
        return x, Z


# =============================================================================
# GSTAM (Graph Spectral Temporal Attention Module) 图光谱时序注意力模块
# 功能：利用来自另一个时相的注意力图进行跨时相特征融合
# =============================================================================
# 输入参数：
#   - num_patches: 光谱波段数量 (int)
#   - d: 注意力维度 (int)，默认64
#   - eps: LayerNorm的epsilon值 (float)，默认0
# 输出：无（初始化时），forward返回融合后的特征
# =============================================================================
class GSTAM(nn.Module):
    def __init__(self, num_patches, d=64, eps=0,):
        super(GSTAM, self).__init__()
        self.num_patches = num_patches  # 光谱波段数
        self.d = d  # 注意力维度
        # w1: 线性变换层，类似GSAM
        self.w1 = nn.Linear(num_patches, 2 * num_patches + d)
        # w2: 输出线性层
        self.w2 = nn.Linear(num_patches, num_patches)
        # LayerNorm: 层归一化
        self.LayerNorm = nn.LayerNorm(num_patches, eps=eps)
        # g: ReLU激活函数
        self.g = nn.ReLU()
        # drop: Dropout层
        self.drop = nn.Dropout(0.5)


    # -------------------------------------------------------------------------
    # forward: 前向传播
    # 输入：
    #   - x: 当前时相的特征 (batch_size, num_patches, n)
    #   - Z2: 来自另一个时相的注意力图 (batch_size, n, n)
    #         用于指导当前时相的特征聚合
    # 输出：
    #   - x: 融合后的特征 (batch_size, num_patches, n)
    # -------------------------------------------------------------------------
    def forward(self, x, Z2):
        x0, x = x, self.LayerNorm(x)  # 保留原始输入，归一化
        # w1变换后分割为 p, v 两部分（不使用o，因为注意力图Z2来自外部）
        # p: 投影特征 (batch, num_patches, n)
        # v: 值特征 (batch, num_patches, n)
        p, v, _ = torch.split(self.g(self.w1(x)), [self.num_patches, self.num_patches, self.d], dim=-1)
        # 使用外部传入的Z2进行注意力加权
        # 实现跨时相信息交换：Z2包含了另一个时相的空间关系
        x = p * torch.einsum("bnm, bme->bne", self.drop(Z2), v)
        # 输出变换 + 残差连接
        x = self.w2(x) + self.drop(x0)
        return x
