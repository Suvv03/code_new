# =============================================================================
# main.py - HyGSTAN 高光谱变化检测主程序
# 功能：加载数据集、训练模型、评估性能并保存结果
# 该程序实现了一个完整的变化检测流程，包含数据预处理、模型训练和测试
# =============================================================================

import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from HyGSTAN import hygstan
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch.nn.functional as F
import time
import os
import json
from PIL import Image

# =============================================================================
# 命令行参数配置
# 功能：定义程序运行时的超参数和配置选项
# 使用方式：python main.py --gpu_id 0 --batch_size 32 --epoches 100
# =============================================================================
parser = argparse.ArgumentParser("HSI")
# gpu_id: GPU设备编号，默认使用1号GPU
parser.add_argument('--gpu_id', default='1', help='gpu id')
# seed: 随机种子，保证实验可复现
parser.add_argument('--seed', type=int, default=0, help='number of seed')
# batch_size: 训练批次大小，影响显存使用和训练速度
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
# test_freq: 测试频率，每多少轮(epoch)评估一次模型
parser.add_argument('--test_freq', type=int, default=100, help='number of evaluation')
# patches: 空间邻域大小，如5表示使用5x5的空间窗口
parser.add_argument('--patches', type=int, default=5, help='number of patches')
# band_patches: 光谱邻域大小，用于邻域波段提取
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
# epoches: 训练轮数
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
# learning_rate: 学习率，控制参数更新步长
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
# gamma: 学习率衰减系数
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
# weight_decay: 权重衰减（L2正则化系数），防止过拟合
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
# train_number: 训练样本比例，如0.01表示使用1%的数据作为训练集
parser.add_argument('--train_number', type=float, default=0.01, help='train_number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# =============================================================================
# 函数: chooose_train_and_test_point
# 功能：从标签数据中分离训练集和测试集的样本位置
# 输入：
#   - train_data: 训练集标签图 (height, width)，非零值为类别标签
#   - test_data: 测试集标签图 (height, width)，非零值为类别标签
#   - num_classes: 类别数量 (int)，如2表示变化和未变化两类
# 输出：
#   - total_pos_train: 所有训练样本的坐标 (N_train, 2)，每行为(y, x)坐标
#   - total_pos_test: 所有测试样本的坐标 (N_test, 2)，每行为(y, x)坐标
#   - number_train: 每类训练样本数量列表 [N_class0, N_class1, ...]
#   - number_test: 每类测试样本数量列表 [N_class0, N_class1, ...]
# =============================================================================
def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []  # 存储每类训练样本数
    pos_train = {}   # 存储每类训练样本的位置字典
    number_test = []  # 存储每类测试样本数
    pos_test = {}    # 存储每类测试样本的位置字典
    # 遍历每个类别，收集训练样本位置
    for i in range(num_classes):
        # argwhere: 找到所有等于当前类别标签(i+1)的位置
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])  # 记录数量
        pos_train[i] = each_class  # 保存位置
    # 合并所有训练样本位置
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    # 同样方式处理测试集
    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test


# =============================================================================
# 函数: mirror_hsi
# 功能：对高光谱图像进行镜像填充（边界扩展），处理边缘像素
# 原理：以图像边界为轴进行镜像反射，确保边缘像素也能提取完整patch
# 输入：
#   - height: 图像高度 (int)
#   - width: 图像宽度 (int)
#   - band: 光谱波段数 (int)
#   - input_normalize: 归一化后的高光谱数据 (height, width, band)
#   - patch: 邻域窗口大小 (int)，如5表示5x5
# 输出：
#   - mirror_hsi: 镜像填充后的数据 (height+2*padding, width+2*padding, band)
# =============================================================================
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2  # 计算填充大小，如patch=5时padding=2
    # 初始化填充后的数组
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 将原始数据放入中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 镜像填充：左侧边界
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 镜像填充：右侧边界
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 镜像填充：上侧边界
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 镜像填充：下侧边界
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]
    return mirror_hsi

# =============================================================================
# 函数: gain_neighborhood_pixel
# 功能：从镜像图像中提取指定位置的patch（空间邻域）
# 输入：
#   - mirror_image: 镜像填充后的图像 (H', W', band)
#   - point: 样本坐标数组 (N, 2)，每行为(y, x)坐标
#   - i: 当前样本索引 (int)
#   - patch: 邻域大小 (int)，默认5
# 输出：
#   - temp_image: 提取的patch (patch, patch, band)，包含空间邻域信息
# =============================================================================
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]   # 获取行索引（对应y坐标/垂直位置）
    y = point[i, 1]  # 获取列索引（对应x坐标/水平位置）
    # 从镜像图像中提取patch
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image

# =============================================================================
# 函数: gain_neighborhood_band
# 功能：提取光谱邻域（波段方向的上下文信息）
# 原理：对于每个波段，收集其相邻波段的信息，形成光谱上下文
# 输入：
#   - x_train: 空间patch数据 (N, patch, patch, band)
#   - band: 波段数 (int)
#   - band_patch: 光谱邻域大小 (int)，如3表示左右各1个邻域波段
#   - patch: 空间邻域大小 (int)
# 输出：
#   - x_train_band: 包含光谱邻域的数据 (N, patch*patch*band_patch, band)
# =============================================================================
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2  # 每侧邻域数量
    pp = (patch * patch) // 2  # 中间位置索引
    # 重塑为 (N, patch*patch, band)
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    # 初始化输出数组
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=np.float16)
    # 设置中心区域（当前波段对应的空间信息）
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    # 填充左侧邻域波段（低频/短波方向）
    for i in range(nn):
        if pp > 0:
            # 将高频波段的信息复制到低频位置
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    # 填充右侧邻域波段（高频/长波方向）
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[
                                                                                                        :, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[
                                                                                                        :, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band

# =============================================================================
# 函数: train_and_test_data
# 功能：汇总生成训练集和测试集的完整数据
# 输入：
#   - mirror_image: 镜像填充后的图像
#   - band: 波段数 (int)
#   - train_point: 训练样本坐标 (N_train, 2)
#   - test_point: 测试样本坐标 (N_test, 2)
#   - patch: 空间邻域大小 (int)，默认5
#   - band_patch: 光谱邻域大小 (int)，默认3
# 输出：
#   - x_train_band: 训练数据 (N_train, patch*patch*band_patch, band)
#   - x_test_band: 测试数据 (N_test, patch*patch*band_patch, band)
# =============================================================================
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    # 初始化数据数组
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # 提取训练样本的patches
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    # 提取测试样本的patches
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("**************************************************")
    # 添加光谱邻域信息
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band

# =============================================================================
# 函数: train_and_test_label
# 功能：生成训练集和测试集的标签
# 输入：
#   - number_train: 每类训练样本数量列表 [N_class0, N_class1, ...]
#   - number_test: 每类测试样本数量列表 [N_class0, N_class1, ...]
#   - num_classes: 类别数量 (int)
# 输出：
#   - y_train: 训练标签 (N_train,)，值为0, 1, ..., num_classes-1
#   - y_test: 测试标签 (N_test,)，值为0, 1, ..., num_classes-1
# =============================================================================
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    # 为每个类别生成对应数量的标签
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)  # 类别i的标签为i
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("**************************************************")
    return y_train, y_test

# =============================================================================
# 类: AvgrageMeter
# 功能：用于计算和存储训练/测试过程中的平均指标（如损失、准确率）
# 方法：
#   - reset(): 重置计数器
#   - update(val, n): 更新累积值，val是当前值，n是样本数
# =============================================================================
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0  # 当前平均值
        self.sum = 0  # 累积总和
        self.cnt = 0  # 累积样本数

    def update(self, val, n=1):
        self.sum += val * n  # 加权累加
        self.cnt += n        # 累加样本数
        self.avg = self.sum / self.cnt  # 更新平均值

# =============================================================================
# 函数: accuracy
# 功能：计算分类准确率（Top-K准确率）
# 输入：
#   - output: 模型输出 (batch_size, num_classes)，未归一化的logits
#   - target: 真实标签 (batch_size,)
#   - topk: 计算Top-K准确率，默认(1,)表示只计算Top-1
# 输出：
#   - res: 准确率列表 [top1_acc, top5_acc, ...]
#   - target: 真实标签（返回用于后续分析）
#   - pred.squeeze(): 预测标签（Top-1）
# =============================================================================
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)  # 最大K值
    batch_size = target.size(0)  # 批次大小
    # topk: 获取预测值中最大的maxk个值的索引
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置，便于与目标比较
    # 比较预测值和真实值，生成布尔矩阵
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    # 计算每个K值的准确率
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

# =============================================================================
# 类: Loss_fn
# 功能：自定义损失函数（结合Focal Loss和二次惩罚项）
# 原理：通过(1-Pt)^delta因子降低易分样本的权重，关注难分样本
# 输入参数：
#   - delta: 调制因子系数 (float)，默认1.0，控制难易样本权重差异
#   - lambda_param: 二次惩罚项系数 (float)，默认0.5
#   - num_classes: 类别数量 (int)，默认2
# 方法forward：
#   - 输入：logits (batch, num_classes), labels (batch,)
#   - 输出：平均损失值 (scalar tensor)
# =============================================================================
class Loss_fn(nn.Module):
    def __init__(self, delta=1.0, lambda_param=0.5, num_classes=2):
        super(Loss_fn, self).__init__()
        self.delta = delta
        self.lambda_param = lambda_param
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # Softmax将logits转换为概率
        probabilities = F.softmax(logits, dim=1)
        # 获取真实标签对应的预测概率Pt
        Pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        # 主损失：类似Focal Loss，使用(1-Pt)^delta调制
        loss_primary = -(1 - Pt) ** self.delta * torch.log(Pt)
        # 二次损失：额外的惩罚项，强化对低置信度的惩罚
        loss_secondary = self.lambda_param * (1 - Pt) ** (self.delta + 1)
        # 总损失
        total_loss = loss_primary + loss_secondary
        return total_loss.mean()

# =============================================================================
# 函数: train_epoch
# 功能：执行一个训练轮次（遍历整个训练集）
# 输入：
#   - model: 神经网络模型 (hygstan)
#   - train_loader: 训练数据加载器 (DataLoader)
#   - criterion: 损失函数 (Loss_fn)
#   - optimizer: 优化器 (AdamW)
# 输出：
#   - top1.avg: 平均训练准确率 (float)
#   - objs.avg: 平均训练损失 (float)
#   - tar: 所有训练样本的真实标签 (numpy array)
#   - pre: 所有训练样本的预测标签 (numpy array)
# =============================================================================
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()  # 损失记录器
    top1 = AvgrageMeter()  # 准确率记录器
    tar = np.array([])     # 真实标签
    pre = np.array([])     # 预测标签
    # 遍历每个batch
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(train_loader):
        # 将数据移至GPU
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        batch_pred = model(batch_data_t1, batch_data_t2)
        # 计算损失
        loss = criterion(batch_pred, batch_target)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 计算准确率
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        # 更新统计量
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

# =============================================================================
# 函数: valid_epoch
# 功能：执行一个验证轮次（评估模型在测试集上的性能）
# 输入：与train_epoch相同
# 输出：
#   - tar: 所有验证样本的真实标签
#   - pre: 所有验证样本的预测标签
# =============================================================================
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data_t1, batch_data_t2)
        loss = criterion(batch_pred, batch_target)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

# =============================================================================
# 函数: test_epoch
# 功能：在完整数据集上进行预测（生成变化检测图）
# 输入：
#   - model: 训练好的模型
#   - test_loader: 完整数据加载器
#   - criterion, optimizer: 占位参数（保持接口一致）
# 输出：
#   - pre: 所有像素的预测标签 (numpy array)，值为0或1
# =============================================================================
def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(test_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data_t1, batch_data_t2)
        # 获取Top-1预测结果
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

# 导入评估指标计算函数
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

# =============================================================================
# 函数: output_metric
# 功能：计算完整的分类评估指标
# 输入：
#   - tar: 真实标签 (numpy array)
#   - pre: 预测标签 (numpy array)
# 输出：
#   - OA: 总体准确率 (float)，所有样本中预测正确的比例
#   - F1: F1分数均值 (float)，精确率和召回率的调和平均
#   - Pr: 平均精确率 (float)，查准率
#   - Re: 平均召回率 (float)，查全率
#   - Kappa: Kappa系数 (float)，考虑偶然一致性的评估指标
#   - AA_mean: 平均准确率均值 (float)
#   - AA: 每类准确率数组 (numpy array)
# =============================================================================
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)  # 混淆矩阵
    total_samples = np.sum(matrix)
    # OA (Overall Accuracy): 对角线元素和 / 总样本数
    OA = np.sum(np.diag(matrix)) / total_samples if total_samples > 0 else 0.0
    # 计算每类的精确率、召回率、F1分数
    class_precision = precision_score(tar, pre, average=None, zero_division=0)
    class_recall = recall_score(tar, pre, average=None, zero_division=0)
    class_f1 = f1_score(tar, pre, average=None, zero_division=0)
    # 取均值作为整体指标
    Pr = np.mean(class_precision)
    Re = np.mean(class_recall)
    F1 = np.mean(class_f1)
    # Kappa系数计算
    po = OA  # 观测一致率
    pe = np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total_samples ** 2) if total_samples > 0 else 0.0  # 期望一致率
    Kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    # AA (Average Accuracy): 每类准确率的均值
    row_sums = np.sum(matrix, axis=1)
    valid_classes = row_sums > 0
    AA = np.zeros(len(row_sums))
    if np.any(valid_classes):
        AA[valid_classes] = np.divide(
            np.diag(matrix)[valid_classes],
            row_sums[valid_classes],
            out=np.zeros(np.sum(valid_classes)),
            where=row_sums[valid_classes] > 0
        )
    AA_mean = np.mean(AA) if len(AA) > 0 else 0.0
    return OA, F1, Pr, Re, Kappa, AA_mean, AA

# =============================================================================
# 函数: cal_results
# 功能：从混淆矩阵计算评估指标（备用函数）
# 输入：
#   - matrix: 混淆矩阵 (num_classes, num_classes)
# 输出：与output_metric类似，但直接从混淆矩阵计算
# =============================================================================
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

# =============================================================================
# 函数: save_binary_mask
# 功能：保存二值掩码图像（用于调试）
# 输入：
#   - mask: 二值掩码 (numpy array)，值为0或1
#   - dataset: 数据集名称 (str)
#   - run: 运行次数编号 (int)
#   - name: 掩码名称 (str)，如"tp", "fp", "fn", "tn"
# 输出：无（直接保存为PNG文件）
# =============================================================================
def save_binary_mask(mask, dataset, run, name):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(f"{dataset}_binary_{name}_run{run}.png")

# =============================================================================
# LABEL_CONFIG: 数据集标签配置字典
# 功能：定义不同数据集的标签映射关系
# 变化检测通常分为两类：变化(change)和未变化(no_change)
# 由于不同数据集的原始标签定义不同，需要统一映射
# =============================================================================
LABEL_CONFIG = {
    'hermiston': {'change': 1, 'no_change': 2},
    'farmland': {'change': 1, 'no_change': 2},
    'river': {'change': 1, 'no_change': 2},  # 预处理后变为 [1,2]
    'Barbara': {'change': 1, 'no_change': 2},  # 预处理后
    'BayArea': {'change': 1, 'no_change': 2},   # 预处理后
}

# =============================================================================
# 函数: plot_prediction
# 功能：绘制并保存变化检测结果可视化图
# 颜色定义：
#   - 白色 [255,255,255]: TP (真正例，正确检测的变化)
#   - 红色 [255,0,0]: FP (假正例，误报)
#   - 绿色 [0,255,0]: FN (假负例，漏报)
#   - 黑色 [0,0,0]: TN (真负例，正确检测的未变化)
#   - 灰色 [100,100,100]: 其他（异常值）
# 输入：
#   - prediction_matrix: 预测结果图 (height, width)，值为1或2
#   - labels: 真实标签图 (height, width)
#   - dataset: 数据集名称 (str)
#   - save_path: 保存路径 (str)
# 输出：无（直接保存图像文件）
# =============================================================================
def plot_prediction(prediction_matrix, labels, dataset, save_path):
    # 从配置字典获取标签映射
    config = LABEL_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Dataset {dataset} not found in LABEL_CONFIG")

    change_label = config['change']
    no_change_label = config['no_change']

    # 计算分类结果
    tp = (prediction_matrix == change_label) & (labels == change_label)
    fp = (prediction_matrix == change_label) & (labels == no_change_label)
    fn = (prediction_matrix == no_change_label) & (labels == change_label)
    tn = (prediction_matrix == no_change_label) & (labels == no_change_label)
    other = ~(tp | fp | fn | tn)

    # 颜色映射
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    result[tp] = [255, 255, 255]  # 白色：正确变化
    result[fp] = [255, 0, 0]  # 红色：误报
    result[fn] = [0, 255, 0]  # 绿色：漏报
    result[tn] = [0, 0, 0]  # 黑色：正确未变化
    result[other] = [100, 100, 100]

    img = Image.fromarray(result)
    img.save(save_path)


# =============================================================================
# 主程序：数据加载、模型训练和测试流程
# 处理多个数据集：hermiston, BayArea, Barbara
# 每个数据集运行多次，记录最高和最低value的结果
# =============================================================================

# 数据集列表
datasets = ['hermiston', 'BayArea', 'Barbara']
all_results = {}  # 存储所有结果
output_dir = './output'  # 输出目录
# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个数据集
for dataset in datasets:
    # =============================================================================
    # 数据集加载部分
    # 每个数据集有两个时相的图像和一张变化标签图
    # data_t1: 时相1高光谱图像 (height, width, band)
    # data_t2: 时相2高光谱图像 (height, width, band)
    # data_label: 变化标签图 (height, width)，1表示变化，2表示未变化
    # uc_position: 未变化像素坐标 (N_uc, 2)
    # c_position: 变化像素坐标 (N_c, 2)
    # =============================================================================
    if dataset == 'BayArea':
        data_t1 = loadmat(r'D:/XAIZAI/datasets/bayArea/Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'D:/XAIZAI/datasets/bayArea/Bay_Area_2015.mat')['HypeRvieW']
        data_label = loadmat(r'D:/XAIZAI/datasets/bayArea/bayArea_gtChanges2.mat')[
            'HypeRvieW']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
    elif dataset == 'Barbara':
        data_t1 = loadmat(r'D:/XAIZAI/datasets/santaBarbara/barbara_2013.mat')['HypeRvieW']
        data_t2 = loadmat(r'D:/XAIZAI/datasets/santaBarbara/barbara_2014.mat')['HypeRvieW']
        data_label = loadmat(r'D:/XAIZAI/datasets/santaBarbara/barbara_gtChanges.mat')[
            'HypeRvieW']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
    elif dataset == 'river':
        data_t1 = loadmat(r'D:/XAIZAI/datasets/river/river_before.mat')['river_before']
        data_t2 = loadmat(r'D:/XAIZAI/datasets/river/river_after.mat')['river_after']
        data_label = loadmat(r'D:/XAIZAI/datasets/river/groundtruth.mat')['lakelabel_v1']
        uc_position = np.array(np.where(data_label == 0)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 255)).transpose(1, 0)
        # 归一化标签到[0,1]然后映射到[1,2]
        data_label = (data_label - data_label.min()) / (data_label.max() - data_label.min())
        data_label[data_label == 0] = 2
    elif dataset == 'farmland':
        data_t1 = loadmat(r'D:/XAIZAI/datasets/farm/farm06.mat')['imgh']
        data_t2 = loadmat(r'D:/XAIZAI/datasets/farm/farm07.mat')['imghl']
        data_label = loadmat(r'D:/XAIZAI/datasets/farm/label.mat')['label']
        uc_position = np.array(np.where(data_label == 0)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        data_label[data_label == 0] = 2
    elif dataset == 'hermiston':
        data_t1 = loadmat(r'D:/XAIZAI/datasets/Hermiston/hermiston2004.mat')['HypeRvieW']
        data_t2 = loadmat(r'D:/XAIZAI/datasets/Hermiston/hermiston2007.mat')['HypeRvieW']
        data_label = loadmat(r'D:/XAIZAI/datasets/Hermiston/label.mat')['label']
        uc_position = np.array(np.where(data_label == 0)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        data_label[data_label == 0] = 2
    else:
        raise ValueError("Unknown dataset")

    # 打印数据集标签信息
    print(f"Dataset: {dataset}, labels unique after preprocess: {np.unique(data_label)}")

    # =============================================================================
    # 数据归一化
    # 对每个波段分别进行Min-Max归一化
    # 输入：data_t1, data_t2 (height, width, band)
    # 输出：input1_normalize, input2_normalize (height, width, band)，值域[0,1]
    # =============================================================================
    input1_normalize = np.zeros(data_t1.shape)
    input2_normalize = np.zeros(data_t1.shape)
    for i in range(data_t1.shape[2]):
        # 取两个时相的最大最小值，保证一致性
        input_max = max(np.max(data_t1[:, :, i]), np.max(data_t2[:, :, i]))
        input_min = min(np.min(data_t1[:, :, i]), np.min(data_t2[:, :, i]))
        denominator = input_max - input_min
        if denominator == 0:  # 避免除零
            input1_normalize[:, :, i] = 1
            input2_normalize[:, :, i] = 1
        else:
            input1_normalize[:, :, i] = (data_t1[:, :, i] - input_min) / denominator
            input2_normalize[:, :, i] = (data_t2[:, :, i] - input_min) / denominator
    height, width, band = data_t1.shape

    # =============================================================================
    # 多次运行实验（蒙特卡洛交叉验证）
    # 每次运行随机选择训练样本，评估模型稳定性
    # 输入：args.train_number 训练样本比例（如0.01表示1%）
    # 输出：dataset_results 包含多次运行的评估指标
    # =============================================================================
    dataset_results = []
    for run in range(1):
        # 设置随机种子，保证可复现
        np.random.seed(args.seed + run)
        torch.manual_seed(args.seed + run)
        torch.cuda.manual_seed(args.seed + run)
        cudnn.deterministic = True
        cudnn.benchmark = False

        # =============================================================================
        # 训练/测试集划分
        # 从未变化和变化区域分别随机选取训练样本
        # TR: 训练集标签图，值为1(变化)或2(未变化)
        # TE: 测试集标签图
        # =============================================================================
        selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number * uc_position.shape[0]),
                                       replace=False)
        selected_c = np.random.choice(c_position.shape[0], int(args.train_number * c_position.shape[0]), replace=False)
        selected_uc_position = uc_position[selected_uc]
        selected_c_position = c_position[selected_c]
        TR = np.zeros(data_label.shape)
        for i in range(int(args.train_number * c_position.shape[0])):
            TR[selected_c_position[i][0], selected_c_position[i][1]] = 1
        for i in range(int(args.train_number * uc_position.shape[0])):
            TR[selected_uc_position[i][0], selected_uc_position[i][1]] = 2
        TE = data_label - TR
        num_classes = np.max(TR)
        num_classes = int(num_classes)

        # =============================================================================
        # 数据预处理：生成训练和测试数据
        # 1. 划分训练/测试样本位置
        # 2. 镜像填充图像边界
        # 3. 提取空间邻域patches
        # 4. 提取光谱邻域信息
        # =============================================================================
        total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
        mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
        mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
        x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test,
                                                              patch=args.patches, band_patch=args.band_patches)
        x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test,
                                                              patch=args.patches, band_patch=args.band_patches)
        y_train, y_test = train_and_test_label(number_train, number_test, num_classes)

        # =============================================================================
        # 转换为PyTorch张量并创建DataLoader
        # 数据形状变换：
        #   numpy: (N, patch*patch*band_patch, band)
        #   torch: (N, band, patch*patch*band_patch) 通过transpose实现
        # =============================================================================
        x_train_t1 = torch.from_numpy(x_train_band_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
        x_train_t2 = torch.from_numpy(x_train_band_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        Label_train = Data.TensorDataset(x_train_t1, x_train_t2, y_train)
        x_test_t1 = torch.from_numpy(x_test_band_t1.transpose(0, 2, 1)).type(torch.FloatTensor)
        x_test_t2 = torch.from_numpy(x_test_band_t2.transpose(0, 2, 1)).type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test).type(torch.LongTensor)
        Label_test = Data.TensorDataset(x_test_t1, x_test_t2, y_test)
        label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
        label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)

        # =============================================================================
        # 准备完整数据集（用于最终预测整张变化检测图）
        # 遍历图像每个像素位置，提取其邻域patch
        # x1_true, x2_true: (height*width, patches, patches, band)
        # y_true: 位置索引（实际使用时会被替换）
        # =============================================================================
        x1_true = np.zeros((height * width, args.patches, args.patches, band), dtype=np.float32)
        x2_true = np.zeros((height * width, args.patches, args.patches, band), dtype=np.float32)
        y_true = []
        for i in range(height):
            for j in range(width):
                x1_true[i * width + j, :, :, :] = mirror_image_t1[i:(i + args.patches), j:(j + args.patches), :]
                x2_true[i * width + j, :, :, :] = mirror_image_t2[i:(i + args.patches), j:(j + args.patches), :]
                y_true.append(i)
        y_true = np.array(y_true)
        x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
        x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
        x1_true_band = torch.from_numpy(x1_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
        x2_true_band = torch.from_numpy(x2_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
        y_true = torch.from_numpy(y_true).type(torch.LongTensor)
        Label_true = Data.TensorDataset(x1_true_band, x2_true_band, y_true)
        label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

        print('------测试数据加载完毕------')

        # =============================================================================
        # 模型、损失函数、优化器、学习率调度器初始化
        # model: HyGSTAN网络，输入两个时相数据，输出变化概率
        # criterion: 自定义损失函数，带难分样本加权
        # optimizer: AdamW优化器，带权重衰减
        # scheduler: 学习率衰减，每step_size轮乘以gamma
        # =============================================================================
        model = hygstan(
            image_size=args.patches,
            num_patches=band,
            p=12,
            d=64,
        )
        model = model.cuda()
        criterion = Loss_fn(delta=4.0, lambda_param=0.3).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                      amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 20, gamma=args.gamma)

        print(f"开始第 {run + 1} 次训练 - 数据集: {dataset}")
        tic = time.time()
        # =============================================================================
        # 训练循环
        # 每轮：训练 -> 计算指标 -> 定期评估测试集
        # 记录训练损失、准确率、F1、精确率、召回率、Kappa系数
        # =============================================================================
        for epoch in range(args.epoches):
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            OA1, F11, Pr1, Re1, Kappa1, AA_mean1, AA1 = output_metric(tar_t, pre_t)
            print(
                "Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f} | F1: {:.4f} | Pr: {:.4f} | Re: {:.4f} | Kappa: {:.4f}"
                .format(epoch + 1, train_obj, train_acc, F11, Pr1, Re1, Kappa1))
            # 定期评估
            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                OA2, F12, Pr2, Re2, Kappa2, AA_mean2, AA2 = output_metric(tar_v, pre_v)
                print("OA: {:.4f} | F1: {:.4f} | Pr: {:.4f} | Re: {:.4f} | Kappa: {:.4f} | AA: {:.4f}"
                      .format(OA2, F12, Pr2, Re2, Kappa2, AA_mean2))
                scheduler.step()

        toc = time.time()
        train_time = toc - tic
        print(f"训练完成 - 耗时: {train_time:.2f} 秒")

        # =============================================================================
        # 保存模型权重
        # 路径: log/hygstan_{dataset}_run{run}.pth
        # =============================================================================
        log_dir = 'log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_save_path = f"{log_dir}/hygstan_{dataset}_run{run}.pth"
        torch.save(model.state_dict(), model_save_path)

        # =============================================================================
        # 测试阶段：加载模型，对整张图像进行预测
        # 生成预测图 prediction_matrix (height, width)
        # =============================================================================
        print(f"开始第 {run + 1} 次测试 - 数据集: {dataset}")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(height):
            for j in range(width):
                prediction_matrix[i, j] = pre_u[i * width + j] + 1

        # 绘制并保存预测结果可视化图
        plot_save_path = f"output/{dataset}_prediction_run{run}.png"
        plot_prediction(prediction_matrix, data_label, dataset, plot_save_path)

        print(f"Dataset: {dataset}, prediction_matrix unique: {np.unique(prediction_matrix)}")

        # =============================================================================
        # 计算最终评估指标
        # 展平标签和预测，计算OA、F1、Pr、Re、Kappa、AA
        # value: 综合指标，五项指标之和
        # =============================================================================
        tar_test = data_label.flatten() - 1  # 调整标签以匹配预测（0-based）
        pre_test = prediction_matrix.flatten() - 1
        OA_test, F1_test, Pr_test, Re_test, Kappa_test, AA_mean_test, AA_test = output_metric(tar_test, pre_test)

        # 计算value（五个指标的和）
        value = OA_test + F1_test + Pr_test + Re_test + Kappa_test

        # 记录结果
        result = {
            "run": run,
            "train_time": train_time,
            "OA": OA_test,
            "Kappa": Kappa_test,
            "F1": F1_test,
            "Pr": Pr_test,
            "Re": Re_test,
            "value": value  # 综合指标
        }
        dataset_results.append(result)
        print(f"第 {run + 1} 次运行完成 - Value: {value:.4f}, OA: {OA_test:.4f}, Kappa: {Kappa_test:.4f}")

    # =============================================================================
    # 汇总当前数据集的多次运行结果
    # 记录最高value和最低value的结果
    # =============================================================================
    if dataset_results:
        max_value_result = max(dataset_results, key=lambda x: x["value"])
        min_value_result = min(dataset_results, key=lambda x: x["value"])
        all_results[dataset] = {
            "max_value": max_value_result,
            "min_value": min_value_result,
            "all_runs": dataset_results  # 保存所有运行结果
        }
        print(f"\n数据集 {dataset} 完成所有运行")
        print(f"最高Value: {max_value_result['value']:.4f} (Run {max_value_result['run'] + 1})")
        print(f"最低Value: {min_value_result['value']:.4f} (Run {min_value_result['run'] + 1})")

    # 构建完整的保存路径
    results_path = os.path.join(output_dir, 'results.json')

    # 保存结果到JSON文件
    try:
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n数据集 {dataset} 结果已保存到 {results_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

print(f"\n所有数据集运行完成，最终结果已保存到 {results_path}")
