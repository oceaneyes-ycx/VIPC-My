import torch
import torch.nn as nn
import os
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader2
from models.encoders import img_tokenizer,img_tokenizer2,pc_tokenizer,pc_tokenizer2
from config_vipc import cfg
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.average_meter import AverageMeter, AverageMeter2
from datasets.data import ShapeNetRender, ModelNet40SVM, ScanObjectNNSVM

import torchvision.transforms as transforms

from PIL import ImageFile

def load_and_infer(img_rgb, pc_t1, model_path='best_model.pth'):
    # 加载模型
    checkpoint = torch.load(model_path)
    img_model = img_tokenizer2().cuda()
    point_model = pc_tokenizer2().cuda()
    
    # 加载模型权重
    img_model.load_state_dict(checkpoint['img_model_state_dict'])
    point_model.load_state_dict(checkpoint['point_model_state_dict'])
    
    # 将模型设置为评估模式
    img_model.eval()
    point_model.eval()
    
    with torch.no_grad():
        # 将输入数据转移到 GPU
        img_rgb = img_rgb.cuda()
        pc_t1 = pc_t1.cuda()

        # 提取图像和点云的特征
        img_rgb_feat, _, _ = img_model(img_rgb)
        _, _, point_feats = point_model(pc_t1, pc_t1.transpose(1, 2).contiguous())

    return img_rgb_feat, point_feats

if __name__ == '__main__':
    # 示例测试输入
    # 假设 img_rgb 和 pc_t1 是从数据集中提取的图像和点云数据
    img_rgb = torch.randn(16, 3, 224, 224)
    pc_t1 = torch.randn(16, 3500, 3)

    # 使用测试输入进行推理
    img_rgb_feat, point_feats = load_and_infer(img_rgb, pc_t1)
    print("Image Feature:", img_rgb_feat)
    print("Point Cloud Feature:", point_feats)


# 示例测试输入
# 假设 img_rgb 和 pc_t1 是从数据集中提取的图像和点云数据
# img_rgb = ...  # 输入图像张量，形状应为 (B, 3, H, W)
# pc_t1 = ...    # 输入点云张量，形状应为 (B, N, 3)

# 使用测试输入进行推理
# img_rgb_feat, point_feats = load_and_infer(img_rgb, pc_t1)
# print("Image Feature:", img_rgb_feat)
# print("Point Cloud Feature:", point_feats)
