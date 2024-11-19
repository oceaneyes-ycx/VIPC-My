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



def train(cfg):
    #训练decoder

    #初始化wandb
    wandb.init(project='decoder-train',config=cfg)
    torch.backends.cudnn.benchmark = True

    #加载数据集
    ViPC_train = ViPCDataLoader2(
        r'/home/ycx/code/VIPC-My/train_list.txt',
        data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
        status='train',
        view_align=False,
        category=cfg.TRAIN.CATE
    )

    train_data_loader = DataLoader(
        ViPC_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
        prefetch_factor=cfg.CONST.DATA_perfetch
    )

    #加载预训练模型
    model_path='best_model.pth'
    checkpoint = torch.load(model_path)
    img_model=img_tokenizer2().cuda()
    point_model=pc_tokenizer2().cuda()

    #加载模型权重
    img_model.load_state_dict(checkpoint['img_model_state_dict'])
    point_model.load_state_dict(checkpoint['point_model_state_dict'])

    #将模型设置为评估模式
    img_model.eval()
    point_model.eval()

    






