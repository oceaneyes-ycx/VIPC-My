'''
V4版本
图像编码器：ResNet50
点云编码器：XMNET DGCNN
和V3-2版本相比，点云编码器由crossnet的DGCNN改为XMNET的DGCNN，同时代码结构进行了优化

v4-2版本
图像编码器：ResNet50
点云编码器：XMNET DGCNN
和v4相比修改了resnet,修改为和XMNET保持一致的resnet18
总而言之，此版本就是和XMNET保持一致的版本

V5版本
Easydic配置单独配置pretrain部分
相较于V4-2版本,编码器添加反向头，更好实现对比学习效果。
'''



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
from utils.loss_utils import *

import torchvision.transforms as transforms
from models.resnet import ResNet,ResNet2
from torchvision.models import resnet50, resnet18, resnet101

from PIL import ImageFile

from encoder_dgcnn.dgcnn import DGCNN,DGCNN2

def train(cfg):
    # 初始化WandB
    wandb.init(project='Pretrainv5', config=cfg)
    torch.backends.cudnn.benchmark = True

    # 加载数据集
    ViPC_train = ViPCDataLoader2(
        r'/home/ycx/code/VIPC-My/train_list.txt',
        data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
        status='train',
        view_align=False,
        category=cfg.PRETRAIN.CATE
    )

    train_data_loader = DataLoader(
        ViPC_train,
        batch_size=cfg.PRETRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
        prefetch_factor=cfg.CONST.DATA_perfetch
    )

    # 模型和优化器初始化
    #img_model = img_tokenizer2().cuda()
    img_model = ResNet2(resnet18(), feat_dim = 512).cuda()
    #point_model = pc_tokenizer2(embed_dim=256).cuda()
    point_model = DGCNN2().cuda()
    
    # 使用DataParallel封装模型
    #img_model = nn.DataParallel(img_model)
    #point_model = nn.DataParallel(point_model)
    
    wandb.watch(img_model, log="all")
    wandb.watch(point_model, log="all")

    params = list(img_model.parameters()) + list(point_model.parameters())
    opt = optim.Adam(params, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=cfg.TRAIN.N_EPOCHS, eta_min=0, last_epoch=-1)

    # 损失函数
    criterion = NTXentLoss(temperature=0.1).cuda()
    # 加载模型检查点
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = 'best_model.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        img_model.load_state_dict(checkpoint['img_model_state_dict'])
        point_model.load_state_dict(checkpoint['point_model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    for epoch in range(start_epoch, cfg.TRAIN.N_EPOCHS):
        train_loss = AverageMeter2()
        train_intra_loss = AverageMeter2()
        train_cross_loss = AverageMeter2()

        point_model.train()
        img_model.train()

        print('Start training epoch:', epoch)

        for (img_rgb, pc_t1, pc_t2) in tqdm(train_data_loader):
            img_rgb, pc_t1, pc_t2 = img_rgb.cuda(), pc_t1.cuda(), pc_t2.cuda()
            #img_rgb: [B, 3, 224, 224]
            #下采样调整点云数量
            pc_t1=farthest_point_sample(pc_t1,cfg.DATASETS.SHAPENET.N_POINTS)
            #torch.Size([B, 2048, 3])
            pc_t2=farthest_point_sample(pc_t2,cfg.DATASETS.SHAPENET.N_POINTS)
            batch_size = img_rgb.size(0)

            opt.zero_grad()

            # 特征提取和损失计算
            data = torch.cat((pc_t1, pc_t2))  #torch.Size([2B, 2048, 3])
            #point_feats = point_model(data, data.transpose(1, 2).contiguous())
            _ , point_feats = point_model(data.transpose(1, 2).contiguous())
            #torch.Size([2B,256])


            point_t1_feats = point_feats[:batch_size, :]
            #torch.Size([B,256])
            point_t2_feats = point_feats[batch_size:, :]
            #torch.Size([B,256])

            loss_intra = criterion(point_t1_feats, point_t2_feats)

            _ ,img_rgb_feat= img_model(img_rgb)#修改输出
            #torch.Size([B, 256])
            #point_feats_combined = torch.cat([point_t1_feats, point_t2_feats], dim=1)
            point_feats_combined = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
            #torch.Size([B,256])
            loss_cross = criterion(img_rgb_feat, point_feats_combined)

            total_loss = loss_intra + loss_cross
            total_loss.backward()
            opt.step()
            scheduler.step()

            # 更新平均损失
            train_loss.update(total_loss.item(), batch_size)
            train_intra_loss.update(loss_intra.item(), batch_size)
            train_cross_loss.update(loss_cross.item(), batch_size)

            # 每个batch日志记录
            wandb.log({
                'Batch Total Loss': total_loss.item(),
                'Batch Intra Loss': loss_intra.item(),
                'Batch Cross Loss': loss_cross.item(),
                'Learning Rate': scheduler.get_last_lr()[0]
            })

        # 每个epoch日志记录
        print(f'Epoch: {epoch}, Train Loss: {train_loss.avg}, Intra Loss: {train_intra_loss.avg}, Cross Loss: {train_cross_loss.avg}')
        wandb.log({
            'Epoch': epoch,
            'Train Loss': train_loss.avg,
            'Intra Loss': train_intra_loss.avg,
            'Cross Loss': train_cross_loss.avg
        })

        # 保存最好的模型
        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save({
                'epoch': epoch,
                'img_model_state_dict': img_model.state_dict(),
                'point_model_state_dict': point_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': train_loss.avg,
            }, 'best_model.pth')

    # 保存最后一个epoch的模型
    torch.save({
        'epoch': epoch,
        'img_model_state_dict': img_model.state_dict(),
        'point_model_state_dict': point_model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': train_loss.avg,
    }, 'last_model.pth')

if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train(cfg)


