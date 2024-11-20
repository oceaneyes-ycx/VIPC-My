'''
V3-2版本
图像编码器选择resnet50
点云编码器选择XMNet的DGCNN
'''

'''
V3版本，使用resnet18作为backbone，替代之前的图片编码器
'''


'''
v2版本
添加了点云的预处理步骤，将原始点云的3500个点采样到2048个点，同时修改编码器的输出，为了适应后期的多模态对齐
此次训练改为使用数据集中的全部数据，不再使用数据集中的部分数据
dataloader进行修改，不再使用全部的GT点云数据去对比训练，每一个GT点云只使用一张对应的图片
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
from models.resnet import ResNet
from models.dgcnn import DGCNN_cls, DGCNN_partseg, DGCNN_semseg
from torchvision.models import resnet50, resnet18, resnet101

from PIL import ImageFile

import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn_cls', metavar='N',
                        choices=['dgcnn_cls', 'dgcnn_part', 'dgcnn_sem', 'pointnet_cls'],
                        help='Model to use, [dgcnn_cls, dgcnn_part, dgcnn_sem, pointnet_cls]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--n_imgs', type=int, default=1, metavar='N',
                        help='number of images ')
    parser.add_argument('--alpha', type=float, default=1, metavar='A',
                        help='cross-model weight (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args(args=[])




def train(cfg,args):
    # 初始化WandB
    wandb.init(project='Pretrain', config=cfg)
    torch.backends.cudnn.benchmark = True

    # 加载数据集
    ViPC_train = ViPCDataLoader2(
        r'/home/ycx/code/VIPC-My/train_list_contrasive.txt',
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

    # 模型和优化器初始化
    #img_model = img_tokenizer2().cuda()
    img_model = ResNet(resnet50(), feat_dim = 2048).cuda()
    point_model = DGCNN_cls(args).cuda()
    
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
            data = torch.cat((pc_t1, pc_t2))
            point_feats, _  = point_model(data.transpose(1, 2).contiguous())
            #out_p:3个点云中心点，0：[2B,2048,3] 1:[2B,512,3] 2:[2B,128,3]
            #out_f:0:[2B,3,2048] 1:[2B,64,512] 2:[2B,128,128]
            #global_feature:torch.Size([2B, 128])

            point_t1_feats = point_feats[:batch_size, :]
            #torch.Size([B, 128])
            point_t2_feats = point_feats[batch_size:, :]
            #torch.Size([B, 128])

            loss_intra = criterion(point_t1_feats, point_t2_feats)

            img_rgb_feat= img_model(img_rgb)#修改输出
            #torch.Size([B, 256])
            #point_feats_combined = torch.cat([point_t1_feats, point_t2_feats], dim=1)
            point_feats_combined = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
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
    args = parse_args()
    train(cfg,args)