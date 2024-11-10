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
from utils.average_meter import AverageMeter,AverageMeter2
from datasets.data import ShapeNetRender, ModelNet40SVM, ScanObjectNNSVM

import torchvision.transforms as transforms

from PIL import ImageFile





def train(cfg):
    #加载VIPC数据集
    torch.backends.cudnn.benchmark = True
    ViPC_train = ViPCDataLoader2(r'/home/ycx/code/EGIInet-analy/train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)
    #只会加载train_list.txt中的类别为cfg.TRAIN.CATE的数据
    #print('ViPC_train:',ViPC_train)
    #print('ViPC_train.__len__():',ViPC_train.__len__())
    #img,gt,part=ViPC_train.__getitem__(3000)
    #print('img:',img)
    #print('gt:',gt)
    #print('part:',part)

    train_data_loader = DataLoader(ViPC_train,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.CONST.NUM_WORKERS,
                              shuffle=True,
                              drop_last=True,
                              prefetch_factor=cfg.CONST.DATA_perfetch)
    #print('train_data_loader:',train_data_loader)
    #print('train_data_loader.__len__():',train_data_loader.__len__())
    #img,gt,part=next(iter(train_data_loader))#获取一个batch的数据
    #print('img:',img)
    #print('gt:',gt)
    #print('part:',part)




    img_model=img_tokenizer2()
    img_model=img_model.cuda()
    #print('img_model:',img_model)
    point_model=pc_tokenizer2()
    point_model=point_model.cuda()
    #print('point_model:',point_model)

    #pro5ject='Pretrain'

    #wandb.init(project=project)
    #wandb.watch(img_model)

    params = list(img_model.parameters()) + list(point_model.parameters())
    #print('params:',params)

    #设置优化器
    opt=optim.Adam(params,lr=cfg.TRAIN.LEARNING_RATE,weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    #设置学习率调度器
    scheduler = CosineAnnealingLR(opt, T_max=cfg.TRAIN.N_EPOCHS, eta_min=0, last_epoch=-1)

    #设置损失函数
    criterion = NTXentLoss(temperature = 0.1).cuda()
    #对比学习损失函数，设定温度参数为0.1，控制相似度分布的平滑程度

    best_acc = 0#最佳准确率

    for epoch in range(0, cfg.TRAIN.N_EPOCHS):
        ############################
        # Train
        #在shapenet-VIPC数据集上进行自监督训练
        ############################
        train_loss=AverageMeter2()#总平均损失
        train_intra_loss=AverageMeter2()#模态内平均损失
        train_cross_loss=AverageMeter2()#模态间平均损失

        point_model.train()
        img_model.train()

        wandb.log={}#记录日志
        print('Start training epoch:',epoch)

        for (img_rgb,pc_t1,pc_t2) in tqdm(train_data_loader):
            #img_rgb:RGB图像
            #gt_pc:点云
            #part_pc:点云的部分
            img_rgb = img_rgb.cuda()
            #print('img_rgb:',img_rgb)
            #print('img_rgb.shape:',img_rgb.shape)
            #img_rgb:[16,3,224,224]  [batch_size,通道数,高,宽]
            #gt_pc = gt_pc.cuda()
            #part_pc = part_pc.cuda()
            #gt_pc=gt_pc.numpy()

            #print('gt_pc:',gt_pc)
            #print('gt_pc.shape:',gt_pc.shape)
            #gt_pc:[16,3500,3]  [batch_size,点云数量,3]

            pc_t1 = pc_t1.cuda()
            pc_t2 = pc_t2.cuda()
            batch_size = img_rgb.size(0)
            #print('batch_size:',batch_size)
            #print('pc_t1_shape:',pc_t1.shape)
            #torch.Size([16, 3500, 3])
            #print('pc_t2_shape:',pc_t2.shape)
            #torch.Size([16, 3500, 3])

            opt.zero_grad()#梯度清零

            #print('pc_t1_shape:',pc_t1.shape)
            #print('pc_t1:',pc_t1)

            data=torch.cat((pc_t1,pc_t2))#将两个点云数据拼接在一起 
            #data = data.transpose(2, 1).contiguous()#转置，(B,3,N) -> (B,N,3)           
            #print('data_shape:',data.shape)
            #torch.Size([32, 3500, 3])

            _ , _ , point_feats = point_model(data,data.transpose(1,2).contiguous())#点云模型输出特征
            #print('point_feats:',point_feats)
            #这里输出的特征是点云的特征,是一个包含了三个tensor的list
            #0：torch.size([32, 3, 3500])原始点云的特征
            #1: torch.size([32,192,875])经过第一阶段卷积和下采样之后的点云特征
            #2: torch.size([32,384,218])经过第二阶段卷积和下采样之后的点云特征

            
            #point_feats = point_feats[2]  # Extract the tensor from the tuple
            #print('point_feats.shape:',point_feats.shape)
            #torch.Size([32, 3 ,3500])


            point_t1_feats = point_feats[:batch_size, :]#第一个点云数据的特征
            #[16,384,218]
            #更新之后：torch.Size([16, 384])
            point_t2_feats = point_feats[batch_size:, :]
            #[16,384,218]
            #更新之后：torch.Size([16, 384])


            loss_intra = criterion(point_t1_feats, point_t2_feats)


            #img_rgb:[16,3,224,224]
            img_rgb_feat, _ , _  = img_model(img_rgb)#2d图像的特征
            #print('img_rgb_feat.shape:',img_rgb_feat.shape)
            #torch.Size([16,768])
            
            point_feats = torch.cat([point_t1_feats, point_t2_feats], dim=1)
            #torch.Size([16,768])


            
            alpha=1

            loss_cross = alpha*(criterion(img_rgb_feat,point_feats))

            total_loss = loss_intra + loss_cross#总损失=模态内损失+模态间损失
            total_loss.backward()#反向传播
            opt.step()#优化器更新参数
            scheduler.step()#学习率调度器更新学习率

            train_loss.update(total_loss.item(),batch_size)
            train_intra_loss.update(loss_intra.item(),batch_size)
            train_cross_loss.update(loss_cross.item(),batch_size)

        print('Epoch:',epoch,'Train Loss:',train_loss.avg,'Intra Loss:',train_intra_loss.avg,'Cross Loss:',train_cross_loss.avg)

        wandb.log({'Train Loss':train_loss.avg,'Intra Loss':train_intra_loss.avg,'Cross Loss':train_cross_loss.avg})







if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train(cfg)



