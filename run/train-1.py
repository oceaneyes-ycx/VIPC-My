import logging
import os
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from run.test import test_net
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *

def train_net(cfg):
    # 接受一个配置文件作为参数

    # 设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    # 创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader(r'/project/EGIInet/train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)
    # 创建训练数据加载器，使用pytorch自带的DataLoader类
    train_data_loader = DataLoader(ViPC_train,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.CONST.NUM_WORKERS,
                              shuffle=True,
                              drop_last=True,
                              prefetch_factor=cfg.CONST.DATA_perfetch)
    # 创建ViPC测试数据加载器
    ViPC_test = ViPCDataLoader(r'/project/EGIInet/test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    # 创建测试数据加载器
    val_data_loader = DataLoader(ViPC_test,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=cfg.CONST.NUM_WORKERS,
                                   shuffle=True,
                                   drop_last=True,
                                   prefetch_factor=cfg.CONST.DATA_perfetch)

    # Set up folders for logs and checkpoints
    # 定义输出目录
    output_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TRAIN.CATE, '%s', datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    # 定义检查点目录
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    # 定义日志目录
    cfg.DIR.LOGS = output_dir % 'logs'
    # 如果检查点目录不存在，则创建
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    # 创建一个SummaryWriter对象，用于记录训练过程中的数据
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    # 创建一个SummaryWriter对象，用于记录测试过程中的数据
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    
    # 创建EGIInet模型
    model = EGIInet()#.apply(weights_init_normal)
    # 将模型移动到GPU上
    model.cuda()
    # 如果GPU可用
    if torch.cuda.is_available():
        # 将模型包装成DataParallel模型，以便在多个GPU上进行并行计算
        model = torch.nn.DataParallel(model).cuda()
    
    # Create the optimizers
    # 定义优化器，使用Adam优化算法
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  # 过滤掉不需要梯度的参数
                                 lr=cfg.TRAIN.LEARNING_RATE,  # 学习率
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,  # 权重衰减
                                 betas=cfg.TRAIN.BETAS)  # Adam优化算法的参数

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    #多步学习率调度器
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)
    #渐进式学习率调度器

    init_epoch = 0  #初始化训练epoch为0
    best_metrics = float('inf')#初始化最佳指标为无穷大
    steps = 0#初始化步数为0
    BestEpoch = 0

    if 'WEIGHTS' in cfg.CONST: #使用预训练权重
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE
        logging.info('Recover complete.')

    print('recon_points: ',cfg.DATASETS.SHAPENET.N_POINTS, 'Parameters: ', sum(p.numel() for p in model.parameters()))
    #exit()
    # Training/Testing the network

    #以下为训练（测试）模型的代码
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0

        n_batches = len(train_data_loader)#获取batch_size
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        #
        with tqdm(train_data_loader) as t:
            for batch_idx, (view,gt_pc,part_pc,) in enumerate(t):

                partial = part_pc.cuda()#部分点云数据，[16,2048,3],将数据移动到GPU上
                gt = gt_pc.cuda()#[16,2048,3]，GT点云数据
                png = view.cuda()#图像数据
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                #对部分点云数据进行最远点采样
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)
                #对GT点云数据进行最远点采样
                      
                recon,style_loss=model(partial,png)#前向传播，返回重建点云数据和风格损失
                cd = chamfer_sqrt(recon,gt)#计算重建点云数据和GT点云数据之间的CD差距
                loss_total=cd+style_loss*1e-2#总损失
                losses=[loss_total,cd,style_loss*1e-2]#损失列表
                optimizer.step()#优化器更新

                cd_pc_item = losses[0].item() * 1e3#loss_total
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3#cd
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3#style_loss
                total_cd_p2 += cd_p2_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx#迭代次数
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item]])
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches#计算平均总损失
        avg_cd1 = total_cd_p1 / n_batches#计算平均CD损失
        avg_cd2 = total_cd_p2 / n_batches#计算平均风格损失

        lr_scheduler.step()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        logging.info(
                '[Epoch %d/%d]  Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS,
                 ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2]]))

        # Validate the current model
        #评价当前模型
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        # Save checkpoints
        #保存检查点
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    train_writer.close()
    val_writer.close()
model.load_state_dict(checkpoint['model'])