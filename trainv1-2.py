#V1版本训练框架
'''
Cross-Model 对比学习用于VIPC任务,使用预训练好的点云和图像编码器（参数冻结）
代码基于XMFNet,设定几乎完全一致


1-2版本：损失函数设定为cd L2
'''

import torch
import os
import time
from model_pre import Network
from decoder.utils.utils import *
import wandb
import argparse
import logging
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from datetime import datetime
from testv1 import test_net

import open3d as o3d
import os
import numpy as np
from PIL import Image


loss_cd=L2_ChamferLoss()#训练损失函数
loss_cd_eval=L1_ChamferEval()#评估损失函数


# 确保结果文件夹的创建 （用于调试）
'''
result_dir = "../output"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, "partial_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "gt_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "completed_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "input_images"), exist_ok=True)

# 保存点云为 PLY 文件
def save_point_cloud_as_ply(filepath, point_cloud):
    """
    将点云保存为 PLY 文件。
    
    Args:
        filepath: 保存路径。
        point_cloud: numpy array, 形状 [N, 3] 的点云。
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filepath, pc)


# 随机抽取并保存结果
def save_random_samples(partial_pc, gt_pc, completed_pc, images, epoch, max_samples=5):
    """
    从当前数据中随机抽样，并保存到 PLY 文件和图像文件。

    Args:
        partial_pc: numpy array, 形状 [B, N, 3] 的部分点云。
        gt_pc: numpy array, 形状 [B, N, 3] 的GT点云。
        completed_pc: numpy array, 形状 [B, N, 3] 的补全点云。
        images: numpy array, 形状 [B, H, W, 3] 的输入图像。
        epoch: 当前训练的epoch号。
        max_samples: 每个 batch 中随机抽样的数量。
    """
    num_samples = min(partial_pc.shape[0], max_samples)
    indices = random.sample(range(partial_pc.shape[0]), num_samples)  # 随机抽取样本索引

    for idx in indices:
        # 构造文件名
        filename = f"epoch_{epoch}_sample_{idx}"

        # 保存部分点云
        partial_path = os.path.join(result_dir, "partial_pc", f"{filename}.ply")
        save_point_cloud_as_ply(partial_path, partial_pc[idx])

        # 保存GT点云
        gt_path = os.path.join(result_dir, "gt_pc", f"{filename}.ply")
        save_point_cloud_as_ply(gt_path, gt_pc[idx])

        # 保存补全点云
        completed_path = os.path.join(result_dir, "completed_pc", f"{filename}.ply")
        save_point_cloud_as_ply(completed_path, completed_pc[idx])

        # 保存对应的输入图像
        img_path = os.path.join(result_dir, "input_images", f"{filename}.png")
        img = Image.fromarray((images[idx] * 255).astype(np.uint8))  # 归一化图像转uint8
        img.save(img_path)




# 保存点云为txt文件和图像为png
def save_results(partial_pc, gt_pc, completed_pc, images, epoch, batch_idx):
    """
    保存当前批次的部分点云、GT点云、补全点云和对应的2D图像。

    Args:
        partial_pc: numpy array, 形状 [B, N, 3] 的部分点云
        gt_pc: numpy array, 形状 [B, N, 3] 的GT点云
        completed_pc: numpy array, 形状 [B, N, 3] 的补全点云
        images: numpy array, 形状 [B, H, W, 3] 的输入图像
        epoch: 当前训练的epoch号
        batch_idx: 当前批次索引
    """
    for i in range(partial_pc.shape[0]):
        # 构造文件名
        filename = f"epoch_{epoch}_batch_{batch_idx}_sample_{i}"

        # 保存部分点云
        partial_path = os.path.join(result_dir, "partial_pc", f"{filename}.txt")
        np.savetxt(partial_path, partial_pc[i], fmt="%.6f")

        # 保存GT点云
        gt_path = os.path.join(result_dir, "gt_pc", f"{filename}.txt")
        np.savetxt(gt_path, gt_pc[i], fmt="%.6f")

        # 保存补全点云
        completed_path = os.path.join(result_dir, "completed_pc", f"{filename}.txt")
        np.savetxt(completed_path, completed_pc[i], fmt="%.6f")

        # 保存对应的输入图像
        img_path = os.path.join(result_dir, "input_images", f"{filename}.png")
        img = Image.fromarray((images[i] * 255).astype(np.uint8))  # 归一化图像转uint8
        img.save(img_path)

'''

def train_net(cfg):
    #接收一个配置文件作为参数

    # Initialize wandb
    wandb.init(project="ViPC-Training-V1", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    #创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader(r'./train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)
    
    #创建训练数据加载器，使用pytorch自带的DataLoader类
    train_loader = DataLoader(ViPC_train,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.CONST.NUM_WORKERS,
                                  shuffle=True,
                                  drop_last=True,
                                  prefetch_factor=cfg.CONST.DATA_perfetch)
    #创建ViPC测试数据加载器
    ViPC_test = ViPCDataLoader(r'./test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    #创建测试数据加载器
    test_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    
    #创建模型，加载预训练的编码器
    model = Network(pretrained_model_path=r'../checkpoints/pretrain/best_model.pth')
    model.cuda()
    
    '''
    # 如果GPU可用
    if torch.cuda.is_available():
    # 将模型包装成DataParallel模型，以便在多个GPU上进行并行计算
        model = torch.nn.DataParallel(model).cuda()
    '''


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
    best_epoch = 0


    #开始训练
    for epoch in range(init_epoch+1, cfg.TRAIN.N_EPOCHS + 1):
        
        #遍历每个epoch
        model.train()

        total_cd_pc= 0 #L1 CD loss
        

        total_loss = 0 #总损失

        n_batches = len(train_loader)#获取训练数据加载器的长度

        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])
        with tqdm(train_loader) as t:
            for batch_idex, (view,gt_pc,part_pc) in enumerate(t):

                partial=part_pc.cuda()
                gt=gt_pc.cuda()
                png=view.cuda()

                #对部分点云和GT点云进行FPS采样
                partial=farthest_point_sample(partial,2048)
                gt=farthest_point_sample(gt,2048)

                complete=model(partial,png)   #输出补全的点云

                #计算损失
                cd=loss_cd(complete,gt) #计算L1 CD loss

                loss_total=cd  #目前就一个CD Loss

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()


                '''
                # 如果当前epoch需要保存结果，保存当前训练批次的结果
                if epoch % 30 == 0:
                    save_random_samples(
                        partial_pc=partial.cpu().detach().numpy(),
                        gt_pc=gt_pc.cpu().detach().numpy(),
                        completed_pc=complete.cpu().detach().numpy(),
                        images=view.cpu().permute(0, 2, 3, 1).detach().numpy(),  # [B, C, H, W] -> [B, H, W, C]
                        epoch=epoch,
                        max_samples=5
                    )
                '''

                cd_pc_item=cd.item()*1e3
                total_cd_pc+=cd_pc_item

                loss_item = loss_total.item() * 1e3
                total_loss += loss_item
                t.set_description(f"[Epoch {epoch}/{cfg.TRAIN.N_EPOCHS}] Batch {batch_idex+1}/{len(train_loader)}")
                t.set_postfix(cd_loss=loss_total.item())


        # Average CD loss for epoch
        avg_cdc = total_cd_pc / n_batches #平均L1 CD loss
        wandb.log({"Train/CD_Loss": avg_cdc, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr']})
        
        avg_loss = total_loss / n_batches #平均总损失
        wandb.log({"Train/Total_Loss": avg_loss, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Grad_Norm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)})

        lr_scheduler.step() #学习率调度器更新

        print('epoch:',epoch,'cd_pc:',avg_cdc,'loss:',avg_loss)



        #开始测试
        cd_eval=test_net(cfg,epoch,test_loader,model=model)
        wandb.log({"Val/CD_Loss": cd_eval, "Epoch": epoch})


        # Save checkpoints
        if epoch % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                best_epoch = epoch
                file_name = 'ckpt-best.pth'

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)

    logging.info(f"Training complete. Best performance at epoch {best_epoch} with CD loss {best_metrics:.4f}")
    wandb.finish()     





if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train_net(cfg)

                



