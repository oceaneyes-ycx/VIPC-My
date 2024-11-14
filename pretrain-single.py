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
    # 初始化WandB
    wandb.init(project='Pretrain', config=cfg)
    torch.backends.cudnn.benchmark = True

    # 加载数据集
    ViPC_train = ViPCDataLoader2(
        r'/home/ycx/code/EGIInet-analy/train_list.txt',
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
    img_model = img_tokenizer2().cuda()
    point_model = pc_tokenizer2().cuda()
    
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
            batch_size = img_rgb.size(0)

            opt.zero_grad()

            # 特征提取和损失计算
            data = torch.cat((pc_t1, pc_t2))
            _, _, point_feats = point_model(data, data.transpose(1, 2).contiguous())

            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]

            loss_intra = criterion(point_t1_feats, point_t2_feats)

            img_rgb_feat, _, _ = img_model(img_rgb)
            point_feats_combined = torch.cat([point_t1_feats, point_t2_feats], dim=1)
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