import os
import torch
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader
from Symm.symm import SymmCompletion3
from utils.loss_utils import calc_cd
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.average_meter import AverageMeter
from decoder.utils.utils import farthest_point_sample
import config_vipc  # 加载配置


def save_pointcloud_batch(pcd_batch, save_dir, prefix, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    for i, pcd in enumerate(pcd_batch):
        pcd_np = pcd.cpu().numpy()
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        filename = os.path.join(save_dir, f"{prefix}_{batch_idx}_{i}.ply")
        o3d.io.write_point_cloud(filename, pcd_o3d)


def save_images_batch(img_batch, save_dir, prefix, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(img_batch):
        img_np = img.cpu().numpy().transpose(1, 2, 0) * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        filename = os.path.join(save_dir, f"{prefix}_{batch_idx}_{i}.png")
        img_pil.save(filename)


def main():
    cfg = config_vipc.cfg  # 获取配置
    cfg.TEST.CATE = cfg.TRAIN.CATE  # 保证测试类别一致

    # 加载测试数据
    test_dataset = ViPCDataLoader('./test_list.txt',
                                   data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
                                   status='test',
                                   view_align=False,
                                   category=cfg.TEST.CATE)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             num_workers=cfg.CONST.NUM_WORKERS,
                             shuffle=False,
                             drop_last=False,
                             prefetch_factor=cfg.CONST.DATA_perfetch)

    # 加载模型
    model = SymmCompletion3(cfg).cuda()
    checkpoint_path = os.path.join(cfg.DIR.CHECKPOINTS, 'chair-ckpt-best.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 初始化输出路径
    out_dir = "../output"
    os.makedirs(out_dir, exist_ok=True)

    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])

    # 测试过程
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as t:
            for batch_idx, (view, gt_pc, part_pc) in enumerate(t):
                partial = part_pc.cuda()
                gt = gt_pc.cuda()
                png = view.cuda()

                partial = farthest_point_sample(partial, 2048)
                gt = farthest_point_sample(gt, 2048)

                _, recon, _, _ = model(partial, png)

                cdl1, cdl2, f1 = calc_cd(recon, gt, calc_f1=True)
                cdl1 = cdl1.mean().item() * 1e3
                cdl2 = cdl2.mean().item() * 1e3
                f1 = f1.mean().item()

                test_losses.update([cdl1, cdl2, f1])

                # 保存图像和点云
                save_images_batch(view, os.path.join(out_dir, "input-images"), "input_img", batch_idx)
                save_pointcloud_batch(partial, os.path.join(out_dir, "partial-PC"), "partial_pcd", batch_idx)
                save_pointcloud_batch(gt, os.path.join(out_dir, "gt-PC"), "gt_pcd", batch_idx)
                save_pointcloud_batch(recon, os.path.join(out_dir, "PC-complete"), "completed_pcd", batch_idx)

    # 输出测试结果
    print('\n==================== FINAL TEST RESULTS ====================')
    print(f"CD-L1: {test_losses.avg(0):.4f}  |  CD-L2: {test_losses.avg(1):.4f}  |  F1: {test_losses.avg(2):.4f}")
    print('============================================================')


if __name__ == "__main__":
    main()
