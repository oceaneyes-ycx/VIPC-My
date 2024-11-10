import logging
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.data_loaders
#import utils.helpers
from tqdm import tqdm

from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
#import open3d as o3d
#import os
#import numpy as np
#from PIL import Image


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    #接收一个配置文件作为参数
    #cfg:配置文件
    #epoch_idx:epoch的索引
    #test_data_loader:测试数据加载器
    #test_writer:测试数据写入器
    #model:待测试的模型

    torch.backends.cudnn.benchmark = True   #启用cuDNN的benchmark模式，以加速卷积操作

    if test_data_loader is None:
        # Set up data loader
        # 创建ViPC测试数据加载器

        ViPC_test = ViPCDataLoader(r'/home/ycx/code/EGIInet-analy/test_list.txt',
                                   data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
                                   status='test',
                                   view_align=False, category=cfg.TEST.CATE)
        test_data_loader = DataLoader(ViPC_test,
                                     batch_size=cfg.TEST.BATCH_SIZE,
                                     num_workers=cfg.CONST.NUM_WORKERS,
                                     shuffle=True,
                                     drop_last=True,
                                     prefetch_factor=cfg.CONST.DATA_perfetch)

    # Setup networks and initialize networks
    if model is None:#未指定模型
        model = EGIInet()#.apply(weights_init_normal)
        #创建模型
        model.cuda()#将模型移动到GPU上
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)#加载权重
        model.load_state_dict(checkpoint['model'])#加载模型的状态字典

    # Switch models to evaluation mode
    model.eval()#将模型切换到评估模式，不再更新梯度

    n_samples = len(test_data_loader)#测试数据集的样本数
    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])#CD L1、CD L2和F1的平均值
    test_metrics = AverageMeter(['CDl1', 'CDl2', 'F1'])
    category_metrics = dict()

    # 创建保存文件夹
    #output_dir = r'/home/ycx/code/EGIInet-analy/output-PC-complete'
    #input_img_dir = r'/home/ycx/code/EGIInet-analy/input-images'
    #partial_pc_dir = r'/home/ycx/code/EGIInet-analy/partial-PC'
    #gt_pc_dir = r'/home/ycx/code/EGIInet-analy/gt-PC'

    #os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(input_img_dir, exist_ok=True)
    #os.makedirs(partial_pc_dir, exist_ok=True)
    #os.makedirs(gt_pc_dir, exist_ok=True)

    # Testing loop
    #开始测试循环
    with tqdm(test_data_loader) as t:
        for model_idx, (view,gt_pc,part_pc) in enumerate(t):

            with torch.no_grad():#不计算梯度

                partial = part_pc.cuda()  # [16,2048,3]
                gt = gt_pc.cuda()  # [16,2048,3]
                png = view.cuda()
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                #部分点云采样
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)
                #完整点云采样

                model.eval()#将模型切换到评估模式，不再更新梯度
              
                pcds_pred,_ = model(partial, png)
                cdl1, cdl2, f1 = calc_cd(pcds_pred, gt, calc_f1=True)
                   
                cdl1 = cdl1.mean().item() * 1e3
                cdl2 = cdl2.mean().item() * 1e3
                f1 = f1.mean().item()

                _metrics = [cdl1, cdl2, f1]
                test_losses.update([cdl1, cdl2, f1])

                test_metrics.update(_metrics)

    '''                                # 保存输入的2D图像
                for i, img in enumerate(png):
                    img_np = img.cpu().numpy().transpose(1, 2, 0) * 255
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                    img_pil.save(os.path.join(input_img_dir, f'input_img_{model_idx}_{i}.png'))

                # 保存部分点云
                for i, pcd in enumerate(partial):
                    pcd_np = pcd.cpu().numpy()
                    output_path = os.path.join(partial_pc_dir, f'partial_pcd_{model_idx}_{i}.ply')
                    pcd_o3d = o3d.geometry.PointCloud()
                    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
                    o3d.io.write_point_cloud(output_path, pcd_o3d)

                # 保存GT完整点云
                for i, pcd in enumerate(gt):
                    pcd_np = pcd.cpu().numpy()
                    output_path = os.path.join(gt_pc_dir, f'gt_pcd_{model_idx}_{i}.ply')
                    pcd_o3d = o3d.geometry.PointCloud()
                    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
                    o3d.io.write_point_cloud(output_path, pcd_o3d)
    '''

    #保存补全之后的点云文件
    #output_dir=r'/home/ycx/code/EGIInet-analy/output-PC-complete'
    '''
    for i, pcd in enumerate(pcds_pred):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(output_dir, f'completed_pcd_{model_idx}_{i}.ply')
        
        # 使用 open3d 保存为 .ply 文件
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)   
    '''




    # Print testing results
    print('============================ TEST RESULTS ============================')

    """
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    """
    
    print('Overall', end='\t\t\t')
    #输出测试结果
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
        #cdl1, cdl2, f1
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)#返回第一个测试损失的平均值