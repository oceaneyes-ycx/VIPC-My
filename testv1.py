'''
测试代码，用于在测试集上测试模型的性能
以此来进行优化
'''



import logging
import torch
from torch.utils.data import DataLoader
from model_pre import Network

import wandb
from tqdm import tqdm
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *

import open3d as o3d
import os
import numpy as np
from PIL import Image



def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    #接收一个配置文件作为参数
    #cfg:配置文件
    #epoch_idx:epoch的索引
    #test_data_loader:测试数据加载器
    #test_writer:测试数据写入器
    #model:待测试的模型

    torch.backends.cudnn.benchmark = True   #启用cuDNN的benchmark模式，以加速卷积操作

    if test_data_loader is None:
        #创建VIPC测试数据加载器

        ViPC_test = ViPCDataLoader(r'./test_list.txt',
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
        model = Network()#.apply(weights_init_normal) #创建模型
        model.cuda()
    '''
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
    '''

    # Switch models to evaluation mode
    model.eval()#将模型切换到评估模式，不再更新梯度

    n_samples = len(test_data_loader)#测试数据集的样本数

    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])#CD L1、CD L2和F1的平均值
    test_metrics = AverageMeter(['CDl1', 'CDl2', 'F1'])
    category_metrics = dict()



    # 创建保存文件夹
    output_dir = r'../output/PC-complete'
    input_img_dir = r'../output/input-images'
    partial_pc_dir = r'../output/partial-PC'
    gt_pc_dir = r'../output/gt-PC'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_img_dir, exist_ok=True)
    os.makedirs(partial_pc_dir, exist_ok=True)
    os.makedirs(gt_pc_dir, exist_ok=True)


    # Testing loop  开始测试循环
    with tqdm(test_data_loader) as t:
        for model_idx, (view, gt_pc, part_pc ) in enumerate(t):

            with torch.no_grad():#不需要计算梯度

                partial = part_pc.cuda() 
                gt=gt_pc.cuda()
                png=view.cuda()

                partial = farthest_point_sample(partial, 2048)
                gt= farthest_point_sample(gt, 2048)

                model.eval()#不更新梯度
                
                #partial: (4,2048,3)    png:(4,3,224,224)

                recon=model(partial,png)#重建点云
                cdl1,cdl2,f1 = calc_cd(recon,gt,calc_f1=True)

                cdl1=cdl1.mean().item() * 1e3
                cdl2=cdl2.mean().item() * 1e3
                f1=f1.mean().item()

                _metrics=[cdl1,cdl2,f1]
                test_losses.update([cdl1,cdl2,f1])

                test_metrics.update(_metrics)




                # 保存输入的2D图像(最后一轮)
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
    

    #保存补全之后的点云文件
    #output_dir=r'/home/ycx/code/EGIInet-analy/output-PC-complete'
    
    for i, pcd in enumerate(recon):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(output_dir, f'completed_pcd_{model_idx}_{i}.ply')
        
        # 使用 open3d 保存为 .ply 文件
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)   
    


    # Print testing results
    #打印测试结果

    print('============================ TEST RESULTS ============================')


    print('Overall', end='\t\t\t')
    #输出测试结果
    for value in test_metrics.avg():
        print('{:.4f}'.format(value), end='\t')
        #cdl1,cdl2,f1
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    wandb.log({
        "Test/Overall_CDl1": test_losses.avg(0),
        "Test/Overall_CDl2": test_losses.avg(1),
        "Test/Overall_F1": test_losses.avg(2),
        "Epoch": epoch_idx
    })



    return test_losses.avg(0)





def test_net2(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    #接收一个配置文件作为参数
    #cfg:配置文件
    #epoch_idx:epoch的索引
    #test_data_loader:测试数据加载器
    #test_writer:测试数据写入器
    #model:待测试的模型

    torch.backends.cudnn.benchmark = True   #启用cuDNN的benchmark模式，以加速卷积操作

    if test_data_loader is None:
        #创建VIPC测试数据加载器

        ViPC_test = ViPCDataLoader(r'./test_list.txt',
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
        model = Network()#.apply(weights_init_normal) #创建模型
        model.cuda()
    '''
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
    '''

    # Switch models to evaluation mode
    model.eval()#将模型切换到评估模式，不再更新梯度

    n_samples = len(test_data_loader)#测试数据集的样本数

    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])#CD L1、CD L2和F1的平均值
    test_metrics = AverageMeter(['CDl1', 'CDl2', 'F1'])
    category_metrics = dict()



    # 创建保存文件夹
    output_dir = r'../output/PC-complete'
    input_img_dir = r'../output/input-images'
    partial_pc_dir = r'../output/partial-PC'
    gt_pc_dir = r'../output/gt-PC'
    generated_pc_dir = r'../output/generated-PC'
    coarse_pc_dir = r'../output/coarse-PC'
    pred_pc_dir = r'../output/pred-PC'


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_img_dir, exist_ok=True)
    os.makedirs(partial_pc_dir, exist_ok=True)
    os.makedirs(gt_pc_dir, exist_ok=True)
    os.makedirs(generated_pc_dir, exist_ok=True)
    os.makedirs(coarse_pc_dir, exist_ok=True)
    os.makedirs(pred_pc_dir, exist_ok=True)
    


    # Testing loop  开始测试循环
    with tqdm(test_data_loader) as t:
        for model_idx, (view, gt_pc, part_pc ) in enumerate(t):

            with torch.no_grad():#不需要计算梯度

                partial = part_pc.cuda() 
                gt=gt_pc.cuda()
                png=view.cuda()

                partial = farthest_point_sample(partial, 2048)
                gt= farthest_point_sample(gt, 2048)

                model.eval()#不更新梯度
                
                #partial: (4,2048,3)    png:(4,3,224,224)

                recon, coarse,pc_generate, pred_pcds=model(partial,png)#重建点云
                cdl1,cdl2,f1 = calc_cd(recon,gt,calc_f1=True)

                cdl1=cdl1.mean().item() * 1e3
                cdl2=cdl2.mean().item() * 1e3
                f1=f1.mean().item()

                _metrics=[cdl1,cdl2,f1]
                test_losses.update([cdl1,cdl2,f1])

                test_metrics.update(_metrics)




                # 保存输入的2D图像(最后一轮)
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

        #保存生成的点云
    for i, pcd in enumerate(pc_generate):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(generated_pc_dir, f'generated_pcd_{model_idx}_{i}.ply')
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)  

        #保存粗糙点云
    for i, pcd in enumerate(coarse):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(coarse_pc_dir, f'coarse_pcd_{model_idx}_{i}.ply')
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)

        #保存未经处理的点云
    for i, pcd in enumerate(pred_pcds):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(pred_pc_dir, f'pred_pcd_{model_idx}_{i}.ply')
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)
    

    #保存补全之后的点云文件
    #output_dir=r'/home/ycx/code/EGIInet-analy/output-PC-complete'
    
    for i, pcd in enumerate(recon):
        pcd_np = pcd.cpu().numpy()
        output_path = os.path.join(output_dir, f'completed_pcd_{model_idx}_{i}.ply')
        
        # 使用 open3d 保存为 .ply 文件
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        o3d.io.write_point_cloud(output_path, pcd_o3d)   
    


    # Print testing results
    #打印测试结果

    print('============================ TEST RESULTS ============================')


    print('Overall', end='\t\t\t')
    #输出测试结果
    for value in test_metrics.avg():
        print('{:.4f}'.format(value), end='\t')
        #cdl1,cdl2,f1
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    wandb.log({
        "Test/Overall_CDl1": test_losses.avg(0),
        "Test/Overall_CDl2": test_losses.avg(1),
        "Test/Overall_F1": test_losses.avg(2),
        "Epoch": epoch_idx
    })



    return test_losses.avg(0)