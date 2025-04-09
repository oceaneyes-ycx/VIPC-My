import pytorch3d
import torch
from torch import nn as nn
from utils.ViPCdataloader import ViPCDataLoader2
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import random
import os
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings
)

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)


from render.core.cloud import PointClouds3D

def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, black)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, white)
    
    return final_conv

def rgb2gray(rgb):
    return torch.inner(rgb[...,:3], torch.tensor([0.2989, 0.5870, 0.1140]).to(rgb.device))


def create_lights():

    lights = DirectionalLights(device=device)

    return lights


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


opt = params()


class Renderer(nn.Module):

    """A class for rendering a batch of points in a differentiable manner.
     It combines rasterization, lightning and compositing. This class is based on Pytorch3D differentiable rendering."""
    #一个渲染器类，作用是基于pytorch3d实现的可微分渲染，主要功能是对一批点云数据进行渲染，结合光栅化，光照和合成操作生成图像

    def __init__(self, colors):
        #colors:点云的颜色特征
        super().__init__()
        self.image_size = opt.image_size
        #渲染的图像的大小，从opt全局配置文件中读取
        self.radius = opt.radius
        #点的半径，用于光栅化
        self.points_per_pixel = opt.points_per_pixel
        #每个像素中允许的最大点数
        self.dist = opt.dist#摄像机与场景的距离
        self.compositor = AlphaCompositor()
        #合成器，这里使用的pytorch3d中的alphacompositor，基于alpha通道进行合成
        self.colors = colors

    def forward(self, points, eye):
        #points:点云数据，eye:相机位置
        point_cloud = PointClouds3D(points=points, features=self.colors)
        #使用PointClouds3D类来创建点云对象，包含点的坐标和颜色特征
        R, T = look_at_view_transform(eye=eye)
        #根据相机位置计算旋转矩阵R和平移向量T
        cameras = FoVPerspectiveCameras(R=R, T=T, device=points.device)
        #创建透视相机对象，使用R和T来设置相机的视角和位置
       

        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size, radius=self.radius, points_per_pixel=self.points_per_pixel)
        #设置光栅化参数，包括图像大小，点的半径和每个像素的点数
        
        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings)
        #创建光栅化器对象，使用相机和光栅化参数来设置渲染的视角和细节
        #负责将点云投影到图像平面。
        renderer = PointsRenderer(
            rasterizer=rasterizer, compositor = self.compositor)
        #使用光栅化器和合成器创建渲染器对象（也就是最终的渲染图像）

        images = renderer(point_cloud)
        #使用渲染器对点云进行渲染，返回渲染后的图像
        
        return images











if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ViPCDataset_train = ViPCDataLoader(
        'train_list2.txt', data_path=opt.dataroot, status="train", category="plane")
    train_loader = DataLoader(ViPCDataset_train,
                              batch_size=16,
                              num_workers=opt.nThreads,
                              shuffle=True,
                              drop_last=True)
    
    renderer = Renderer().to(device)

    for data in tqdm(train_loader):
       
       
        break
