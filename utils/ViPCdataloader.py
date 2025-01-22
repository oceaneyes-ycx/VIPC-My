#dataloader for ViPC dataset

import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import copy
from datasets import data_utils as d_utils

trans_pc_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),#将点云转化为Tensor
                d_utils.PointcloudNormalize(),#归一化
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),#缩放
                d_utils.PointcloudRotate(),#随机旋转，增强旋转不变性
                d_utils.PointcloudTranslate(0.5, p=1),#平移，增强空间偏移鲁棒性
                d_utils.PointcloudJitter(p=1),#添加噪声
                d_utils.PointcloudRandomInputDropout(p=1),#随机丢弃一些点，应对输入数据的缺失可能
            ])
    
trans_pc_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_img_1 = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace = True)])

trans_img_2 = transforms.Compose([transforms.Grayscale(1),
                                transforms.Resize((224, 224)),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(0.485, 0.229, inplace = True)])



def collate_fn(batch):
    #处理batch数据，lamba函数检查每个元素是否为空，如果为空则删除
    #filter返回一个迭代器，通过list将其转换为列表
    batch = list(filter(lambda x: x is not None, batch))
    #如果batch为空，返回None
    return torch.utils.data.dataloader.default_collate(batch)
    #pytorch默认整理函数，将batch中的数据整理成一个batch


def rotation_z(pts, theta):
    #旋转矩阵,将二维平面上的点绕z轴旋转theta角度
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    #旋转矩阵,将二维平面上的点绕y轴旋转theta角度
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    #旋转矩阵,将二维平面上的点绕x轴旋转theta角度
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T

class ViPCDataLoader(Dataset):
    #DATALOADER，继承自torch.utils.data.Dataset
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        #初始化函数
        #filepath:文件路径,trina_list.txt或者test_list.txt的路径
        #data_path:数据集路径，ShapeNetViPC-Dataset的路径
        #status:train或者test,表示数据集的状态
        #pc_input_num:点云输入的数量，默认为3500
        #view_align:是否对齐视图，默认为False,如果为True,则图像视图和部分点云视图对齐
        #category:类别，默认为all,表示所有类别，如果为其他类别，则只加载该类别的数据
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num#赋值
        self.status = status
        self.filelist = []
        self.cat = []#存储类别的列表
        self.key = []#存储选中的数据的路径
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }#类别映射
        with open(filepath,'r') as f:
            #打开txt文件
            line = f.readline()#读取一行
            while (line):#当line不为空时
                self.filelist.append(line)#将line添加到filelist中，记录文件路径
                line = f.readline()#读取下一行
        '''
        txt文件格式：
        03691459/414fac13be35799d19077496eff9d4b9/10
        04256520/a76a2d34a1aca99611e76965808086c8/09
        02691156/85fd79b9fe5948ff62b600da24e0965/17
        03001627/3427d2276fb713e5864c004309dc898d/17
        04379243/bd3dddce162f59ead92dd8f41946f1b2/17
        03001627/45e2cdf1708680822165c54dcecaacb4/05
        路径格式：类别/模型ID/视图ID
        也因此，filelist中存储的就是上面这样的粗略路径了
        实际上还需要根据种类去对应的文件夹去找到具体的文件
        '''
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        #部分点云路径
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        #GT点云路径
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        #2d视图路径

        for key in self.filelist:
            #遍历filelist(具体的视图对象的路径)
            if category !='all':#如果category不是all的时候，只需要选出指定的那一类就可以了
                if key.split('/')[0]!= self.cat_map[category]:
                    #filelist中被"/"分割后的第一个元素不等于category对应的类别,则跳过
                    continue
            self.cat.append(key.split(';')[0])#将filelist中被";"分割后的第一个元素(类别)添加到cat中
            self.key.append(key)#将filelist中的元素路径添加到key中

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])#图像变换器，将图像resize为224*224，然后转换为张量

        print(f'{status} data num: {len(self.key)}')#打印数据集的数量


    def __getitem__(self, idx):
        #获取数据集中的数据
        #idx:数据集中的索引
        
        key = self.key[idx]#获取key中的第idx个元素，即数据的路径
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        #部分点云路径
        #使用路径拼接的方法，imcomplete_path为部分点云的总体路径，之后读取key中的类别编号，模型ID，视图ID，然后拼接成具体的路径
        #加载的点云文件是.dat文件
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = True代表图像视图和部分点云视图相等，即对应同一个方向采集到的点云和图像
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')#从24个视图中随机选择一个视图
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        #GT点云路径,gt路径+类别编号+模型ID+视图ID+.dat
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        #2D视图路径,rendering路径+类别编号+模型ID+视图ID+.png
        #Inserted to correct a bug in the splitting for some lines 
        #插入以纠正某些行拆分中的错误
        if(len(ran_key.split('/')[-1])>3):
            #检查key中的最后一项(也就是视图的ID)的长度是否大于(有可能模型ID和视图ID之间的分隔符不是"/"导致这俩合并在了一起)
            print("bug")#大于3的时候打印bug
            print(ran_key.split('/')[-1])#打印视图ID
            fin = ran_key.split('/')[-1][-2:]#提取视图ID的最后两位
            interm = ran_key.split('/')[-1][:-2]#提取视图ID的最后两位前面的部分
            
            #重新拼接路径，使用新的视图ID
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            #这里还移除了可能存在的换行符
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))#对2D视图进行变换，resize，转换为tensor
        views = views[:3,:,:]#取前三个通道，即RGB通道
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)#加载GT点云，转换为float32 numpy数组
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)#加载部分点云，转换为float32 numpy数组
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:#如果部分点云的数量小于3500
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]#将部分点云重复，直到数量达到3500


        # load the view metadata
        # 加载视图元数据
        image_view_id = view_path.split('.')[0].split('/')[-1]
        #获取视图ID
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')
        #加载视图元数据，txt文件中存储了视图的角度信息（即位姿信息）

        theta_part = math.radians(view_metadata[int(part_view_id),0])#部分点云的角度信息转化为弧度
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)#旋转部分点云，使其与视图对齐
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        # 将部分点云和GT归一化到相同的尺度
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        # 返回：视图，GT点云，部分点云
        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.key)

class ViPCDataLoader2(Dataset):
    #DATALOADER，继承自torch.utils.data.Dataset
    #用于对比学习使用的自定义dataloader
    #2号dataloader,添加了点云和图像的数据增强模块,专门用于对比学习使用
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        #初始化函数
        #filepath:文件路径,trina_list.txt或者test_list.txt的路径
        #data_path:数据集路径，ShapeNetViPC-Dataset的路径
        #status:train或者test,表示数据集的状态
        #pc_input_num:点云输入的数量，默认为3500
        #view_align:是否对齐视图，默认为False,如果为True,则图像视图和部分点云视图对齐
        #category:类别，默认为all,表示所有类别，如果为其他类别，则只加载该类别的数据
        super(ViPCDataLoader2,self).__init__()
        self.pc_input_num = pc_input_num#赋值
        self.status = status
        self.filelist = []
        self.cat = []#存储类别的列表
        self.key = []#存储选中的数据的路径
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }#类别映射
        with open(filepath,'r') as f:
            #打开txt文件
            line = f.readline()#读取一行
            while (line):#当line不为空时
                self.filelist.append(line)#将line添加到filelist中，记录文件路径
                line = f.readline()#读取下一行
        '''
        txt文件格式：
        03691459/414fac13be35799d19077496eff9d4b9/10
        04256520/a76a2d34a1aca99611e76965808086c8/09
        02691156/85fd79b9fe5948ff62b600da24e0965/17
        03001627/3427d2276fb713e5864c004309dc898d/17
        04379243/bd3dddce162f59ead92dd8f41946f1b2/17
        03001627/45e2cdf1708680822165c54dcecaacb4/05
        路径格式：类别/模型ID/视图ID
        也因此，filelist中存储的就是上面这样的粗略路径了
        实际上还需要根据种类去对应的文件夹去找到具体的文件
        '''
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        #部分点云路径
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        #GT点云路径
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        #2d视图路径

        for key in self.filelist:
            #遍历filelist(具体的视图对象的路径)
            if category !='all':#如果category不是all的时候，只需要选出指定的那一类就可以了
                if key.split('/')[0]!= self.cat_map[category]:
                    #filelist中被"/"分割后的第一个元素不等于category对应的类别,则跳过
                    continue
            self.cat.append(key.split(';')[0])#将filelist中被";"分割后的第一个元素(类别)添加到cat中
            self.key.append(key)#将filelist中的元素路径添加到key中

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])#图像变换器，将图像resize为224*224，然后转换为张量

        print(f'{status} data num: {len(self.key)}')#打印数据集的数量


    def __getitem__(self, idx):
        #获取数据集中的数据
        #idx:数据集中的索引
        
        key = self.key[idx]#获取key中的第idx个元素，即数据的路径
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        #部分点云路径
        #使用路径拼接的方法，imcomplete_path为部分点云的总体路径，之后读取key中的类别编号，模型ID，视图ID，然后拼接成具体的路径
        #加载的点云文件是.dat文件
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = True代表图像视图和部分点云视图相等，即对应同一个方向采集到的点云和图像
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')#从24个视图中随机选择一个视图
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        #GT点云路径,gt路径+类别编号+模型ID+视图ID+.dat
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        #2D视图路径,rendering路径+类别编号+模型ID+视图ID+.png
        #Inserted to correct a bug in the splitting for some lines 
        #插入以纠正某些行拆分中的错误
        if(len(ran_key.split('/')[-1])>3):
            #检查key中的最后一项(也就是视图的ID)的长度是否大于(有可能模型ID和视图ID之间的分隔符不是"/"导致这俩合并在了一起)
            print("bug")#大于3的时候打印bug
            print(ran_key.split('/')[-1])#打印视图ID
            fin = ran_key.split('/')[-1][-2:]#提取视图ID的最后两位
            interm = ran_key.split('/')[-1][:-2]#提取视图ID的最后两位前面的部分
            
            #重新拼接路径，使用新的视图ID
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            #这里还移除了可能存在的换行符
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))#对2D视图进行变换，resize，转换为tensor
        views = views[:3,:,:]#取前三个通道，即RGB通道
        # load GT points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)#加载GT点云，转换为float32 numpy数组
        # load partial points
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)#加载部分点云，转换为float32 numpy数组
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:#如果部分点云的数量小于3500
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]#将部分点云重复，直到数量达到3500


        # load the view metadata
        # 加载视图元数据
        image_view_id = view_path.split('.')[0].split('/')[-1]
        #获取视图ID
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')
        #加载视图元数据，txt文件中存储了视图的角度信息（即位姿信息）

        theta_part = math.radians(view_metadata[int(part_view_id),0])#部分点云的角度信息转化为弧度
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)#旋转部分点云，使其与视图对齐
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        # 将部分点云和GT归一化到相同的尺度
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        pc1=trans_pc_1(pc)#用于对比学习的1号点云，原GT点云经过随机处理
        pc2=trans_pc_2(pc)#用于对比学习的2号点云

        # 返回：视图，GT点云，部分点云
        #return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()
        #返回：视图，对比学习的1号点云，对比学习的2号点云
        return views.float(), pc1, pc2

    def __len__(self):
        return len(self.key)
    

class ViPCDataLoader3(Dataset):
    #3号dataloader, 同时返回图像，GT点云和部分点云，经过数据增强的点云
    #DATALOADER，继承自torch.utils.data.Dataset
    #用于对比学习使用的自定义dataloader
    #2025/1/20：经检查这个dataloarder代码有问题，输出的cd点云似乎不对，需要进一步检查
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        #初始化函数
        #filepath:文件路径,trina_list.txt或者test_list.txt的路径
        #data_path:数据集路径，ShapeNetViPC-Dataset的路径
        #status:train或者test,表示数据集的状态
        #pc_input_num:点云输入的数量，默认为3500
        #view_align:是否对齐视图，默认为False,如果为True,则图像视图和部分点云视图对齐
        #category:类别，默认为all,表示所有类别，如果为其他类别，则只加载该类别的数据
        super(ViPCDataLoader3,self).__init__()
        self.pc_input_num = pc_input_num#赋值
        self.status = status
        self.filelist = []
        self.cat = []#存储类别的列表
        self.key = []#存储选中的数据的路径
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }#类别映射
        with open(filepath,'r') as f:
            #打开txt文件
            line = f.readline()#读取一行
            while (line):#当line不为空时
                self.filelist.append(line)#将line添加到filelist中，记录文件路径
                line = f.readline()#读取下一行
        '''
        txt文件格式：
        03691459/414fac13be35799d19077496eff9d4b9/10
        04256520/a76a2d34a1aca99611e76965808086c8/09
        02691156/85fd79b9fe5948ff62b600da24e0965/17
        03001627/3427d2276fb713e5864c004309dc898d/17
        04379243/bd3dddce162f59ead92dd8f41946f1b2/17
        03001627/45e2cdf1708680822165c54dcecaacb4/05
        路径格式：类别/模型ID/视图ID
        也因此，filelist中存储的就是上面这样的粗略路径了
        实际上还需要根据种类去对应的文件夹去找到具体的文件
        '''
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        #部分点云路径
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        #GT点云路径
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        #2d视图路径

        for key in self.filelist:
            #遍历filelist(具体的视图对象的路径)
            if category !='all':#如果category不是all的时候，只需要选出指定的那一类就可以了
                if key.split('/')[0]!= self.cat_map[category]:
                    #filelist中被"/"分割后的第一个元素不等于category对应的类别,则跳过
                    continue
            self.cat.append(key.split(';')[0])#将filelist中被";"分割后的第一个元素(类别)添加到cat中
            self.key.append(key)#将filelist中的元素路径添加到key中

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])#图像变换器，将图像resize为224*224，然后转换为张量

        print(f'{status} data num: {len(self.key)}')#打印数据集的数量


    def __getitem__(self, idx):
        #获取数据集中的数据
        #idx:数据集中的索引
        
        key = self.key[idx]#获取key中的第idx个元素，即数据的路径
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        #部分点云路径
        #使用路径拼接的方法，imcomplete_path为部分点云的总体路径，之后读取key中的类别编号，模型ID，视图ID，然后拼接成具体的路径
        #加载的点云文件是.dat文件
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = True代表图像视图和部分点云视图相等，即对应同一个方向采集到的点云和图像
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')#从24个视图中随机选择一个视图
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        #GT点云路径,gt路径+类别编号+模型ID+视图ID+.dat
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        #2D视图路径,rendering路径+类别编号+模型ID+视图ID+.png
        #Inserted to correct a bug in the splitting for some lines 
        #插入以纠正某些行拆分中的错误
        if(len(ran_key.split('/')[-1])>3):
            #检查key中的最后一项(也就是视图的ID)的长度是否大于(有可能模型ID和视图ID之间的分隔符不是"/"导致这俩合并在了一起)
            print("bug")#大于3的时候打印bug
            print(ran_key.split('/')[-1])#打印视图ID
            fin = ran_key.split('/')[-1][-2:]#提取视图ID的最后两位
            interm = ran_key.split('/')[-1][:-2]#提取视图ID的最后两位前面的部分
            
            #重新拼接路径，使用新的视图ID
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            #这里还移除了可能存在的换行符
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))#对2D视图进行变换，resize，转换为tensor
        views = views[:3,:,:]#取前三个通道，即RGB通道
        # load GT points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)#加载GT点云，转换为float32 numpy数组
        # load partial points
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)#加载部分点云，转换为float32 numpy数组
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:#如果部分点云的数量小于3500
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]#将部分点云重复，直到数量达到3500


        # load the view metadata
        # 加载视图元数据
        image_view_id = view_path.split('.')[0].split('/')[-1]
        #获取视图ID
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')
        #加载视图元数据，txt文件中存储了视图的角度信息（即位姿信息）

        theta_part = math.radians(view_metadata[int(part_view_id),0])#部分点云的角度信息转化为弧度
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)#旋转部分点云，使其与视图对齐
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        # 将部分点云和GT归一化到相同的尺度
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        
        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        pc1=trans_pc_1(copy.deepcopy(pc))#用于对比学习的1号点云，原GT点云经过随机处理
        pc2=trans_pc_2(copy.deepcopy(pc))#用于对比学习的2号点云

        # 返回：视图，GT点云，部分点云
        #return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()
        #返回：视图，GT 点云, 部分点云, 对比学习的1号点云，对比学习的2号点云
        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float(), pc1, pc2

    def __len__(self):
        return len(self.key)




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    category = "table"
    ViPCDataset = ViPCDataLoader2('test_list.txt',data_path='/home/ycx/dataset/ShapeNetViPC/ShapeNetViPC-Dataset',status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    #for image, gt, partial in tqdm(train_loader):
        
        #print(image.shape)
        
        #pass
    for image,pc1,pc2 in tqdm(train_loader):
        print(image.shape)
        print(pc1.shape)
        print(pc2.shape)
        pass
    