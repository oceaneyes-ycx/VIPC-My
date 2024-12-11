#草稿纸


import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet18,resnet101
import torch



class ResNet(nn.Module):
    #crossnet原版resnet
    def __init__(self, model, feat_dim = 2048, type = None):
        super(ResNet, self).__init__()
        if type == 'gray':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet = model
        self.resnet.fc = nn.Identity()
        
        self.inv_head = nn.Sequential(
                            nn.Linear(feat_dim, 512, bias = False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256, bias = False)
                            ) 
        
    def forward(self, x):
        # print(self.resnet)
        x = self.resnet(x)
        x = self.inv_head(x)
        return x
    

class ResNet2(nn.Module):
    #resnet2 版本，使用xmnet的方式输出了空间特征，同时修订了全局特征的提取方式
    #采用了crossnet的逆向特征头的方式，此类调用时只可以使用resnet18,同时feat_dim设定为512!
    def __init__(self, model, feat_dim = 2048, type = None):
        super(ResNet2, self).__init__()
        
        # 修改第一层卷积以适应灰度图像
        if type == 'gray':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 保留所有层以获取全局特征
        self.resnet = model
        
        # 去掉最后的全连接层
        self.resnet.fc = nn.Identity()
        
        # 添加inv_head模块
        self.inv_head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False)
        )
        
        # 提取空间特征的层
        self.base_layers = nn.Sequential(*list(model.children())[:-3])
        self.spatial_features_size = 256  # layer3输出通道数为256

    def forward(self, x):
        #x:[16, 3, 224, 224]

        # 获取空间特征
        spatial_features = self.base_layers(x)  # shape: (16, 256, 14, 14)

        spatial_features = spatial_features.view(x.size(0), self.spatial_features_size, -1)  # shape: (batch_size, 256, H*W)
        #[16, 256, 196]   需要的空间特征
        # 继续前向传播到全局特征
        global_features = self.resnet(x)  # shape: (batch_size, 512)     (16,512)
        global_features = self.inv_head(global_features)  # shape: (batch_size, 256)
        
        return spatial_features, global_features




if __name__ == '__main__':

    img_model = ResNet2(resnet18(), feat_dim = 512).cuda()
    img = torch.randn(16, 3, 224, 224).cuda()

    # 前向传播
    processed_features, raw_features = img_model(img)
    
    # 打印输出的特征张量
    print("global_features:", processed_features.shape)
    print("spatial_features:", raw_features.shape)


    #img_model2=ResNet(resnet50(), feat_dim = 2048).cuda()
    #x=img_model2(img)
    #print(x.shape)