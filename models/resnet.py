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
    #大修的resnet，谨慎！
    def __init__(self, model, feat_dim=2048, type=None):
        super(ResNet2, self).__init__()
        if type == 'gray':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 提取空间特征的主干网络
        self.spatial_base = nn.Sequential(*list(model.children())[:-3])

        # 多步降维，逐步将通道数降至 256
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 提取全局特征的完整主干网络
        self.resnet = model
        self.resnet.fc = nn.Identity()

        # 全局特征映射头
        self.inv_head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False)
        )

    def forward(self, x):
        # 提取空间特征
        spatial_features = self.spatial_base(x)  # [batch_size, 1024, 14, 14]
        spatial_features = self.spatial_projection(spatial_features)  # [batch_size, 256, 14, 14]
        spatial_features = spatial_features.view(spatial_features.size(0), 256, -1)  # [batch_size, 256, 196]

        # 提取全局特征
        global_features = self.resnet(x)  # [batch_size, feat_dim]
        global_features = self.inv_head(global_features)  # [batch_size, 256]

        return global_features, spatial_features




if __name__ == '__main__':

    img_model = ResNet2(resnet50(), feat_dim = 2048).cuda()
    img = torch.randn(16, 3, 224, 224).cuda()

    # 前向传播
    processed_features, raw_features = img_model(img)
    
    # 打印输出的特征张量
    print("global_features:", processed_features.shape)
    print("spatial_features:", raw_features.shape)


    img_model2=ResNet(resnet50(), feat_dim = 2048).cuda()
    x=img_model2(img)
    print(x.shape)