"""VGTNet: view-guided geometric transformation for point cloud completion.

Adapted from the research implementation SymmCompletion3. See THIRD_PARTY_NOTICES.md.
The historical lstnet/sgformer/corssformer attribute spellings are retained so that
existing state_dict checkpoints can be loaded without renaming their parameters.
"""
import torch
from torch import nn, einsum
from torchvision.models import resnet18
from extensions.chamfer_dist import ChamferDistanceL1
from models.model_utils import (
    MLP_CONV, Transformer, PointNet_SA_Module_KNN, ImageSelfAttentionBlock,
    fps_subsample as farthest_point_sample,
)

class CrossFormer(nn.Module):

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(negative_slope=0.2), nn.Linear(dim, out_dim))

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x


def gram_matrix(tensor):
    gram = einsum('b i n, b n j  -> b i j', tensor.transpose(1, 2), tensor)
    return gram


class ImageEncoder2(nn.Module):

    def __init__(self, out_dim=128, num_local_tokens=512):
        super().__init__()
        backbone = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)
        self.local_proj = nn.Conv2d(512, out_dim, kernel_size=1)
        self.local_upsample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=False)
        self.num_local_tokens = num_local_tokens

    def forward(self, x):
        feat_map = self.feature_extractor(x)
        global_feat = self.pool(feat_map).squeeze(-1).squeeze(-1)
        global_feat = self.fc(global_feat)
        local_feat = self.local_proj(feat_map)
        local_feat = self.local_upsample(local_feat)
        local_feat = local_feat.flatten(2)
        return (global_feat, local_feat)


class CSTNet(nn.Module):
    """CGTNet: image-guided affine transformation of 512 point keypoints."""

    def __init__(self, out_dim=512, img_feat_dim=128):
        super(CSTNet, self).__init__()
        self.img_encoder = ImageEncoder2(out_dim=img_feat_dim)
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.transformer_2 = ImageSelfAttentionBlock(128)
        self.cross_attn = Fusion(in_channel=128)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])
        self.mlp = nn.Sequential(nn.Linear(512 * 2, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 9 + 3))

    def forward(self, point_cloud, image):
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud
        img_feat_G, img_feat_L = self.img_encoder(image)
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)
        keyfeatures = self.transformer_1(keyfeatures, keypoints)
        img_feat = img_feat_L.transpose(1, 2).contiguous()
        img_feat = self.transformer_2(img_feat)
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()
        keyfeatures0 = keyfeatures
        img_feat0 = img_feat
        keyfeatures = self.cross_attn(keyfeatures, img_feat)
        img_feat = self.cross_attn(img_feat, keyfeatures0)
        im_target_gram = gram_matrix(img_feat)
        im_target_gram0 = gram_matrix(img_feat0)
        pc_target_gram = gram_matrix(keyfeatures)
        pc_target_gram0 = gram_matrix(keyfeatures0)
        style_transfer_loss = torch.mean((im_target_gram - pc_target_gram0) ** 2 + (pc_target_gram - im_target_gram0) ** 2) / (128 * 512)
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()
        img_feat = img_feat.transpose(1, 2).contiguous()
        img_feat0 = img_feat0.transpose(1, 2).contiguous()
        feat = self.expanding(keyfeatures)
        feat = feat.transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(2), 1)], dim=-1)
        ret = self.mlp(feat)
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)
        return (coarse, symmetry_points, keyfeatures, img_feat0, style_transfer_loss, keypoints)


class Fusion(nn.Module):

    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)

    def forward(self, feat_x, feat_y):
        feat = self.corssformer_1(feat_x, feat_y)
        feat = self.corssformer_2(feat, feat)
        return feat


class SGFormer2(nn.Module):
    """Image/geometry-guided refinement of the coarse cloud to 4096 points."""

    def __init__(self, gf_dim=512, img_feat_dim=128, up_factor=2, attn_dim=256):
        super(SGFormer2, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_4 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.cross_attn_partial = CrossFormer(dim=img_feat_dim, out_dim=attn_dim)
        self.cross_attn_symmetry = CrossFormer(dim=img_feat_dim, out_dim=attn_dim)
        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)
        self.fusion_4 = Fusion(in_channel=256)
        self.fusion_5 = Fusion(in_channel=256)
        self.mlp_fusion = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(negative_slope=0.2), nn.Linear(512, 512))
        self.fusion_3 = Fusion(in_channel=512)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(negative_slope=0.2), nn.Linear(512, 128), nn.LeakyReLU(negative_slope=0.2), nn.Linear(128, 3 * self.up_factor))
        self.expand_img_feat = nn.Sequential(nn.Linear(128, 256), nn.LeakyReLU(0.2))

    def forward(self, coarse, symmetry_feat, partial_feat, img_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)
        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)
        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()
        img_feat_exp = self.expand_dim_4(img_feat)
        img_feat_exp = img_feat_exp.transpose(2, 1).contiguous()
        feat_partial_attn = partial_feat
        feat = self.fusion_1(feat, img_feat_exp)
        feat_symmetry_attn = self.fusion_4(symmetry_feat, img_feat_exp)
        feat_p = self.fusion_5(feat, feat_partial_attn)
        feat_s = self.fusion_2(feat, feat_symmetry_attn)
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        return pcd_up


class local_encoder(nn.Module):

    def __init__(self, out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)
        return feat


class VGTNet(nn.Module):
    """Paper model with historical submodule names for checkpoint compatibility."""

    def __init__(self):
        super().__init__()
        self.lstnet = CSTNet(out_dim=512)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer2(gf_dim=512, up_factor=4)
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, rets, gt, view=None):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total

    def forward(self, point_cloud, view):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        coarse, symmetry_points, keyfeatures, img_feat, style_loss, keypoints = self.lstnet(point_cloud.transpose(2, 1).contiguous(), view)
        feat_symmetry = self.local_encoder(symmetry_points)
        feat_partial = keyfeatures
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial, img_feat)
        rets = [coarse.transpose(2, 1).contiguous(), fine1]
        self.pred_dense_point = rets[-1]
        complete = farthest_point_sample(rets[-1], 2048)
        rets.append(complete)
        mid = [keypoints, symmetry_points, coarse, fine1, complete]
        return (rets, complete, style_loss, mid)
