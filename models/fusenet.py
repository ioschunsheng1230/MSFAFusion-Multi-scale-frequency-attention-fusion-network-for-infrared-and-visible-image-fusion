import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fourier import MultiSpectralAttentionLayer
from models.attn_module import  PatchEmbed, DePatch, Block
from models.MCF_module import MCF


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction) # 8
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(inp,oup,1,1,0)
    def forward(self, x):
        identity = x
        identity = self.conv(identity)
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
        return out


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        # 每个像素进行归一化
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


# class MBSConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=(1, 3, 5)):
#         super(MBSConv, self).__init__()
#         self.depthwise_convs = nn.ModuleList([
#             nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
#             for k in kernel_sizes
#         ])
#         self.pointwise_conv = nn.Conv2d(in_channels * len(kernel_sizes), out_channels, kernel_size=1)

#     def forward(self, x):
#         features = []
#         for conv in self.depthwise_convs:
#             feature = conv(x)
#             features.append(feature)
#         x = torch.cat(features, dim=1)
#         x = self.pointwise_conv(x)
#         return x
    
class MDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(1, 3, 5)):
        super(MDSConv, self).__init__()
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in kernel_sizes
        ])
        self.pointwise_conv = nn.Conv2d(in_channels * len(kernel_sizes), out_channels, kernel_size=1)

    def forward(self, x):
        features = []
        for conv in self.depthwise_convs:
            feature = conv(x)
            features.append(feature)
        x = torch.cat(features, dim=1)
        x = self.pointwise_conv(x)
        return x


class MDSCBlock(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(MDSCBlock, self).__init__()
        self.mbs = MDSConv(in_channels,out_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,1,1,0)
        self.CA = CoordAtt(in_channels*4,out_channels*2)
        self.pixel = PixelNorm()
        self.conv_1 = nn.Conv2d(in_channels,out_channels*2,1,1,0)
        
    def forward(self,x):
        residual = x
        x1 = self.mbs(self.mbs(self.mbs(self.mbs(x))))
        x2 = self.conv(self.mbs(self.mbs(x)))
        x3 = self.conv(self.mbs(x))
        x4 = self.conv(x)
        out = torch.cat((x1,x2,x3,x4), dim=1) # c=32
        residual = self.conv_1(residual)
        out = self.CA(out)
        out = self.pixel(out)
        out = out + residual
        out = F.leaky_relu(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())

        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class CroAttSpaFusion(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super(CroAttSpaFusion, self).__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)
        self.patchembed2 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

        self.QKV_Block1 = Block(dim=dim, num_heads=num_heads)
        self.QKV_Block2 = Block(dim=dim, num_heads=num_heads)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.depatch = DePatch(channel=channel, embed_dim=dim, patch_size=patch_size)
        
    def forward(self, in_1, in_2):
        # Patch Embeding1
        in_emb1 = self.patchembed1(in_1)
        B, N, C = in_emb1.shape

        # cross self-attention Feature Extraction
        _, q1, k1, v1 = self.QKV_Block1(in_emb1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # Patch Embeding2
        in_emb2 = self.patchembed2(in_2)

        _, q2, k2, v2 = self.QKV_Block2(in_emb2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # cross attention
        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

        out = in_1 + in_2 + out1

        return out

def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
 
 
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
 
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to(device)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width).to(device)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
 
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1).to(device)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3).to(device)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
  

class FusionModule1(nn.Module):
     def __init__(self, patch_size=16, dim=16, num_heads=8, channels=16, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(FusionModule1, self).__init__()

        self.vifilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        self.irfilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        
        # Fusion Block
        self.CASFusion = CroAttSpaFusion(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                         proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                         attn_drop=attn_drop)

        self.MCF = MCF(dim, num_heads, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim,dim//2,1,1,0)


     def forward(self, img1, img2):

        x = img1
        y = img2
        x_decoder = self.vifilter(x)
        y_decoder = self.irfilter(y)

        # x_decoder = x
        # y_decoder = y        
        
        feature_x = self.CASFusion(x, y) # Spatial
        feature_y = x_decoder + y_decoder
        z = self.MCF(feature_x , feature_y)
        return z

class FusionModule2(nn.Module):
     def __init__(self, patch_size=16, dim=32, num_heads=8, channels=32, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(FusionModule2, self).__init__()

        self.vifilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        self.irfilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        
        # Fusion Block
        self.CASFusion = CroAttSpaFusion(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                         proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                         attn_drop=attn_drop)
        self.MCF = MCF(dim, num_heads, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim,dim//2,1,1,0)


     def forward(self, img1, img2):

        x = img1
        y = img2
        x_decoder = self.vifilter(x)
        y_decoder = self.irfilter(y)

        # x_decoder = x
        # y_decoder = y        
        
        feature_x = self.CASFusion(x, y) # Spatial
        feature_y = x_decoder + y_decoder
        z = self.MCF(feature_x , feature_y)
        return z



class FusionModule3(nn.Module):
     def __init__(self, patch_size=16, dim=64, num_heads=8, channels=64, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(FusionModule3, self).__init__()

        self.vifilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        self.irfilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        
        # Fusion Block
        self.CASFusion = CroAttSpaFusion(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                         proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                         attn_drop=attn_drop)

        self.MCF = MCF(dim, num_heads, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim,dim//2,1,1,0)


     def forward(self, img1, img2):

        x = img1
        y = img2
        x_decoder = self.vifilter(x)
        y_decoder = self.irfilter(y)

        # x_decoder = x
        # y_decoder = y        
        
        feature_x = self.CASFusion(x, y) # Spatial
        feature_y = x_decoder + y_decoder
        z = self.MCF(feature_x , feature_y)
        return z

     
class FusionModule4(nn.Module):
     def __init__(self, patch_size=16, dim=128, num_heads=8, channels=128, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(FusionModule4, self).__init__()

        self.vifilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        self.irfilter = MultiSpectralAttentionLayer(channel=channels, dct_h=7, dct_w=7, reduction = 16, freq_sel_method = 'top16')
        
        # Fusion Block
        self.CASFusion = CroAttSpaFusion(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                         proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                         attn_drop=attn_drop)
        self.MCF = MCF(dim, num_heads, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim,dim//2,1,1,0)


     def forward(self, img1, img2):

        x = img1
        y = img2
        x_decoder = self.vifilter(x)
        y_decoder = self.irfilter(y)

        # x_decoder = x
        # y_decoder = y        
        
        feature_x = self.CASFusion(x, y) # Spatial
        feature_y = x_decoder + y_decoder
        z = self.MCF(feature_x , feature_y)
        return z

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class IFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels):        
        super(IFP, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        x = (self.conv_block(x) + 1) / 2
        return x

class DSRM(nn.Module):
    def __init__(self, in_channel=256, out_channel=256):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.block4 = nn.Sequential(
            BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out


class Fusenet(nn.Module):
    def __init__(self, input_nc=1):
        super(Fusenet, self).__init__()

        kernel_size = 1
        stride = 1 
       
        
        self.mbsc1 = MDSCBlock(in_channels=8,out_channels=8)
        self.mbsc2 = MDSCBlock(in_channels=16,out_channels=16)
        self.mbsc3 = MDSCBlock(in_channels=32,out_channels=32)
        self.mbsc4 = MDSCBlock(in_channels=64,out_channels=64)
        
        
        self.conv_in1 = ConvLayer(input_nc, out_channels=8, kernel_size=3, stride=1)
        
        
        self.dense = DSRM(128,128)
        self.pred_fusion = IFP([128, 64])
        self.pred_fusion1 = IFP([64, 32])
        self.fusion = IFP([32, 1])
    
        self.fourierFusion1 = FusionModule1(patch_size=16, dim=16, num_heads=8, channels=16, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)
        self.fourierFusion2 = FusionModule2(patch_size=16, dim=32, num_heads=8, channels=32, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)
        self.fourierFusion3 = FusionModule3(patch_size=16, dim=64, num_heads=8, channels=64, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)
        self.fourierFusion4 = FusionModule4(patch_size=16, dim=128, num_heads=8, channels=128, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)
        self.conv1 = nn.Conv2d(256,128,1,1,0)

        self.conv_1 = nn.Conv2d(240,128,1,1,0)

       
    def forward(self, vi, ir):
        ir = self.conv_in1(ir)
        vi = self.conv_in1(vi)
        
        x1 = self.mbsc1(vi)
        x2 = self.mbsc2(x1)
        x3 = self.mbsc3(x2)
        x4 = self.mbsc4(x3)

        y1 = self.mbsc1(ir)
        y2 = self.mbsc2(y1)
        y3 = self.mbsc3(y2)
        y4 = self.mbsc4(y3)

        fusion1 = self.fourierFusion1(x1,y1)
        fusion2 = self.fourierFusion2(x2,y2)
        fusion3 = self.fourierFusion3(x3,y3)
        fusion4 = self.fourierFusion4(x4,y4)
        
        fusion = self.conv_1(torch.cat([fusion1,fusion2,fusion3,fusion4],dim=1))
        
        fusion_vi = fusion * x4
        fusion_ir = fusion * y4

        f = self.conv1(torch.cat((fusion_ir,fusion_vi),dim=1))
        x = self.dense(f)
        x = self.pred_fusion(x)
        x = self.pred_fusion1(x)
        x = self.fusion(x)
        
        return x

