import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#  Dynamic Multi-scale depth-wise convolution (D-MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.num_branches = len(kernel_sizes)

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        # Learnable dynamic weights for each branch
        self.weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #(B, C, 1, 1)
            nn.Conv2d(in_channels, self.num_branches, kernel_size=1, bias=False) #(B, num_branches, 1, 1)
        )
        # Fusion conv after concat
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * self.num_branches, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer(self.activation, inplace=True)
        )

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # 多尺度分支输出
        branch_outs = [dw(x) for dw in self.dwconvs]  # List of [B, C, H, W]

        # 全局平均池化后用 1x1 卷积计算权重
        x_pool = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        weights = self.weight_fc(x_pool)      # [B, N, 1, 1]
        weights = F.softmax(weights, dim=1)   # [B, N, 1, 1]
        weights = weights.unsqueeze(-1)       # [B, N, 1, 1, 1]

        # 权重乘以对应分支输出
        weighted_outs = [w * o for w, o in zip(torch.unbind(weights, dim=1), branch_outs)]

        # 拼接 + 融合
        fused = torch.cat(weighted_outs, dim=1)
        # 1x1卷积统一通道
        out = self.fusion_conv(fused)

        return out
    
class ConvAtt(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6',attn_bias=False, proj_drop=0.):
        super(ConvAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False
        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
    
        self.msdc = MSDC(self.in_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
            
        self.qkv = nn.Conv2d(in_channels, 3*in_channels,1, stride=1, padding=0, bias=attn_bias)

        self.dwc = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=in_channels)

        self.proj = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self, x):
        dout = self.msdc(x) 
        q, k, v = self.qkv(dout).chunk(3, dim=1)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out + x

#   Multi-scale convolution Attention (MSCA)

def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
        convs = []
        convatt = ConvAtt(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        convs.append(convatt)
        if n > 1:
            for i in range(1, n):
                convatt = ConvAtt(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation =activation)
                convs.append(convatt)
        conv = nn.Sequential(*convs)
        return conv
        
class EdgeAttention(nn.Module):
    def __init__(self, channels):
        super(EdgeAttention, self).__init__()
        # Sobel边缘检测卷积核（固定权重）
        sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]]], dtype=torch.float32)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x.unsqueeze(0))
        self.register_buffer('sobel_kernel_y', sobel_kernel_y.unsqueeze(0))

        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取强度梯度作为边缘特征
        grad_x = F.conv2d(x, self.sobel_kernel_x.expand(x.size(1), 1, 3, 3), groups=x.size(1), padding=1)
        grad_y = F.conv2d(x, self.sobel_kernel_y.expand(x.size(1), 1, 3, 3), groups=x.size(1), padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edge_feat = self.edge_conv(edge)
        attn_map = self.attn(edge_feat)
        return x * attn_map + x  # 残差增强
        
#   Efficient up-convolution block (EUCB)
# --- 双上采样模块 + 边缘注意力 ---
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels,scale_factor):
        super(EUCB,self).__init__()
        self.factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels


        #self.dwconv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
        '''
        self.up_p = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False)
        )
'''
        self.up_b = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.PReLU(),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=False)
        )
        self.edge_attn = EdgeAttention(out_channels)


    def forward(self, x):   
       # x_p = self.up_p(x)
        x_b = self.up_b(x)

        #out = self.dwconv(torch.cat([x_p, x_b], dim=1))
        #out = self.dwconv(x_b)
        out = self.edge_attn(x_b)  # 加入边缘注意力增强
        return out

#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi
    
#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

#   Efficient multi-scale convolutional attention decoding (EMCAD)
class EMCAD(nn.Module):
    def __init__(self, channels=[768, 384, 192, 96], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCAD,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
	
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1],scale_factor=2)
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=lgag_ks, groups=channels[1]//2)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2],scale_factor=2)
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2, kernel_size=lgag_ks, groups=channels[2]//2)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3],scale_factor=2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.mscb4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)
                
        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.mscb3(d3)
        
        # EUCB2
        d2 = self.eucb2(d3)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.mscb2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.mscb1(d1)
        
        return [d4, d3, d2, d1]
