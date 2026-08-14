# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
import numpy as np
from lib.UniRepLKNet_AFF import unireplknet_s
#from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD


class EMCADNet(nn.Module):
    def __init__(self,  embed_dim=512, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='unireplknet_s', pretrain=True, pretrained_dir='/root/EMCAD-main/pretrained_pth/pvt/'):
        super(EMCADNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if  encoder == 'unireplknet_s':
            self.backbone = unireplknet_s()
            path = pretrained_dir + 'unireplknet_s.pth'
            channels=[768, 384, 192, 96]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder unireplknet_s.')
            self.backbone = unireplknet_s()  
            path = pretrained_dir + 'unireplknet_s.pth'
            channels=[768, 384, 192, 96]
            
        if pretrain==True and 'unireplknet' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, embed_dim=embed_dim, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
    def forward(self, x, mode='test'):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # encoder
        x1, x2, x3, x4 = self.backbone(x)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
        dec_outs = self.decoder(x1, x2, x3, x4)
        
        # prediction heads  
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        if mode == 'test':
            return [p4, p3, p2, p1]
        
        return [p4, p3, p2, p1]
               
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    #print("Total params: %.2fM" % (total/1e6))
    #logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练参数量：", total_trainable)


        
if __name__ == '__main__':
    model = EMCADNet().cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    P = model(input_tensor)

    flops = cal_params_flops(model,input_tensor, 'model_name')

    print(f"Total Model FLOPs: %.3f GFLOPs", flops)

    # 测试推理速度
    run_time = []
    for _ in range(100):  # 跑 100 次取平均
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            P = model(input_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        run_time.append(end - start)

    # 去掉第一次结果（避免冷启动干扰）
    run_time.pop(0)

    # 统计推理时间
    mean_time = np.mean(run_time)
    fps = 1 / mean_time
    print(f"Mean inference time: {mean_time:.6f} seconds")
    print(f"FPS: {fps:.2f}")


    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())

