import sys
import time
import math
import random
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# sys.path.append('models/backbone/')
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU
from unet.unet_vae import UNet_VAE_old
from unet.unet_vae_RQ_scheme1_encoder import UNet_VAE_RQ_scheme1_encoder


class DPC_RNN_UNet(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size,device,num_seq=8, seq_len=5, pred_step=3, network='resnet50', model_weight='',freeze=False):
        super(DPC_RNN_UNet, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        _, self.param = select_resnet('resnet18', track_running_stats=False)
        
        if network == "unet-vae" or network == "rqunet-vae-encoder":
            self.param['feature_size'] = 960
            if network == "unet-vae":
                self.backbone = UNet_VAE_old(num_classes=13,segment=False,in_channels=13,depth=5,is_encoder=True)
            else:
                self.backbone = UNet_VAE_RQ_scheme1_encoder(num_classes=3,segment=False,in_channels=3,depth=5,is_encoder=True,alpha = 0.9)

            encoder_dict = self.backbone.state_dict()
            if model_weight is not None:
                requires_grad = not freeze
                pretrained_dict = torch.load(model_weight, map_location=device)
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
                # 2. overwrite entries in the existing state dict
                encoder_dict.update(pretrained_dict) 
                # 3. load the new state dict
                self.backbone.load_state_dict(encoder_dict)
                for param in self.backbone.parameters():
                    param.requires_grad = requires_grad
        else:
            self.backbone, self.param = select_resnet(network, track_running_stats=False)

        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU # 1024 for resnet50, 256 for resnet18

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=128,
                               kernel_size=3,
                               num_layers=self.param['num_layers'])

        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )

        self.segment_head = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        # self._initialize_weights(self.network_pred)

    def freeze(self, freeze: bool) -> None:
        """Freeze UNet-VAE weights excluding classifier"""
        requires_grad = not freeze

        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        # print(f'block shape: {block.shape}')
        (B, N, SL, C, H, W) = block.shape
        # block = block.view(B*N, C, SL, H, W)

        block = rearrange(block, "b n sl c h w -> (b n sl) c h w")

        ### encoder
        feature = self.backbone(block)
        # _, feature = self.backbone(block)

        del block

        # print(f'feature shape: {feature.shape}')

        # feature_inf_all = torch.reshape(feature, (B,N,SL,C,feature.shape[2],feature.shape[3]))

        feature = self.relu(feature) # [0, +inf)

        # print(f'feature after relu shape: {feature.shape}')

        feature = torch.reshape(feature, (B,N,SL,feature.shape[1],feature.shape[2],feature.shape[3]))

        # feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()

        # print(f'feature_inf_all shape: {feature_inf_all.shape}')
        # del feature_inf_all

        
        ### aggregate, predict future ###

        # feats = feature[:, 0:N-self.pred_step, :].contiguous()
        feats = rearrange(feature, "b n sl c h w -> (b n) sl c h w")

        # feats = feature[:, 0:N-self.pred_step, :].contiguous() # need to reshape
        # print(f'feats shape: {feats.shape}')

        c_t, hidden = self.agg(feats)
        
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

        # print(f'feature shape: {feature.shape}')
        # print(f'feats shape: {feats.shape}')
        # print(f'hidden shape: {hidden.shape}')
        # print(f'c_t shape: {c_t.shape}')
        # print(f'last hidden layer shape: {hidden.shape}')

        # c_t = rearrange(c_t, "b t f h w -> (b t) f h w")
        
        # pred = self.segment_head(c_t)
        
        # pred = []
        # for i in range(self.pred_step):
        #     # sequentially pred future
        #     p_tmp = self.network_pred(hidden)
        #     pred.append(p_tmp)
        #     print(f'pred shape: {p_tmp.shape}')
        #     _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
        #     hidden = hidden[:,-1,:]
        # pred = torch.stack(pred, 1) # B, pred_step, xxx
        # del hidden

        # print(f'pred shape: {pred.shape}')


        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        # N = self.pred_step
        # # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
        # pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        # feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)
        # score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        # del feature_inf, pred

        # if self.mask is None: # only compute mask once
        #     # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
        #     mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()
        #     mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
        #     for k in range(B):
        #         mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
        #     tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
        #     for j in range(B*self.last_size**2):
        #         tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
        #     mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
        #     self.mask = mask

        c_t = rearrange(c_t, "(b n) sl c h w -> b n sl c h w", n=N)

        return c_t

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

