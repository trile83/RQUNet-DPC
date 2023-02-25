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
from unet.unet_test import UNet_test


class DPC_RNN_UNet(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, device, num_seq=8, seq_len=5, pred_step=3, num_class=2, \
        network='resnet50', model_weight='', freeze=False, dropout=0.5, segment=True):

        super(DPC_RNN_UNet, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        self.network = network
        self.segment = segment
        self.num_class = num_class
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.param = {}

        self.param['feature_size'] = 0
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = 0

        print('Initializing backbone')
        
        if network == "unet-vae" or network == "rqunet-vae-encoder" or network == "unet":
            self.param['feature_size'] = 960
            self.param['num_layers'] = 1 # param for GRU
            self.param['hidden_size'] = self.param['feature_size'] # param for GRU # 1024 for resnet50, 256 for resnet18

            if network == "unet-vae":
                self.backbone = UNet_VAE_old(num_classes=10,segment=False,in_channels=10,depth=5,is_encoder=True)
            elif network == 'unet':
                self.backbone = UNet_test(num_classes=10, in_channels=10, segment=False,is_encoder=True)
            elif network == 'rqunet-vae-encoder':
                self.backbone = UNet_VAE_RQ_scheme1_encoder(num_classes=10,segment=False,in_channels=10,depth=5,is_encoder=True,alpha = 0.1)

            print('Load backbone weights!')
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

            print('Done with backbone! Initializing ConvGRU!')

            self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=128,
                               kernel_size=3,
                               num_layers=self.param['num_layers'])

            print('Done with ConvGRU!')

        else:
            self.backbone, self.param = select_resnet(network, track_running_stats=False)

            self.param['num_layers'] = 1 # param for GRU
            self.param['hidden_size'] = self.param['feature_size'] # param for GRU # 1024 for resnet50, 256 for resnet18

            # self.agg = ConvGRU(input_size=self.param['feature_size'],
            #                        hidden_size=128,
            #                        kernel_size=3,
            #                        num_layers=self.param['num_layers'])
        
            self.agg = ConvGRU(input_size=self.param['feature_size'],
                                hidden_size=self.param['hidden_size'],
                                kernel_size=1,
                                num_layers=self.param['num_layers'])

            self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )


            # model-3d-lc
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()

            self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(self.param['feature_size'], self.num_class))
            self._initialize_weights(self.final_fc)

            self._initialize_weights(self.network_pred)

        # self.segment_head = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)
        self.segment_head = UNet_test(num_classes=2,segment=True,in_channels=128)

        self.relu = nn.ReLU(inplace=False)
        self.mask = None
        
        self._initialize_weights(self.agg)


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
        if self.network == 'unet-vae' or self.network == "rqunet-vae-encoder" or self.network == 'unet':
            
            block = rearrange(block, "b n sl c h w -> (b n sl) c h w")
            ### encoder
            feature = self.backbone(block)
            # _, feature = self.backbone(block)

            del block

            feature = self.relu(feature) # [0, +inf)
            feature = torch.reshape(feature, (B,N,SL,feature.shape[1],feature.shape[2],feature.shape[3]))

            ### aggregate, predict future ###

            # feats = feature[:, 0:N-self.pred_step, :].contiguous()
            feats = rearrange(feature, "b n sl c h w -> (b n) sl c h w")

            # feats = feature[:, 0:N-self.pred_step, :].contiguous() # need to reshape
            # print(f'feats shape: {feats.shape}')
                
            if not self.segment:
                
                c_t, hidden = self.agg(feats)
                # hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
                c_t = rearrange(c_t, "(b n) sl c h w -> b n sl c h w", n=N)

                return c_t

            else:

                # print('feature in dpc shape: ', feats.shape)
                context, _ = self.agg(feats)
                context = context[:,-1,:].unsqueeze(1)

                # print('context vector shape: ', context.shape)

                output, _ = self.segment_head(context.mean(dim=0))
                # print('output shape: ', output.shape)

                return output, context

        else:
            block = block.view(B*N, C, SL, H, W)
            feature = self.backbone(block)

            del block

            if not self.segment:

                feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

                feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
                feature = self.relu(feature) # [0, +inf)
                feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)

                # print(f'feature shape: {feature.shape}')
                feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
                del feature_inf_all

                ### aggregate, predict future ###
                _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
                hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

                # print(f'hidden shape: {hidden.shape}')
                
                pred = []
                for i in range(self.pred_step):
                    # sequentially pred future
                    p_tmp = self.network_pred(hidden)
                    # print(f'p_tmp shape :{p_tmp.shape}')
                    pred.append(p_tmp)
                    _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
                    hidden = hidden[:,-1,:]
                pred = torch.stack(pred, 1) # B, pred_step, xxx
                del hidden

                ## Get similarity score ###
                # pred: [B, pred_step, D, last_size, last_size]
                # GT: [B, N, D, last_size, last_size]
                N = self.pred_step
                # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
                pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
                feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)
                score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
                del feature_inf, pred

                if self.mask is None: # only compute mask once
                    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
                    mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()
                    mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
                    for k in range(B):
                        mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
                    tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
                    for j in range(B*self.last_size**2):
                        tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
                    mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
                    self.mask = mask

                return [score, self.mask]

            else:

                feature = F.relu(feature)

                print('feature in dpc shape: ', feature.shape)
                
                # feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
                # feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,D,last_size,last_size]
                # context, _ = self.agg(feature)
                # context = context[:,-1,:].unsqueeze(1)
                # context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
                # del feature

                # context = self.final_bn(context.transpose(-1,-2)).transpose(-1,-2) # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
                # output = self.final_fc(context).view(B, -1, self.num_class)

                context, _ = self.agg(feature)
                context = context[:,-1,:].unsqueeze(1)

                print('context vector shape: ', context.shape)

                output, _ = self.segment_head(context)
                print('output shape: ', output.shape)

                return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

