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
# from benchmod.convgru import ConvGRU
from unet.unet_vae import UNet_VAE_old
from unet.unet_vae_RQ_scheme1_encoder import UNet_VAE_RQ_scheme1_encoder
from unet.unet_test import UNet_test


def reverse_chunks(chunks, num_seq):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L2,N,SL,C,H,W) = chunks.shape
    
    all_arr = torch.zeros((I,L2+num_seq-1,SL,C,H,W))
    for j in range(I):
        for i in range(L2):
            if i < L2-1:
                array = chunks[j,i,0,:,:,:,:] # L2, N, SL, C, H, W
                all_arr[j,i,:,:,:,:] = array
            elif i == L2-1:
                array = chunks[j,i,:,:,:,:,:]
                all_arr[j,i:i+num_seq,:,:,:,:] = array
            del array

    # for j in range(I):
    #     all_arr[j,L2+num_seq-1,:,:,:,:] = chunks[j,L2-1,-1,:,:,:,:]

    return all_arr

def reverse_seq(window, seq_length):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L1,SL,C,H,W) = window.shape
    all_arr = torch.zeros((I,L1+seq_length-1,C,H,W))
    for j in range(I):
        for i in range(L1):
            if i < L1-1:
                array = window[j,i,0,:,:,:] # L2, N, SL, C, H, W
                all_arr[j,i,:,:,:] = array
            elif i == L1-1:
                array = window[j,i,:,:,:,:]
                all_arr[j,i:i+seq_length,:,:,:] = array
            del array

    return all_arr


class SegmentationHead_3D(nn.Sequential): 
    """
    Convolutional Segmentation head

    args:
        input_dim:  input layer size
        hidden_dim: hidden layer size
        output_dim: output layer size
        kernel_size: conv kernel size
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            # nn.Upsample(scale_factor=2),
            nn.Conv3d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, output_dim, kernel_size, padding=padding),
        )

## Function to reduce 4D time series to 3D
def stride_over_channels(x):
    # Since our timesteps is in our "channels" position, we slowly
    # reduce it down to 1.
    # This is striding over our channels

    c1 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
    c2 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, padding=1)

    # out1 = c1(x)            # batch-size x 4 x channels x height x width
    # out2 = c2(out1)         # batch-size x 1 x channels x height x width
    # out3 = out2.squeeze(1)  # batch-size x channels x height x width

    x = x.permute(1, 0, 2, 3, 4)

    # c1 = nn.Conv3d(in_channels=10, out_channels=5, kernel_size=3, padding=1)
    # c2 = nn.Conv3d(in_channels=5, out_channels=1, kernel_size=3, padding=1)

    out1 = c1(x.cpu().float())      # batch-size x 4 x channels x height x width
    out2 = c2(out1)                 # batch-size x 1 x channels x height x width
    out3 = out2.squeeze(1)

    out = out3

    return out.to(cuda)


class DPC_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, device, num_seq=8, seq_len=5, pred_step=3, num_class=2, \
        network='resnet50', model_weight='', freeze=False, dropout=0.5, segment=True, segment_model='conv3d'):

        super(DPC_RNN, self).__init__()
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

            ## DPC code | BlackSky Code
            self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=128,
                               kernel_size=3,
                               num_layers=self.param['num_layers'])
            
            ## UTAE paper ConvGRU code
            # self.agg = ConvGRU(
            #         input_dim=self.param['feature_size'],
            #         input_size=(64,64),
            #         hidden_dim=128,
            #         kernel_size=(3,3),
            #         num_layers=self.param['num_layers'],
            #         return_all_layers=False,
            #     )

            print('Done with ConvGRU!')

        else:
            self.backbone, self.param = select_resnet(network, track_running_stats=False)

            self.param['num_layers'] = 1 # param for GRU
            self.param['hidden_size'] = self.param['feature_size'] # param for GRU # 1024 for resnet50, 256 for resnet18
        
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

        self.segment_model = segment_model

        if self.segment_model == 'unet':
            self.segment_head = UNet_test(num_classes=2,segment=True,in_channels=128)
        elif self.segment_model == 'conv3d':
            self.segment_head = SegmentationHead_3D(
                input_dim=128, hidden_dim=10, output_dim=2
            )
        elif self.segment_model == 'conv2d':
            self.segment_head = nn.Conv2d(
                in_channels=128,
                out_channels=2,
                kernel_size=(3,3),
                padding=1,
            )

        # encoder_dim = self.param['feature_size']
        # gar_dim = 128
        # mlp_hidden_dim = 64
        # self.segment_head = SegmentationHead_3D(
        #     input_dim=encoder_dim + gar_dim * 2, hidden_dim=mlp_hidden_dim, output_dim=2
        # )

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
        global cuda; cuda = torch.device('cuda')
        # print(f'block shape: {block.shape}')
        (B, L2, N, SL, C, H, W) = block.shape
        block = rearrange(block, "b l2 n sl c h w -> (b l2) n sl c h w")
        # (B, N, SL, C, H, W) = block.shape
        if self.network == 'unet-vae' or self.network == "rqunet-vae-encoder" or self.network == 'unet':
            
            block = rearrange(block, "b n sl c h w -> (b n sl) c h w")
            ### encoder
            feature = self.backbone(block)
            # _, feature = self.backbone(block)

            del block

            feature = self.relu(feature) # [0, +inf)
            feature = torch.reshape(feature, (L2,N,SL,feature.shape[1],feature.shape[2],feature.shape[3]))

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
                context, last_state = self.agg(feats)
                # print('context vector shape: ', context.shape)
                # print('last state shape: ', last_state.shape)
                # print('last state squeeze shape: ', last_state.mean(dim=0).shape)

                #### required steps for segmenting the context vector
                # context = rearrange(context, "(b l2 n) sl c h w -> b l2 n sl c h w", n=N, l2=L2)
                # context = reverse_seq(reverse_chunks(context, self.num_seq), self.seq_len)

                # context = stride_over_channels(context)
                # print('context vector shape after striding: ', context.shape)

                # context = context.to(cuda, dtype=torch.float32)

                # output, _ = self.segment_head(last_state.mean(dim=0))

                ### using last_state vector
                ##### works as of 03/23/2023
                # output = self.classification_layer(last_state.mean(dim=0)) ## conv2d

                #### works for UNet
                # output, _ = self.segment_head(last_state.mean(dim=0))

                # output = self.head(last_state) ## conv3d

                #######
                if self.segment_model == 'unet':
                    output, _ = self.segment_head(last_state.mean(dim=0))
                elif self.segment_model == 'conv3d':
                    last_state = rearrange(last_state, "b t c h w -> b c t h w")
                    output = self.segment_head(last_state)
                    output = rearrange(output, "b c t h w -> b t c h w")
                    output = output.mean(dim=0)
                elif self.segment_model == 'conv2d':
                    output = self.segment_head(last_state.mean(dim=0))

                #######

                ### using context vector

                # output = self.classification_layer(context.mean(dim=1)) ## conv2d
                # output, _ = self.segment_head(context.mean(dim=1))

                ##### segment using conv3d
                # context = rearrange(context, "b t c h w -> b c t h w")
                # output = self.head(context)
                # output = rearrange(output, "b c t h w -> b t c h w")
                # output = output.mean(dim=1)

                #######
                # if self.segment_model == 'unet':
                #     output, _ = self.segment_head(context.mean(dim=1))
                # elif self.segment_model == 'conv3d':
                #     context = rearrange(context, "b t c h w -> b c t h w")
                #     output = self.segment_head(context)
                #     output = rearrange(output, "b c t h w -> b t c h w")
                #     output = output.mean(dim=1)
                # elif self.segment_model == 'conv2d':
                #     output = self.segment_head(context.mean(dim=1))

                #######

                # print('output shape: ', output.shape)

                ##################
                # # print('feature in dpc shape: ', feats.shape)
                # context, _ = self.agg(feats)
                # # context = context[:,-1,:].unsqueeze(1) # works

                # context = context[:,-1,:].unsqueeze(0)

                # # context = stride_over_channels(context)
                # # # print('context vector shape after striding: ', context.shape)

                # # print('context vector shape: ', context.shape)

                # output, _ = self.segment_head(context.mean(dim=0))
                # output, _ = self.segment_head(context)

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

