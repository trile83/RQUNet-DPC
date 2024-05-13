# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:40:37 2022

@author: xfei
"""

# runtime environmnet will need pytorch and a list of dependencies in disstl/requirements.txt

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from unet.unet_vae import UNet_VAE_old
from unet.unet_test import UNet_test
import collections
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import argparse
import logging
import rioxarray as rxr
import glob
import h5py
from tqdm import tqdm
from einops import rearrange


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--num_chips', default=200, type=int)
parser.add_argument('--num_val', default=40, type=int)
parser.add_argument('--hidden_dim', default=200, type=int)
parser.add_argument('--standardization', default='local', type=str, help='local, global')
parser.add_argument('--normalization', default=10000, type=int)
parser.add_argument('--rescale', default='per-ts', type=str)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--channels', default=10, type=int)


def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros(image.shape)
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_image(
            image: np.ndarray,
            rescale_type: str = 'per-image',
            highest_value: int = 1
        ):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    mask = np.where(image[0, :, :] >= 0, True, False)

    if rescale_type == 'per-image':
        image = (image - np.min(image, initial=highest_value, where=mask)) \
            / (np.max(image, initial=highest_value, where=mask)
                - np.min(image, initial=highest_value, where=mask))
    elif rescale_type == 'per-ts':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    elif rescale_type == 'per-channel':
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i]
                - np.min(image[:, :, i], initial=highest_value, where=mask)) \
                / (np.max(image[:, :, i], initial=highest_value, where=mask)
                    - np.min(
                        image[:, :, i], initial=highest_value, where=mask))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

def standardize_image(
    image,
    standardization_type: str,
    mean: list = None,
    std: list = None
):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    image = image.astype(np.float32)
    mask = np.where(image[0, :, :] >= 0, True, False)

    if standardization_type == 'local':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.mean(image[i, :, :], where=mask)) / \
                (np.std(image[i, :, :], where=mask) + 1e-8)
    elif standardization_type == 'global':
        for i in range(image.shape[-1]):
            image[:, :, i] = (image[:, :, i] - mean[i]) / (std[i] + 1e-8)
    elif standardization_type == 'mixed':
        raise NotImplementedError
    return image

def normalize_image(image: np.ndarray, normalize: float):
    """
    Normalize image within parameter, simple scaling of values.
    Args:
        image (np.ndarray): array to normalize
        normalize (float): float value to normalize with
    Returns:
        normalized np.ndarray
    """
    if normalize:
        image = image / normalize
    return image


def _flatten_temporal_dim(preds, targets):
    # Flatten temporal dim if exists
    if preds.ndim > 2 and preds.ndim < 4:
        preds = rearrange(preds, "b t c -> (b t) c")
        targets = rearrange(targets, "b t -> (b t)")
    if preds.ndim > 4:
        preds = rearrange(preds, "b t c h w-> (b t) c h w")
        targets = rearrange(targets, "b t h w-> (b t) h w")
    return preds, targets


def chipper(ts_stack, mask, input_size):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = np.random.randint(0, h-input_size)
    j = np.random.randint(0, w-input_size)
    
    out_ts = np.array([ts_stack[:, :, i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask

class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y
        # self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]

        return {
            'x': torch.tensor(X),
            'mask': torch.tensor(Y)
        }

def get_composite(ts_arr, ts_len=15):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[0] // ts_len
    # print(step)

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[0], step):
        out_lst.append(ts_arr[i])

    out_array = np.stack(out_lst, axis=0)
    del ts_arr

    # print(out_array.shape)

    return out_array


def get_train_set(args, list_ts, tile='PEV'):
    
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_inc_cloud.hdf5"

    ### UPDATE 09/01 - new datacube with small TS time series
    if tile =='PEV':
        filename = "/projects/kwessel4/hls_datacube/hls-ecas-PEV-0901.hdf5"

    train_ts_set = []
    train_mask_set = []

    temp_ts_set = []
    temp_mask_set = []
    
    for ts_name in list_ts:
    
        print("Get data from Tappan: ", ts_name)
        #### UPDATEs 09/01
        with h5py.File(filename, "r") as file:
            ts_arr = file[f'{str(ts_name)}_{str(tile)}_ts'][()]
            mask_arr = file[f'{str(ts_name)}_{str(tile)}_mask'][()]

        print("out ts arr shape: ", ts_arr.shape)

        print("out ts arr max pixel value: ", np.max(ts_arr))
        print("out ts arr max pixel value: ", np.min(ts_arr))

        if ts_arr.shape[0] > args.ts_length:
            ts_arr = get_composite(ts_arr, args.ts_length)

        #mask_arr[mask_arr != 2] = 0
        #mask_arr[mask_arr == 2] = 1

        input_size = args.img_dim
        total_ts_len = args.ts_length # L

        #ts_arr = np.concatenate((ts_arr[:args.ts_length,1:-4,:,:], ts_arr[:args.ts_length,-2:,:,:]), axis=1)
        if args.channels == 10:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:-4,:,:], ts_arr[:args.ts_length,-2:,:,:]), axis=1)
        elif args.channels == 4:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:4,:,:], np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)), axis=1)  

        print("ts arr shape after composite: ", ts_arr.shape)

        ## TEST: 12/11/2023
        if args.normalization is not None:
            ts_arr = normalize_image(ts_arr, args.normalization)

        for i in range(args.num_chips+args.num_val):
            ts, mask = chipper(ts_arr[:,:,:,:], mask_arr, input_size=args.img_dim)
            ts = np.squeeze(ts)

            ## TEST: 12/11/2023
            if args.rescale is not None:
                if args.rescale == 'per-ts':
                    ts = rescale_image(ts,args.rescale)
                elif args.rescale == 'per-image':
                    for frame in range(ts.shape[0]):
                        ts[frame] = rescale_image(ts[frame], args.rescale)
            if args.standardization is not None:
                for frame in range(ts.shape[0]):
                    ts[frame] = standardize_image(ts[frame],args.standardization)

            
            # Works November 2023

            # if args.rescale == 'per-ts' and args.normalization is None and args.standardization is None:
            #     ts = rescale_image(ts, args.rescale)
                
            # else:
                
            #     if args.normalization is not None:
            #         for frame in range(ts.shape[0]):
            #             ts[frame] = normalize_image(ts[frame], args.normalization)

            #     if args.standardization is not None:
            #         for frame in range(ts.shape[0]):
            #             ts[frame] = standardize_image(ts[frame],args.standardization)

            #     if args.rescale is not None:
            #         ts = rescale_image(ts,args.rescale)

            mask = np.squeeze(mask)

            # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

            temp_ts_set.append(ts)
            temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_mask_set = np.stack(temp_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    return train_ts_set, train_mask_set, args.num_val*len(list_ts)



if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.
    ### hls data
    ts_name=args.dataset
    tile='PEV'
    list_ts = ['Tappan01_WV02_20181217', 'Tappan18_WV02_20170126']

    input_size = args.img_dim

    train_ts_set, train_mask_set, num_val = get_train_set(args, list_ts, tile)

    seq_length = 5
    num_seq = 4
    input_size = args.img_dim

    # get RGB image

    # get 10-band HLS
    ts_arr = train_ts_set
    train_ts_arr = ts_arr[:-num_val]
    test_ts_arr = ts_arr[-num_val:]

    train_ts_arr = rearrange(train_ts_arr, "b t c h w -> (b t) c h w")
    test_ts_arr = rearrange(test_ts_arr, "b t c h w -> (b t) c h w")

    print(f'ts shape: {train_ts_arr.shape}')
    #print(f'mask shape: {mask_arr.shape}')

    #ts, mask = chipper(ts_arr, mask_arr, input_size=input_size)
    #for j in range(ts.shape[0]):
        #ts[j] = rescale_image(ts[j])
    #ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
    #mask = mask.reshape((mask.shape[1],mask.shape[2]))

    train_set = satDataset(train_ts_arr, train_ts_arr)
    val_set = satDataset(test_ts_arr, test_ts_arr)

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_dl = DataLoader(train_set, **loader_args)
    val_dl = DataLoader(val_set, **loader_args)

    EPOCHS= args.epochs
    # LR= 5E-3 # 3E-4
    LR= 3E-4
    MOMENTUM= 0.9
    WEIGHT_DECAY= 0
    LOG_INTERVAL= 1 # UNITS: MINIBATCHES

    C = args.channels
    NUM_CLASSES = args.channels # reconstruction hls
    dir_checkpoint = '/scratch/mle35/dpc/train_checkpoints/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = UNet_VAE_old(num_classes=NUM_CLASSES, segment=False, in_channels=C)
    model = UNet_test(num_classes=NUM_CLASSES, segment=False, in_channels=C)

    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    # criterion1 = models.losses.MultiTemporalCrossEntropy()
    # criterion = models.losses.InfoNCE()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_list = []
    min_loss = np.inf
    for epoch in range(EPOCHS): # no. of epochs
        #get the network output
        model.train()
        ## Example usage:
        loss_item = 0.0
        for batch in train_dl:
            # print(batch['x'].shape)
            x = batch['x']  # [1, 13, 64, 64] , [batch x bands x width x height]
            y = batch['x']

            (B,C,H,W) = x.shape

            # x = rearrange(x, "b c h w -> (b sl) c h w")
            # y = rearrange(y, "b c h w -> (b sl) c h w")

            x = x.to(cuda, dtype=torch.float32)
            y = y.to(cuda, dtype=torch.float32)

            output = model(x)
            # print('prediction shape: ', output[1].shape)
            loss = criterion(output[1], y)
            loss_list.append(loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(loss['loss'].item())
        print("Epoch: " + str(epoch) + "   , Loss: " + str(loss.item()/len(train_dl)) )

        loss_item += loss
        epoch_loss = loss.item()/len(train_dl)

        if min_loss > epoch_loss:
            print(f'Validation Loss Decreased({min_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
            min_loss = epoch_loss
            # Saving State Dict
            torch.save(model.state_dict(), dir_checkpoint + f'recon_0315_{args.channels}band_unet_hls_dim{args.img_dim}_{epoch}_{min_loss}.pth')

    # torch.save(model.state_dict(), dir_checkpoint + f'bidirect_seg_BH_R001_conv3d_5band_unetvae_{epoch}_{loss_item}.pth')

    # for batch in train_dl:
    #     x = batch['x'].cuda()
    #     y = batch['x'].cuda()
    #     #y = batch['binary_activity_mask'].numpy()
    #     output = model(x)
    #     y_pred =output.y_pred.cpu().clone().detach().numpy()

    #     y_pred = rearrange(y_pred, "(b t) c h w -> b t c h w", t=10)

    #     index_array = np.argmax(y_pred, axis=2)
    #     break

    # data_dir = 'output/unetvae/'

    
    # for j in range(10):
    #     for i in range(10):
    #         plt.figure(figsize=(20,20))
    #         plt.subplot(1,3,1)
    #         plt.title("Image")
    #         image = np.transpose(batch['x'].numpy()[j,i,:3,:,:], (1,2,0))  
    #         plt.imshow(rescale_truncate(image))
    #         plt.subplot(1,3,2)
    #         plt.title("Segmentation Label")
    #         #values = np.unique(y.ravel())
    #         plt.imshow(y[j,i,:,:])
        
    #         plt.subplot(1,3,3)
    #         plt.title("Segmentation")
    #         #values = np.unique(y.ravel())
    #         plt.imshow(index_array[j,i,:,:])
    #         plt.savefig(data_dir+str(j)+ '_' + str(i))
        
        
    # for batch1 in train_dl: 
    #     x1 = batch1['x'].cuda()
    #     y1 = batch1['binary_activity_mask'].numpy()
    #     output1 = model(x1)
    #     y_pred1 =output1.y_pred.cpu().clone().detach().numpy()
    #     y_pred1 = rearrange(y_pred1, "(b t) c h w -> b t c h w", t=10)
        
    #     # index_array1 = np.argmax(y_pred1, axis=2)
    #     break
        
    # i = 1  
    # for idx_temp in range(10):
    #     plt.figure(figsize=(20,20))
    #     plt.subplot(1,3,1)
    #     plt.title("Image")
    #     image1 = np.transpose(batch1['x'].numpy()[idx_temp,i,:3,:,:], (1,2,0))  
    #     plt.imshow(rescale_truncate(image1))
    #     plt.subplot(1,3,2)
    #     plt.title("Segmentation Label")
    #     #values = np.unique(y.ravel())
    #     plt.imshow(y1[idx_temp,i,:,:])

    #     plt.subplot(1,3,3)
    #     plt.title("Segmentation")
    #     #values = np.unique(y.ravel())
    #     plt.imshow(index_array1[idx_temp,i,:,:])
        

    # for i in range(10):
    #     plt.figure(figsize=(20,20))
    #     plt.subplot(1,3,1)
    #     plt.title("Image")
    #     image = np.transpose(batch1['x'].numpy()[2,i,:3,:,:], (1,2,0))  
    #     plt.imshow(rescale_truncate(image))
    #     plt.subplot(1,3,2)
    #     plt.title("Segmentation Label")
    #     #values = np.unique(y.ravel())
    #     plt.imshow(y1[2,i,:,:])

    #     plt.subplot(1,3,3)
    #     plt.title("Segmentation")
    #     #values = np.unique(y.ravel())
    #     plt.imshow(index_array1[2,i,:,:])
    #     plt.savefig(data_dir+'new_' + str(i))
