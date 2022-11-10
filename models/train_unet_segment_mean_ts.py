# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:40:37 2022

@author: xfei
"""

# runtime environmnet will need pytorch and a list of dependencies in disstl/requirements.txt

import disstl.models as models
import torch
import torchvision
from disstl.datasets.smart.datasets import from_cube
from disstl.datasets.transforms import ClipBands, MinMaxNormalize
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from unet.unet_vae import UNet_VAE_old
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
from tqdm import tqdm
import h5py


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
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=32, type=int)

def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros(image.shape)
    for band in range(image.shape[-1]):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def rescale_image(image: np.ndarray, rescale_type: str = 'per-image'):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    if rescale_type == 'per-image':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i] - np.min(image[:, :, i])) / \
                (np.max(image[:, :, i]) - np.min(image[:, :, i]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
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


def pad_3d(arr: np.ndarray, out_shape) -> np.ndarray:
    x, y, z = arr.shape
    output = np.zeros(out_shape, dtype=arr.dtype)
    output[:x, :y, :z] = arr
    return output


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


def get_data(image_paths, target_paths, train=True):
    """
    Args:
        image_paths: path to image folder
        target_paths: path to mask folder
    Return:
        data_dict: dictionary
    """
    im_dict = {}
    data_dict={}
    for index, im_file in enumerate(tqdm(image_paths)):
        # image = np.asarray(tifffile.imread(im_file))
        image = rxr.open_rasterio(im_file)
        image = np.array(image)
        tappan_name = image_paths[index][-47:-39]
        image = rescale_image(image)
        desired_shape = (13,335,335)
        image = pad_3d(image, desired_shape)

        if str(tappan_name) != "Tappan01":
            break


        if tappan_name not in data_dict.keys():
            im_dict[tappan_name] = []

        im_dict[tappan_name].append(image)
        
        if tappan_name not in data_dict.keys():
            data_dict[tappan_name] = {}
            data_dict[tappan_name]['ts'] = []
            data_dict[tappan_name]['mask'] = np.zeros((335,335))


    for key in data_dict.keys():
        # print(f"length of image dict{key} :{len(im_dict[key])}")
        data_dict[key]['ts'] = np.stack(im_dict[key], axis=0)

    del im_dict

    for idx, file in enumerate(target_paths):
        mask_name = file[-26:-18]
        if str(mask_name) != "Tappan01":
            break
        mask = np.load(file)
        mask = mask-1
        mask = mask.astype(int)
        # print(mask)
        data_dict[mask_name]['mask'] = mask

    return data_dict


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
            'mask': torch.LongTensor(Y)
        }


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data

    ### hls data
    path_to_hls = sorted(glob.glob('/home/geoint/tri/hls/*.tif'))
    path_to_hls_mask = sorted(glob.glob('/home/geoint/tri/hls_mask/*.npy'))

    # data_dictionary = get_data(path_to_hls, path_to_hls_mask)

    # ts_arr = data_dictionary['Tappan01']['ts']
    # mask_arr = data_dictionary['Tappan01']['mask']

    filename = "/home/geoint/tri/hls_ts_video/hls_data.hdf5"
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        ts_arr = f['Tappan01_ts'][()]
        mask_arr = f['Tappan01_mask'][()]

    seq_length = 5
    num_seq = 4
    print(f'data dict tappan01 ts shape: {ts_arr.shape}')
    print(f'data dict tappan01 mask shape: {mask_arr.shape}')

    # ts, mask = chipper(ts_arr, mask_arr, input_size=64)
    # ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
    # mask = mask.reshape((mask.shape[1],mask.shape[2]))

    train_ts_set = []
    train_mask_set = []
    ## get different chips in the Tappan Square for multiple time series
    iteration = 200

    temp_ts_set = []
    temp_mask_set = []
    for i in range(iteration):
        ts, mask = chipper(ts_arr, mask_arr, input_size=64)
        ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
        mask = mask.reshape((mask.shape[1],mask.shape[2]))

        # stach temporal dim to temporal mean
        ts = np.mean(ts, axis=0)
        # print(f"mean ts shape: {ts.shape}")

        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_mask_set = np.stack(temp_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    test_set = satDataset(train_ts_set , train_mask_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    loader_args_val = dict(batch_size=10, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_dl = DataLoader(test_set, **loader_args)
    val_dl = DataLoader(test_set, **loader_args_val)


    EPOCHS= 100
    # LR= 5E-3 # 3E-4
    LR= 1E-4
    MOMENTUM= 0.9
    WEIGHT_DECAY= 1E-5
    LOG_INTERVAL= 1 # UNITS: MINIBATCHES

    C = 13
    NUM_CLASSES = 2 # reconstruction hls
    dir_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet_VAE_old(num_classes=NUM_CLASSES,segment=True,in_channels=C)

    if torch.cuda.is_available():
        model.cuda()

    criterion = criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()

    loss_list = []
    min_loss = np.inf
    for epoch in range(EPOCHS): # no. of epochs
        
        ## Example usage:
        loss_item = 0.0
        for batch in train_dl:
            # print(batch['x'].shape)
            x = batch['x']  
            y = batch['mask']

            # print(f"x shape: {x.shape}")

            x = x.to(cuda, dtype=torch.float32)
            y = y.to(cuda, dtype=torch.long)

            output = model(x)

            loss = criterion(output[0], y)
            loss_item += loss
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: " + str(epoch) + "   , Loss: " + str(loss.item()/len(train_dl)) )

        epoch_loss = loss.item()/len(train_dl)
        loss_list.append(epoch_loss)

        if min_loss > epoch_loss:
            print(f'Validation Loss Decreased({min_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
            min_loss = epoch_loss
            # Saving State Dict
            torch.save(model.state_dict(), dir_checkpoint + f'test_segment_{C}band_unetvae_hls_{epoch}_{min_loss}.pth')

    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(labels = 'crossentropy loss',loc='upper right')
    plt.savefig('/home/geoint/tri/dpc/models/output/training_segment_loss.png')
    plt.close()


    for batch in train_dl:
        x = batch['x'].cuda()
        y = batch['mask'].numpy()
        output = model(x)
        print(f"output shape: {output[0].shape}")
        y_pred = output[0].cpu().clone().detach().numpy()

        index_array = np.argmax(y_pred, axis=1)
        break

    data_dir = '/home/geoint/tri/dpc/models/output/bidirect_unetvae_0915/'

    
    for j in range(x.shape[0]):
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1)
        plt.title("Image")
        image = np.transpose(batch['x'].numpy()[j,:3,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image))
        plt.subplot(1,3,2)
        plt.title("Segmentation Label")
        #values = np.unique(y.ravel())
        plt.imshow(y[j,:,:])
    
        plt.subplot(1,3,3)
        plt.title("Segmentation")
        #values = np.unique(y.ravel())
        plt.imshow(index_array[j,:,:])
        plt.savefig(str(data_dir)+str(j)+ '.png')

        plt.close()
        