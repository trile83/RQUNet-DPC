#import disstl.models as models
import re
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
import glob
# from dpc.model_3d import *
from dpc.model_3d_unet import DPC_RNN_UNet
from dpc.model_3d_unet_stride import DPC_RNN
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
import argparse
import h5py
import logging
import re
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix, classification_report
import rioxarray as rxr
import xarray as xr
from inference import inference
#from tensorboardX import SummaryWriter
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from benchmod.convgru import ConvGRU_Seg
from unet3d.unet3d import UNet3D
from unet.unet_test import UNet_test
import pickle
from datetime import date
import csv

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='unet', type=str, help='encoder for the DPC')
parser.add_argument('--model', default='dpc-unet', type=str, help='convlstm, dpc-unet, unet')
parser.add_argument('--dataset', default='PEV', type=str, help='PEV, PFV, PEA')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--standardization', default='local', type=str)
parser.add_argument('--normalization', default=10000, type=float)
parser.add_argument('--rescale', default='per-ts', type=str)
parser.add_argument('--segment_model', default='conv3d', type=str)
parser.add_argument('--hidden_dim', default=200, type=int)
parser.add_argument('--channels', default=10, type=int)
parser.add_argument('--criterion', default='crossentropy', type=str)


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


class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask

        return {
            'x': X,
            'mask': Y
        }

class tsDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]

        return {
            'ts': torch.tensor(X),
            'mask': torch.LongTensor(Y)
        }


class segmentDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, Z):
        'Initialization'
        self.data = X
        self.mask = Y
        self.imge = Z
        # self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]
        Z = self.imge[index]

        return {
            'x': X,
            'mask': Y,
            'ts': Z
        }

def get_seq(sequence, seq_length):
    '''
    sequence: long time-series input
    seq_length: length of window for time-series chunk; default = 5
    return: array with all time-series chunk; prediction size =1, num_seq = 4
    '''
    (I,L,C,H,W) = sequence.shape
    all_arr = np.zeros((I,L-seq_length+1,seq_length,C,H,W))
    for j in range(I):
        for i in range(seq_length, L+1):
            array = sequence[j,i-seq_length:i,:,:,:] # SL, C, H, W
            all_arr[j,i-seq_length] = array

    return all_arr

def get_chunks(windows, num_seq):
    '''
    TODO: match with get_seq function
    windows: number of windows in 1 time-series
    number: number of window for time-series chunk; default = 4
    return: array with all time-series chunk; prediction size =1, num_seq (N) = 4; N x 6 x 5 x 32 x 32
    '''
    (I,L1,SL,C,H,W) = windows.shape
    all_arr = np.zeros((I,L1-num_seq+1,num_seq,SL,C,H,W))
    for j in range(I):
        for i in range(num_seq, L1+1):
            array = windows[j,i-num_seq:i,:,:,:,:] # N, SL, C, H, W
            # if not array.any():
            #     print(f"i {i}")
            all_arr[j,i-num_seq] = array

    return all_arr


def reverse_chunks(chunks, num_seq):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L2,N,SL,C,H,W) = chunks.shape
    
    all_arr = np.zeros((I,L2+num_seq-1,SL,C,H,W))
    for j in range(I):
        for i in range(L2):
            if i < L2-1:
                array = chunks[j,i,0,:,:,:,:] # L2, N, SL, C, H, W
                all_arr[j,i,:,:,:,:] = array
            elif i == L2-1:
                array = chunks[j,i,:,:,:,:,:]
                all_arr[j,i:i+num_seq,:,:,:,:] = array
            del array

    return all_arr

def reverse_seq(window, seq_length):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L1,SL,C,H,W) = window.shape
    all_arr = np.zeros((I,L1+seq_length-1,C,H,W))
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


def get_accuracy(y_pred, y_true):

    target_names = ['non-crop','cropland']

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # get overall weighted accuracy
    accuracy = accuracy_score(y_true, y_pred, sample_weight=None)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    # iou_1 = jaccard_score(y_pred, y_true, average=None)
    iou = jaccard_score(y_pred, y_true, pos_label=1, average='binary')
    precision = report['cropland']['precision']
    recall = report['cropland']['recall']
    f1_score = report['cropland']['f1-score']
    return accuracy, precision, recall, f1_score, iou

def save_raster(ref_im, prediction, name, out_dir):

    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(dim="band",
            labels=ref_im.coords["band"].values[1:]
        )

    prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name='dpc',
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('dpc')
    prediction = prediction.transpose("band", "y", "x")

    # Set nodata values on mask
    # nodata = prediction.rio.nodata
    # prediction = prediction.where(ref_im != nodata)
    # prediction.rio.write_nodata(
    #     -1, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'{out_dir}/{name}.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )
    
    ##  save dice probability output
    # prediction.rio.to_raster(
    #     f'{out_dir}/{name}.tiff',
    #     BIGTIFF="IF_SAFER",
    #     compress='LZW',
    #     num_threads='all_cpus',
    #     driver='GTiff',
    #     dtype='float32'
    # )

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

def prepare_data(args, train_ts_set):

    model_option = args.model
    
    print('original time series max value: ', np.max(train_ts_set))
    print('original time series min value: ', np.min(train_ts_set))

    if model_option == 'unet':

        if args.rescale == 'per-ts':
            train_ts_set = rescale_image(train_ts_set, args.rescale)
        xraster = train_ts_set[:args.ts_length,:,:,:].mean(axis=0)
        temporary_tif = xr.where(xraster > -1000, xraster, 2000)

    elif model_option == 'decision-tree' or model_option == 'random-forest':
        # if args.rescale == 'per-ts':
        #     train_ts_set = rescale_image(train_ts_set, args.rescale)
        xraster = train_ts_set[:args.ts_length,:,:,:]
        print('xraster shape: ', xraster.shape)
        temporary_tif = xr.where(xraster > -9000, xraster, -2000)

    elif model_option == '3d-unet':

        xraster = train_ts_set[:,:,:,:]
        xraster = xraster[:args.ts_length,:,:,:]
        temporary_tif = xr.where(xraster > -9000, xraster, -2000)
    
    elif model_option == 'convlstm':

        xraster = train_ts_set
        xraster = xraster[:args.ts_length,:,:,:]
        temporary_tif = xr.where(xraster > -9000, xraster, -800)


    elif model_option == 'convgru':

        xraster = train_ts_set
        xraster = xraster[:args.ts_length,:,:,:]
        temporary_tif = xr.where(xraster > -9000, xraster, -800)


    elif model_option == 'dpc-unet':

        xraster = train_ts_set[:,:,:,:]

        xraster = xraster[:args.ts_length,:,:,:]
        temporary_tif = xr.where(xraster > -9000, xraster, -100)

        print('temp tif shape: ', temporary_tif.shape)

    return temporary_tif

def get_model(args):
    '''
    Args:

    '''

    model_option = args.model # 'dpc-unet' and 'unet', convlstm
    input_size = args.img_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    standardization = args.standardization
    normalization = args.normalization
    rescale = args.rescale

    model_dir = "/projects/kwessel4/dpc/checkpoints/"

    if model_option == 'unet':
        
        unet_segment = UNet_test(num_classes=2,segment=True,in_channels=10)

        ### 10 bands

        unetsegment_checkpoint = f'{str(model_dir)}unet_meanframe_2023-11-11_10band_epoch_10.pth'
        if torch.cuda.is_available():
            unet_segment = unet_segment.to(cuda)

        unet_segment.load_state_dict(torch.load(unetsegment_checkpoint)['state_dict'])

        model = unet_segment
        batch_size = 32

    elif model_option == 'decision-tree':
        filename = f"{model_dir}decisiontree_model_0505.sav"

        clf = pickle.load(open(filename, 'rb'))

        model = clf
        batch_size = 32

    elif model_option == 'random-forest':
        filename = f"{model_dir}random-forest-2024-03-11.sav"

        rf = pickle.load(open(filename, 'rb'))

        model = rf
        batch_size = 16


    elif model_option == '3d-unet':
        model = UNet3D(in_channel=10, n_classes=args.num_classes)

        #model_checkpoint = f'{str(model_dir)}3d-unet_2023-11-01_hidden200_10band_ts01_epoch_89.pth'

        ## model trained with 1ts
        #model_checkpoint = f'{str(model_dir)}3d-unet_2024-03-07_10band_0.04_epoch_39.pth'

        ## model trained with 4ts
        #model_checkpoint = f'{str(model_dir)}3d-unet_2024-03-29_10band_0.08_epoch_134.pth'

        ## model trained w 8 ts and crossentropy loss
        model_checkpoint = f'{str(model_dir)}3d-unet_2024-04-26_10band_0.27_epoch_7.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        model = model
        batch_size = 16
    
    elif model_option == 'convlstm':
        hidden_dim = 200
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=hidden_dim,
            input_dim=10,
            kernel_size=(3, 3)
            )
 
        ### 10 bands

        if hidden_dim == 160:
            model_checkpoint = f'{str(model_dir)}convlstm_0504_10band_ts01_epoch_85.pth' # w only ts01
        elif hidden_dim == 180:
            model_checkpoint = f'{str(model_dir)}convlstm_0504_new_10band_ts01_epoch_95.pth'
        elif hidden_dim == 200:
            #model_checkpoint = f'{str(model_dir)}convlstm_2023-10-26_hidden200_10band_ts01_epoch_93.pth'
            
            ## rescale per-ts, standardization None
            if standardization == 'None' or standardization is None:
                #model_checkpoint = f'{str(model_dir)}convlstm_2024-03-08_10band_0.003_epoch_91.pth'
                ## MODEL w 4TS
                #model_checkpoint = f'{str(model_dir)}convlstm_2024-04-09_10band_0.033_epoch_278.pth'
                ## MODEL w 8TS
                model_checkpoint = f'{str(model_dir)}convlstm_2024-04-26_10band_0.023_epoch_132.pth'
            else:
                model_checkpoint = f'{str(model_dir)}convlstm_2024-03-29_10band_0.011_epoch_129.pth'

        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        model = model
        batch_size = 8

    elif model_option == 'convgru':
        model = ConvGRU_Seg(
                num_classes=args.num_classes,
                input_size=(input_size,input_size),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
 
        ### 10 bands

        # train with TS01 and TS07
        #model_checkpoint = f'{str(model_dir)}convgru_2023-10-26_hidden200_10band_ts01_epoch_88.pth'
        #model_checkpoint = f'{str(model_dir)}convgru_2024-03-08_10band_0.004_epoch_91.pth'

        ## rescale per-ts, standardization None
        if standardization == 'None' or standardization is None:
            #model_checkpoint = f'{str(model_dir)}convgru_2024-03-08_10band_0.005_epoch_86.pth'

            ## MODEL w 4TS
            #model_checkpoint = f'{str(model_dir)}convgru_2024-04-09_10band_0.019_epoch_340.pth'
            ## MODEL w 8TS
            model_checkpoint = f'{str(model_dir)}convgru_2024-04-26_10band_0.013_epoch_143.pth'
        else:
            model_checkpoint = f'{str(model_dir)}convgru_2024-04-09_10band_0.008_epoch_340.pth'

        ## ## rescale None, standardization local
        #model_checkpoint = f'{str(model_dir)}convgru_2024-03-08_10band_0.004_epoch_91.pth'

        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        model = model
        batch_size = 8

    elif model_option == 'dpc-unet':

        network = args.net # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'
        if network == 'unet':
            if input_size == 128:
                encoder_weight = '/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_64_5_8.505512028932572e-05.pth' # unet
            elif input_size == 64:
                if args.channels == 10:
                    encoder_weight = '/projects/kwessel4/dpc/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
                elif args.channels == 4:
                    encoder_weight = '/projects/kwessel4/dpc/checkpoints/recon_0315_4band_unet_hls_dim64_94_6.21471107006073e-05.pth' # unet
                    
        elif network == 'unet-vae' or network == 'rqunet-vae':
            if input_size == 64:
                encoder_weight = '/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_64_102_6.290748715400695e-05.pth' # unet-vae
            elif input_size == 128:
                encoder_weight = f'/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_128_5_6.96440190076828e-05.pth'


        model = DPC_RNN(sample_size=input_size,
                    device=device,
                    num_seq=args.num_seq, 
                    seq_len=args.seq_len,
                    hidden_dim=args.hidden_dim,
                    network=network,
                    pred_step=1,
                    model_weight=encoder_weight,
                    freeze=True,
                    segment_model=args.segment_model,
                    in_channels=args.channels)

        ### 10 bands
        if network == 'unet':

            ## unet segment
            if args.segment_model == 'unet':
                if args.hidden_dim == 128:
                    model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-0504-unet_10band_ts01_epoch16.pth'
                elif args.hidden_dim == 160:
                    model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-0504-unet_10band_ts01_epoch15_hidden_dim160.pth'
                elif args.hidden_dim == 180:
                    model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-0504-unet_180_10band_ts01_epoch16.pth'
                elif args.hidden_dim == 200:
                    # model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-0504-unet_200_10band_ts01_epoch21.pth'

                    ## Work November 2023
                    # model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-2023-11-10-composite-unet_200_10band_ts01_epoch50.pth'
                    # model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-2023-12-18-composite-local_unet_200_10band_ts01_epoch76.pth'
                    
                    ## crossentropy loss
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-05-crossentropy_unet_200_0.037_10band_18_epoch182.pth'
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-06-crossentropy_unet_200_0.434_10band_18_epoch5.pth'
                    
                    ## dice loss
                    model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-06-dice_unet_200_0.207_10band_18_epoch18.pth'


            elif args.segment_model == 'conv3d':
                ## segment 3d
                ##model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-2024-03-02-composite-local_conv3d_200_0.034869713336229326_10band_ts15-18_epoch7.pth'
                
                ## 10 channels
                if args.channels == 10:
                    ## WORKS AS OF MAR 20 2024
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-06-crossentropy_conv3d_200_0.021_10band_18_epoch88.pth'

                    ## TEST 0.15 binary crop balance -> WORKS AS OF MAR 20
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-20-crossentropy_conv3d_std_local_200_0.024_0.15binary_10band_18_epoch30.pth'

                    ## TEST MODEL TRAINED WITH 3 TSs
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-21-crossentropy_conv3d_std_local_200_0.176_0.2binary_10band_18_epoch52.pth'

                    ## TEST MODEL TRAINED WITH 4 TSs and DICE LOSS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-22-dice_conv3d_std_local_200_0.078_0.2binary_10band_18_epoch123.pth'

                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-12-dice_conv3d_std_local_200_0.073_0.2binary_10band_18_epoch104.pth'

                    ## TEST MODEL TRAINED WITH 4 TSs and CROSSENTROPY LOSS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-25-crossentropy_conv3d_std_local_100_4ts_0.19_0.3binary_10band_epoch19.pth'
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-05-02-crossentropy_conv3d_std_local_200_0.012_0.6binary_10band_epoch157.pth'
                    model_checkpoint = f'{str(model_dir)}dpc-unet-2024-05-10-crossentropy_conv3d_std_local_200_0.206_0.3binary_10band_epoch25.pth'

                    ## TEST MODEL TRAINED WITH 8 TSs and DICE LOSS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-24-dice_conv3d_std_local_100_8ts_0.023_0.3binary_10band_epoch171.pth'
                
                    ## TEST MODEL TRAINED WITH 8 TSs and CROSSENTROPY LOSS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-25-crossentropy_conv3d_std_local_100_0.068_0.3binary_10band_epoch71.pth'

                    ## TEST MODEL TRAINED WITH 12 TSs and CROSSENTROPY LOSS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-05-01-crossentropy_conv3d_std_local_200_0.016_0.3binary_10band_epoch179.pth'
                
                    
                elif args.channels == 4:
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-03-15-crossentropy_conv3d_std_local_200_0.022_0.2binary_4band_18_epoch61.pth'

                    ## MODEL TRAINED WITH 1TS, ts_length=18
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-30-crossentropy_conv3d_std_local_100_0.017_0.2binary_4band_1ts_tslen18_epoch88.pth'
                    ## MODEL TRAINED WITH 8TS
                    #model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-26-crossentropy_conv3d_std_local_200_0.088_0.2binary_4band_18_epoch49.pth'

                    ## TEST MODEL TRAINED WITH 12 TSs and CROSSENTROPY LOSS
                    model_checkpoint = f'{str(model_dir)}dpc-unet-2024-04-30-crossentropy_conv3d_std_local_200_0.009_0.2binary_4band_18_epoch190.pth'
                
            elif args.segment_model == 'conv2d':
                model_checkpoint = f'{str(model_dir)}dpc-unet-encoder-2024-01-25-composite-local_conv2d_200_0.6651816606521607_10band_ts01-18_epoch2.pth'

        ## unet-vae feature extraction
        elif network == 'unet-vae':
            # if input_size == 128:
            #     model_checkpoint = f'{str(model_dir)}dpc-unet-unet-vae-encoder-0317_10band_ts01_epoch18.pth'
            
            model_checkpoint = f'{str(model_dir)}dpc-unet-vae-2024-03-14-crossentropy_conv3d_std_local_200_0.014_0.2binary_10band_18_epoch93.pth'

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        # model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model = model.to(cuda)

        # setup tools
        global de_normalize; de_normalize = denorm()

        model = model
        batch_size = 1

    return model, batch_size


def predict(model, batch_size, xraster, ref_im, args):
    '''
    Args:
        model: model to run prediction (dpc-unet, convlstm, convgru)
        batch-size: batch size to run sliding, smaller batch size takes less computer memory to run, recommend 1 for dpc-unet
        xraster: raster for prediction
        ref_im: reference image for outputing GeoTiff file 

    '''
    model_option = args.model # 'dpc-unet' and 'unet', convlstm
    input_size = args.img_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    standardization = args.standardization
    normalization = args.normalization
    criterion_type = args.criterion

    ### WORKS for DPC-Unet
    if args.normalization is not None:
        xraster = normalize_image(xraster, args.normalization)
        
    if criterion_type == 'dice':
    	num_classes = 1
    elif criterion_type == 'crossentropy':
    	num_classes = 2

    prediction = inference.sliding_window_tiler(
            xraster=xraster,
            model=model,
            n_classes=num_classes,
            overlap=0.5,
            batch_size=batch_size,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option,
            channels=args.channels
        )

    ref_im = ref_im.transpose("y", "x", "band")
    print(f'ref im shape: {ref_im.shape}')
    (h,w,c) = ref_im.shape
    
    print(f'prediction before edit shape: {prediction.shape}')

    # if model_option != "decision-tree" and model_option != 'random-forest':

    if prediction.shape[0] > 1:
        prediction = np.argmax(prediction, axis=0)
    else:
        # prediction = np.squeeze(prediction)
        prediction = np.squeeze(
            np.where(prediction > 0.7, 1, 0).astype(np.int16)
        )


    return prediction


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')
    
    today = date.today()

    # Get model
    model, batch_size = get_model(args)

    # prepare data
    ts_name=args.dataset
    if 'PEV_large' in args.dataset or 'PFV_large' in args.dataset or 'PEA_large' in args.dataset or 'PFA_large' in args.dataset:
        hls = True
    elif 'planet' in args.dataset:
        hls = True
    else:
        hls = False

    if not hls:
        
        #### UPDATE 09/01
        if args.dataset == 'PEV':
            tile = "PEV"
            filename = "/projects/kwessel4/hls_datacube/hls-ecas-PEV-0901.hdf5"

        elif args.dataset == 'PFV':
            tile='PFV'
            filename = "/projects/kwessel4/hls_datacube/hls-ecas-PFV-0901.hdf5"

        elif args.dataset == 'PEA':
            tile='PEA'
            filename = "/projects/kwessel4/hls_datacube/hls-eetz-PEA-0908.hdf5"

    else:
        if args.dataset == 'PEV_large_2016':
            tile = "PEV-2016"
            filename = "/projects/kwessel4/hls_datacube/hls-PEV-2016-1206.hdf5"
            
        elif args.dataset == 'PEV_large_2018':
            tile = "PEV-2016"
            filename = "/projects/kwessel4/hls_datacube/hls-PEV-2018-1220.hdf5"

        elif args.dataset == 'PFV_large_2016':
            tile='PFV-2016'
            filename = "/projects/kwessel4/hls_datacube/hls-PFV-2016-1206.hdf5"

        elif args.dataset == 'PFV_large_2018':
            tile='PFV-2018'
            filename = "/projects/kwessel4/hls_datacube/hls-PFV-2018-1220.hdf5"

        elif args.dataset == 'PEA_large_2016':
            tile='PEA-2016'
            filename = "/projects/kwessel4/hls_datacube/hls-PEA-2016-1206.hdf5"

        elif args.dataset == 'PFA_large_2016':
            tile='PFA-2016'
            filename = "/projects/kwessel4/hls_datacube/hls-PFA-2016-0318.hdf5"

        elif args.dataset == 'planet_2021':
            tile='planet-2021'
            filename = "/projects/kwessel4/hls_datacube/planet-2021-0315.hdf5"
        elif args.dataset == 'planet_etz_2021':
            tile='planet-etz-2021'
            filename = "/projects/kwessel4/hls_datacube/planet-etz-2021-0426.hdf5"

    with h5py.File(filename, "r") as file:
        
        if not hls:
            metrics_output_filename = f'/projects/kwessel4/dpc/output/csv/{args.model}_{today}_metrics.csv'
            metrics_csv_columns = ['filename','accuracy', 'precision', 'recall', 'f1_score', 'iou']

            if os.path.isfile(metrics_output_filename):
                write_type = 'a'
            else:
                write_type = 'w'
        
            with open(metrics_output_filename, write_type) as metrics_filename:

                # write row to filename
                writer = csv.writer(
                    metrics_filename, delimiter=',', lineterminator='\n')
                if write_type == 'w':
                    writer.writerow(metrics_csv_columns)


                all_keys = sorted(list(file.keys()))

                for index in range(0, len(all_keys), 2):
                    mask_arr = file[all_keys[index]][()]
                    ts_arr = file[all_keys[index+1]][()]

                    print("timeseries array shape: ", ts_arr.shape)

                    if args.ts_length > ts_arr.shape[0]:
                        continue

                    ts_name = re.search(f'(.*?)_{tile}', all_keys[index]).group(1)
                    if ts_name == "Tappan06_WV03_20151209":
                        continue

                    if ts_name == "Tappan07_WV03_20151209":
                        continue

                    if tile == 'PEV' and ts_name == 'Tappan05_WV02_20181217':
                        #ref_im_fl = f"/projects/kwessel4/resampled_senegal_hls/trimmed/{tile}/{str(ts_name)}.tif"
                        continue

                    if tile == 'PEV' and ts_name == 'Tappan17_WV02_20181217':
                        continue

                    ref_im_fl = f"/projects/kwessel4/resampled_senegal_hls/trimmed/{tile}/{str(ts_name)}.tif"
                    ref_im = rxr.open_rasterio(ref_im_fl)

                    if ts_arr.shape[0] > args.ts_length:
                        ts_arr = get_composite(ts_arr, args.ts_length)

                    mask_arr[mask_arr != 2] = 0
                    mask_arr[mask_arr == 2] = 1

                    total_ts_len = args.ts_length # L

                    padding_size = args.pad_size
                    network = args.net
                    model_option = args.model # 'dpc-unet' and 'unet', convlstm

                    print(f'data dict {ts_name} ts shape: {ts_arr.shape}')
                    print(f'data dict {ts_name} mask shape: {mask_arr.shape}')
                    
                    if ts_arr.shape[0] < args.ts_length and model_option == 'dpc-unet':
                        continue


                    # if model_option == 'dpc-unet' or model_option == '3d-unet':
                    #     if args.channels == 10:
                    #         train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,:,:], ts_arr[:total_ts_len,-2:,:,:]), axis=1)
                    #     elif args.channels == 4:
                    #         train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:4,:,:], np.expand_dims(ts_arr[:total_ts_len,7,:,:], axis=1)), axis=1)
                    # else:
                        
                    if args.channels == 10:
                        train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,:,:], ts_arr[:total_ts_len,-2:,:,:]), axis=1)
                    elif args.channels == 4:
                        train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:4,:,:], np.expand_dims(ts_arr[:total_ts_len,7,:,:], axis=1)), axis=1)

                    # if ts_arr.shape[0] > 10:
                    #     train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,:,:], ts_arr[:total_ts_len,-2:,:,:]), axis=1)
                    # else:
                    #     if model_option == 'dpc-unet' or model_option == '3d-unet':
                    #         continue
                    #     else:
                    #         train_ts_set = np.concatenate((ts_arr[:,1:-4,:,:], ts_arr[:,-2:,:,:]), axis=1)

                    print('train ts shape before predicting: ', train_ts_set.shape)

                    temp_tif = prepare_data(args, train_ts_set)

                    print(f"Start predicting {ts_name} in {tile}!")

                    prediction = predict(model, batch_size, temp_tif, ref_im, args)
                    
                    data_dir = f'/projects/kwessel4/dpc/output/output-{args.model}-{today}/'
                    if not os.path.isdir(data_dir):
                        os.mkdir(data_dir)
                        
                        
                    ts_name_out  = ts_name + tile
                    
                    save_raster(ref_im, prediction, ts_name_out, data_dir)

                    print('prediction shape after final process: ', prediction.shape)
                    
                    plt.figure(figsize=(20,20))
                    plt.subplot(3,4,1)
                    plt.title("Image 1")
                    image = np.transpose(train_ts_set[0,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,2)
                    plt.title("Image 2")
                    image = np.transpose(train_ts_set[1,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,3)
                    plt.title("Image 3")
                    image = np.transpose(train_ts_set[2,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,4)
                    plt.title("Image 4")
                    image = np.transpose(train_ts_set[3,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))

                    plt.subplot(3,4,5)
                    plt.title("Image 5")
                    image = np.transpose(train_ts_set[4,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))

                    plt.subplot(3,4,6)
                    plt.title("Image 6")
                    image = np.transpose(train_ts_set[5,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,7)
                    plt.title("Image 7")
                    image = np.transpose(train_ts_set[6,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,8)
                    plt.title("Image 8")
                    image = np.transpose(train_ts_set[7,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,9)
                    plt.title("Image 9")
                    image = np.transpose(train_ts_set[8,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

                    plt.subplot(3,4,10)
                    plt.title("Image 10")
                    image = np.transpose(train_ts_set[9,1:4,:,:], (1,2,0))
                    image= rescale_image(xr.where(image > -9000, image, -1000))
                    plt.imshow(rescale_truncate(image))
                    # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")


                    plt.subplot(3,4,11)
                    plt.title("Segmentation Label")
                    image = mask_arr
                    plt.imshow(image)
                    # plt.savefig(f"{str(data_dir)}{ts_name}-label.png", dpi=300, bbox_inches='tight')

                    plt.subplot(3,4,12)
                    plt.title(f"Segmentation Prediction")
                    image = prediction
                    plt.imshow(image)
                    
                    if model_option == 'dpc-unet':
                        plt.savefig(f"{str(data_dir)}{ts_name_out}-{model_option}-{network}-{args.segment_model}-{today}.png", dpi=300, bbox_inches='tight')
                    else:
                        plt.savefig(f"{str(data_dir)}{ts_name_out}-{model_option}-{today}.png", dpi=300, bbox_inches='tight')

                    plt.close()
                    
                    accuracy, precision, recall, f1_score, iou = get_accuracy(prediction, mask_arr)
                    print(f'accuracy {accuracy} precision {precision} recall {recall} f1_score {f1_score} mIoU {iou}')
                    writer.writerow([all_keys[index], accuracy, precision, recall, f1_score, iou])

                    del prediction
                    del ref_im

        ## Predict large HLS
        else:

            key = list(file.keys())

            ts_arr = file[key[0]][()]
            
            if args.dataset == 'planet_2021':
                ref_im_fl = f"/projects/kwessel4/hls_large_refim/L15-0943E-1098N-01.tif"
            elif args.dataset == 'planet_etz_2021':
                ref_im_fl = f"/projects/kwessel4/hls_large_refim/L15-0941E-1106N-01.tif"
            else:
                ts_arr = np.transpose(ts_arr, (0,3,1,2))
                ref_im_fl = f"/projects/kwessel4/hls_large_refim/{tile[:3]}-2016.tif"
            
            ref_im = rxr.open_rasterio(ref_im_fl)
            
            print('ts_arr shape: ', ts_arr.shape)


            if ts_arr.shape[0] > args.ts_length:
                ts_arr = get_composite(ts_arr, args.ts_length)

            total_ts_len = args.ts_length # L

            padding_size = args.pad_size
            network = args.net
            model_option = args.model # 'dpc-unet' and 'unet', convlstm

            print(f'data dict {ts_name} ts shape: {ts_arr.shape}')

            
            if 'planet' not in args.dataset:
                if args.channels == 10:
                    train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,:,:], ts_arr[:total_ts_len,-2:,:,:]), axis=1)
                elif args.channels == 4:
                    train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:4,:,:], np.expand_dims(ts_arr[:total_ts_len,7,:,:], axis=1)), axis=1)
            else:
                train_ts_set = ts_arr


            print('train ts shape: ', train_ts_set.shape)

            temp_tif = prepare_data(args, train_ts_set)

            print(f"Start predicting {ts_name}!")

            prediction = predict(model, batch_size, temp_tif, ref_im, args)

            print('prediction shape after final process: ', prediction.shape)

            data_dir = f'/projects/kwessel4/dpc/output/output-{args.model}-{today}/'
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)

            save_raster(ref_im, prediction, ts_name, data_dir)

            plt.figure(figsize=(20,20))
            plt.subplot(1,2,1)
            plt.title("Image")
            image = np.transpose(train_ts_set[1,1:4,:,:], (1,2,0))
            image= rescale_image(xr.where(image > -9000, image, -1000))
            plt.imshow(rescale_truncate(image))


            plt.subplot(1,2,2)
            plt.title(f"Segmentation Prediction")
            image = prediction
            plt.imshow(image)

            if model_option == 'dpc-unet':
                plt.savefig(f"{str(data_dir)}{ts_name}-{model_option}-{network}-{args.segment_model}-{today}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{str(data_dir)}{ts_name}-{model_option}-{today}.png", dpi=300, bbox_inches='tight')

            plt.close()

            del prediction
            del ref_im


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/predict_sliding.py --gpu 0 --model convlstm --dataset Tappan13
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset Tappan15
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset Tappan15 --ts_length 30
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet-vae --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet --dataset PEV_2021
    # python models/predict_sliding.py --gpu 0 --model unet --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset PEV_2021
    # python models/predict_sliding.py --gpu 0 --model convlstm --dataset PEV_2021 --hidden_dim 200
    # python models/predict_sliding.py --gpu 0 --model 3d-unet --dataset Tappan01 --img_dim 16 --ts_length 16 --hidden_dim 160
    # python models/predict_sliding.py --gpu 0 --model 3d-unet --dataset PEV_2021 --img_dim 16 --ts_length 16 --hidden_dim 160
    # python models/predict_sliding.py --gpu 0 --model decision-tree --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet --dataset PEV_2021 --segment_model conv2d
