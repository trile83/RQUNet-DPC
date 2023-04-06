import disstl.models as models
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
from tqdm import tqdm
import argparse
import h5py
import logging
import cv2
import csv
from sklearn.metrics import jaccard_score, balanced_accuracy_score, confusion_matrix, classification_report
import rioxarray as rxr
import rasterio
import xarray as xr
from inference import inference
from tensorboardX import SummaryWriter
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from benchmod.convgru import ConvGRU_Seg
from unet3d.unet3d import UNet3D
from unet.unet_test import UNet_test

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='unet', type=str, help='encoder for the DPC')
parser.add_argument('--model', default='dpc-unet', type=str, help='convlstm, dpc-unet, unet')
parser.add_argument('--dataset', default='Tappan01', type=str, help='PEV_2021, PFV_2021, or Tappan01, Tappan05')
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
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--standardization', default='local', type=str)
parser.add_argument('--normalization', default=0.0001, type=float)
parser.add_argument('--rescale', default='per-ts', type=str)


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
    accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    # iou_1 = jaccard_score(y_pred, y_true, average=None)
    iou = jaccard_score(y_pred, y_true, pos_label=1, average='binary')
    precision = report['cropland']['precision']
    recall = report['cropland']['recall']
    f1_score = report['cropland']['f1-score']
    return accuracy, precision, recall, f1_score, iou


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ts_name=args.dataset
    if 'PEV' in args.dataset or 'PFV' in args.dataset:
        hls = True
    else:
        hls = False

    if not hls:

        if ts_name == 'Tappan14' or ts_name == 'Tappan15':
            filename = "/home/geoint/tri/hls_ts_video/hls_data_1415.hdf5"
        else:
            filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"

        # filename = "/home/geoint/tri/hls_ts_video/hls_data_inc_cloud.hdf5"

        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            if ts_name == 'Tappan06' or ts_name == 'Tappan07':
                ts_arr = file[f'{str(ts_name)}_PFV_ts'][()]
                mask_arr = file[f'{str(ts_name)}_mask'][()]

                ref_im_fl = f"/home/geoint/tri/hls/{str(ts_name)}_HLS.S30.T28PFV.2021179T112119.v2.0.tif"
                ref_im = rxr.open_rasterio(ref_im_fl)
            else:
                ts_arr = file[f'{str(ts_name)}_PEV_ts'][()]
                mask_arr = file[f'{str(ts_name)}_mask'][()]

                ref_im_fl = f"/home/geoint/tri/hls/{str(ts_name)}_HLS.S30.T28PEV.2020275T112119.v2.0.tif"
                ref_im = rxr.open_rasterio(ref_im_fl)

    else:
        # ts_name = 'PEV_2021'
        filename = "/home/geoint/tri/hls_ts_video/hls_data_all.hdf5"
        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            ts_arr = file[f'{str(ts_name)}_ts'][()]
            mask_arr = file[f'{str(ts_name)}_ts'][()]

        ts_arr = np.transpose(ts_arr, (0,3,1,2))
        mask_arr = ts_arr


        if 'PFV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PFV.2021081T110731.v2.0.tif'

        if 'PEV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2022009T112441.v2.0.tif'

        ref_im = rxr.open_rasterio(ref_im_fl)

        print('reference image shape: ', ref_im.shape)

    input_size = args.img_dim
    total_ts_len = args.ts_length # L

    padding_size = args.pad_size

    print(f'data dict {ts_name} ts shape: {ts_arr.shape}')
    print(f'data dict {ts_name} mask shape: {mask_arr.shape}')


    if ts_arr.shape[0] > 9:
        # train_ts_set = ts_arr[:total_ts_len,1:-2,:,:]
        train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,:,:], ts_arr[:total_ts_len,-2:,:,:]), axis=1)
    else:
        train_ts_set = np.concatenate((ts_arr[:,1:-4,:,:], ts_arr[:,-2:,:,:]), axis=1)

    print(train_ts_set.shape)
        
    
    # ignore the no-data edge of cut Tappan Square to HLS
    if not hls:
        if ts_name == 'Tappan02' or ts_name == 'Tappan04':
            # train_ts_set = ts_arr[:,1:-2,1:-2,1:160]
            train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,1:-2,1:160], ts_arr[:total_ts_len,-2:,1:-2,1:160]), axis=1)
            mask_arr = mask_arr[1:-2,1:160]
        elif ts_name == 'Tappan05':
            # train_ts_set = ts_arr[:,1:-2,1:300,1:-2]
            train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,1:300,1:-2], ts_arr[:total_ts_len,-2:,1:300,1:-2]), axis=1)
            mask_arr = np.squeeze(
                rxr.open_rasterio('/home/geoint/tri/senegal_hls_mask/Tappan05_WV02_20110430.tiff').values
                )
            mask_arr = mask_arr[1:300,1:-1]
            mask_arr[mask_arr==1]=0
            mask_arr[mask_arr==2]=1
            mask_arr[mask_arr==3]=0
            mask_arr[mask_arr==4]=0
            mask_arr[mask_arr==5]=0
            mask_arr[mask_arr==7]=0
            plt.imshow(mask_arr)
            plt.savefig('/home/geoint/tri/dpc/tappan05.png')
            plt.close()
            # print(mask_arr.shape)
        else:
            train_ts_set = np.concatenate((ts_arr[:total_ts_len,1:-4,1:-2,1:-2], ts_arr[:total_ts_len,-2:,1:-2,1:-2]), axis=1)
            mask_arr = mask_arr[1:-2,1:-2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_option = args.model # 'dpc-unet' and 'unet', convlstm

    standardization = args.standardization
    normalization = args.normalization

    if model_option == 'unet':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        unet_segment = UNet_test(num_classes=2,segment=True,in_channels=10)

        ### 10 bands
        # unetsegment_checkpoint = f'{str(model_dir)}unet_meanframe_ts01_1train_9band_epoch_155.pth'

        # unetsegment_checkpoint = f'{str(model_dir)}unet_meanframe_ts01_0322_10band_epoch_93.pth'

        unetsegment_checkpoint = f'{str(model_dir)}unet_meanframe_ts01_0322_cloud_10band_epoch_23.pth'
        if torch.cuda.is_available():
            unet_segment = unet_segment.to(cuda)

        unet_segment.load_state_dict(torch.load(unetsegment_checkpoint)['state_dict'])

        if args.rescale == 'per-ts':
            train_ts_set = rescale_image(train_ts_set, args.rescale)

        xraster = train_ts_set[:args.ts_length,:,:,:].mean(axis=0)

        if hls:
            temporary_tif = xr.where(xraster > -1000, xraster, 2000) # 2000 is the optimal value for the nodata
        else:
            temporary_tif = xr.where(xraster > -9000, xraster, 2000)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=unet_segment,
            n_classes=args.num_classes,
            overlap=0.5,
            batch_size=32,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option
        )

    elif model_option == '3d-unet':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        model = UNet3D(in_channel=10, n_classes=args.num_classes)

        model_checkpoint = f'{str(model_dir)}3d-unet_0405_10band_epoch_40.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set[:,:,:,:]

        if hls:
            # 2000 is the optimal value for the nodata
            # xraster = xr.where(xraster[:10,:,:,:] > -9000, xraster[:10,:,:,:], 1000)
            xraster = xraster[:args.ts_length,:,:,:]
            temporary_tif = xr.where(xraster > -9000, xraster, 1000) 
        else:
            xraster = xraster[:args.ts_length,:,:,:]
            temporary_tif = xr.where(xraster > -9000, xraster, 1000)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=args.num_classes,
            overlap=0.5,
            batch_size=16,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option
        )
    
    elif model_option == 'convlstm':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=160,
            input_dim=10,
            kernel_size=(3, 3)
            )
 
        ### 10 bands
        # model_checkpoint = f'{str(model_dir)}convlstm_10band_epoch_188.pth'
        model_checkpoint = f'{str(model_dir)}convlstm_new_10band_epoch_56.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set

        if hls:
            # 2000 is the optimal value for the nodata
            xraster = xr.where(xraster[:args.ts_length,:,:,:] > -9000, xraster[:args.ts_length,:,:,:], 1000)
            temporary_tif = xraster
        else:
            xraster = xraster[:args.ts_length,:,:,:]
            temporary_tif = xr.where(xraster > -9000, xraster, 1000)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=2,
            overlap=0.5,
            batch_size=8,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option
        )

    elif model_option == 'convgru':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        model = ConvGRU_Seg(
                num_classes=args.num_classes,
                input_size=(input_size,input_size),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
 
        ### 10 bands
        # model_checkpoint = f'{str(model_dir)}convgru_10band_epoch_90.pth'
        # model_checkpoint = f'{str(model_dir)}convgru_new_10band_epoch_66.pth'
        # model_checkpoint = f'{str(model_dir)}convgru_0320_10band_epoch_36.pth'

        # model_checkpoint = f'{str(model_dir)}convgru_0320_10band_norm_epoch_197.pth'
        model_checkpoint = f'{str(model_dir)}convgru_0320_10band_norm_epoch_153.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set

        if hls:
            # 2000 is the optimal value for the nodata
            # xraster = xr.where(xraster[:10,:,:,:] > -9000, xraster[:10,:,:,:], 1000)
            xraster = xraster[:args.ts_length,:,:,:]
            temporary_tif = xr.where(xraster > -9000, xraster, 1000) 
        else:
            xraster = xraster[:args.ts_length,:,:,:]
            temporary_tif = xr.where(xraster > -9000, xraster, 1000)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=2,
            overlap=0.5,
            batch_size=8,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option
        )

    elif model_option == 'dpc-unet':

        network = args.net # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'
        if network == 'unet':
            encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
        else:
            encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0217_10band_unetvae_hls_64_97_2.0649683759061257e-05.pth' # unet-vae

        # model = DPC_RNN_UNet(sample_size=input_size,
        #                 device=device,
        #                 num_seq=4, 
        #                 seq_len=6, 
        #                 network=network,
        #                 pred_step=1,
        #                 model_weight=encoder_weight,
        #                 freeze=True)
        
        model = DPC_RNN(sample_size=input_size,
                    device=device,
                    num_seq=args.num_seq, 
                    seq_len=args.seq_len, 
                    network=network,
                    pred_step=1,
                    model_weight=encoder_weight,
                    freeze=True)

        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"

        ### 10 bands
        if network == 'unet':
            # model_checkpoint = f'{str(model_dir)}dpc-unet_9band_epoch24.pth'
            # model_checkpoint = f'{str(model_dir)}dpc-unet-unet-encoder-0306_10band_ts01_epoch14.pth'

            ## classification layer
            # model_checkpoint = f'{str(model_dir)}dpc-rnn-unet-encoder-0319_10band_ts01_epoch14.pth'

            ## unet segment
            model_checkpoint = f'{str(model_dir)}dpc-rnn-unet-encoder-0405-stride_10band_ts01_epoch28.pth'

            ## cloud
            # model_checkpoint = f'{str(model_dir)}dpc-rnn-unet-encoder-0323-cloud-stride_10band_ts01_epoch7.pth'
        else:
            # model_checkpoint = f'{str(model_dir)}dpc-unet_10band_ts01_epoch7.pth'
            # model_checkpoint = f'{str(model_dir)}dpc-unet-unet-vae-encoder-0307_10band_ts01_epoch14.pth'
            model_checkpoint = f'{str(model_dir)}dpc-unet-unet-vae-encoder-0317_10band_ts01_epoch18.pth'

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        # model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model = model.to(cuda)

        ### restart training ###
        ### load data ###

        print("Start prediction!")

        # setup tools
        global de_normalize; de_normalize = denorm()
        
        ### main loop ###
        print("Start segmentation!")

        ## visualize

        # xraster = rescale_image(train_ts_set[:,:,:,:]) # previously works 02-15
        xraster = train_ts_set[:,:,:,:]

        # print('ts xraster min', np.min(xraster))
        
        if hls:
            temporary_tif = xr.where(xraster > -9000, xraster, 2000) # 2000 is the optimal value for the nodata
        else:
            temporary_tif = xraster[:args.ts_length]
            temporary_tif = xr.where(temporary_tif > -9000, temporary_tif, 2000)

            print('temp tif shape: ', temporary_tif.shape)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=2,
            overlap=0.5,
            batch_size=1,
            standardization=standardization,
            mean=0,
            std=0,
            normalize=normalization,
            rescale=args.rescale,
            model_option=model_option
        )

    ref_im = ref_im.transpose("y", "x", "band")
    print(f'ref im shape: {ref_im.shape}')
    (h,w,c) = ref_im.shape

    if prediction.shape[0] > 1:
        prediction = np.argmax(prediction, axis=0)
    else:
        prediction = np.squeeze(
            np.where(prediction > 0.5, 1, 0).astype(np.int16)
        )

    if not hls:
        accuracy, precision, recall, f1_score, iou = get_accuracy(prediction, mask_arr)
        print(f'accuracy {accuracy} precision {precision} recall {recall} f1_score {f1_score} mIoU {iou}')

    print('prediction shape after final process: ', prediction.shape)

    data_dir = '/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/'

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    image = np.transpose(train_ts_set[5,:3,:,:], (1,2,0))
    if hls:
        image= rescale_image(xr.where(image > -9000, image, 600))
    else:
        image= rescale_image(xr.where(image > -1, image, 15))
    # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
    plt.imshow(rescale_truncate(image))
    # # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")
    # plt.subplot(1,3,2)
    # plt.title("Segmentation Label")
    # # image = np.transpose(train_mask_set[:,:], (0,1))
    # image = train_mask_set
    # plt.imshow(image)
    # # plt.savefig(f"{str(data_dir)}{ts_name}-label.png")

    plt.subplot(1,2,2)
    plt.title(f"Segmentation Prediction")
    image = prediction
    plt.imshow(image)
    if model_option == 'dpc-unet':
        plt.savefig(f"{str(data_dir)}{ts_name}-{model_option}-{network}-cloud.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{str(data_dir)}{ts_name}-{model_option}-cloud.png", dpi=300, bbox_inches='tight')

    plt.close()

    #save Tiff file output
    # Drop image band to allow for a merge of mask
    ref_im = ref_im.drop(
        dim="band",
        labels=ref_im.coords["band"].values[1:],
        drop=True
    )

    if hls:
        prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='otcb',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('otcb')
        prediction.attrs['model_name'] = (model_option)
        prediction = prediction.transpose("band", "y", "x")

        # Set nodata values on mask
        nodata = prediction.rio.nodata
        prediction = prediction.where(ref_im != nodata)
        prediction.rio.write_nodata(
            255, encoded=True, inplace=True)

        # TODO: ADD CLOUDMASKING STEP HERE
        # REMOVE CLOUDS USING THE CURRENT MASK

        # Save COG file to disk
        prediction.rio.to_raster(
            f'{data_dir}{ts_name}-{model_option}.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
            driver='GTiff',
            dtype='uint8'
        )


def predict_dpc(data_loader, dpc_model):

    dpc_model.eval()
    global iteration

    feature_lst = []

    for idx, input in enumerate(data_loader):
        tic = time.time()
        input_seq = input["x"]

        (B,N,SL,C,H,W) = input_seq.shape
        input_seq = input_seq.to(cuda, dtype=torch.float32)
        B = input_seq.size(0)
        features = dpc_model(input_seq)
        feature_lst.append(features)

    feature_arr = torch.cat(feature_lst, dim=0)

    return feature_arr.cpu().detach()


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/predict_sliding.py --gpu 0 --model convlstm --dataset Tappan13
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset Tappan15
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset Tappan15 --ts_length 30
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet-vae --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model unet --dataset Tappan01
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset PEV_2021
    # python models/predict_sliding.py --gpu 0 --model 3d-unet --dataset Tappan01 --img_dim 16 --ts_length 16
