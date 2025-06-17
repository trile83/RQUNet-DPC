#import disstl.models as models
import re
from datetime import time
import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
import glob
from dpc.model_3d import *
from dpc.model_3d_unet import *
from dpc.model_3d_unet_stride import *
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy, calc_accuracy
from tqdm import tqdm
import argparse
import h5py
import math
import logging
import torchvision.utils as vutils
import cv2
import rioxarray as rxr
import time
import random
from datetime import date
from scipy import ndimage
from criterions.diceloss import DiceLoss
#from tensorboardX import SummaryWriter

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='unet', type=str, help='encoder for the DPC')
parser.add_argument('--model', default='dpc-unet', type=str, help='convlstm, dpc-unet, unet')
parser.add_argument('--dataset', default='Tappan01_WV02_20181217', type=str, help='PEV_2021, PFV_2021, or Tappan01, Tappan05')
parser.add_argument('--seq_len', default=4, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=16, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--num_chips', default=80, type=int)
parser.add_argument('--num_val', default=10, type=int)
parser.add_argument('--hidden_dim', default=200, type=int)
parser.add_argument('--standardization', default='local', type=str, help='local, global')
parser.add_argument('--normalization', default=15000, type=int)
parser.add_argument('--rescale', default='per-ts', type=str)
parser.add_argument('--segment_model', default='unet', type=str)
parser.add_argument('--loss', default='crossentropy', type=str)
parser.add_argument('--channels', default=10, type=int)
parser.add_argument('--crop_thresh', default=0.2, type=float, help='binary balance for crop class')
parser.add_argument('--noncrop_thresh', default=0.2, type=float, help='binary balance for non-crop class')
parser.add_argument('--noncrop_pct', default=0.7, type=float, help='percentage of image chips for mainly noncrop chips')
parser.add_argument('--addindices', default='False', type=str)

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
    def __init__(self, X, Y, Z):
        'Initialization'
        self.data = X
        self.mask = Y
        self.ts = Z

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask
        Z = self.ts

        return {
            'x': X,
            'mask': Y,
            'ts': Z
        }

class tsDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, Z):
        'Initialization'
        self.data = X
        self.mask = Y
        self.ts = Z

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]
        Z = self.ts[index]

        return {
            'ts': torch.tensor(X),
            'mask': torch.LongTensor(Y),
            'ori': Z
        }


class segmentDataset(Dataset):
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
            'x': X,
            'mask': Y
        }


def get_seq(sequence, seq_length):
    '''
    sequence: long time-series input
    seq_length: length of window for time-series chunk; default = 5
    return: array with all time-series chunk; prediction size =1, num_seq = 4
    '''
    (I,L,C,H,W) = sequence.shape
    all_arr = np.zeros((I,int(L//seq_length),seq_length,C,H,W))
    for j in range(I):
        for i in range(0,L+1-seq_length,seq_length):
            array = sequence[j,i:i+seq_length,:,:,:] # SL, C, H, W
            all_arr[j,math.ceil(i/seq_length)] = array


    # all_arr = np.lib.stride_tricks.sliding_window_view(sequence, seq_length, axis=1)

    print('all array shape: ', all_arr.shape)

    return all_arr.copy()

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
            if not array.any():
                print(f"i {i}")
            all_arr[j,i-num_seq] = array

        # for i in range(L1-num_seq): # same results
        #     array = windows[j,i:i+num_seq,:,:,:,:] # N, SL, C, H, W
        #     all_arr[j,i] = array

    return all_arr


def chipper(ts_stack, mask, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    mask: original ground truth with 7 classes that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = np.random.randint(0, h-input_size)
    j = np.random.randint(0, w-input_size)
    
    out_ts = np.array([ts_stack[:, :, i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask

def specific_chipper(ts_stack, mask, h_index, w_index, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = h_index
    j = w_index
    
    out_ts = np.array([ts_stack[:, :, i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask


def padding_ts(ts, mask, padding_size=10):
    '''
    Args:
        ts: time series input
        mask: ground truth
    Return:
        padded_ts
        padded_mask
    '''
    extra_top = extra_bottom = extra_left = extra_right = padding_size
    npad_ts = ((0, 0), (extra_top, extra_bottom), (extra_left, extra_right))
    npad_mask = ((extra_top, extra_bottom), (extra_left, extra_right))

    padded_ts = np.zeros((ts.shape[0],ts.shape[1],ts.shape[2]+padding_size*2,ts.shape[3]+padding_size*2))
    for i in range(ts.shape[0]):
        # pad border

        p_ts_i = np.copy(np.pad(ts[i], (npad_ts), mode='reflect'))

        padded_ts[i,:,:,:] = p_ts_i

        # print('padded_ts i',padded_ts.shape)

    plt.imshow(np.transpose(padded_ts[5,1:4,:,:], (1,2,0)))
    plt.savefig('/home/geoint/tri/dpc_test_out/train_input_im.png')
    plt.close()

    padded_mask = np.copy(np.pad(mask, (npad_mask), mode='constant', constant_values = 0))
    padded_mask = padded_mask.reshape((padded_mask.shape[0], padded_mask.shape[1]))

    del p_ts_i

    return padded_ts, padded_mask

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


def filtering_holes(mask_array):

    crop_array = mask_array.copy()
    #print("crop_array shape: ", crop_array.shape)
    crop_array[crop_array != 2] = 0
    crop_array[crop_array == 2] = 1

    crop_array_flt = ndimage.binary_fill_holes(
        crop_array,
        structure=np.ones((3,3))
    ).astype(int)

    new_array = crop_array_flt.copy()

    del crop_array_flt
    del mask_array

    return np.squeeze(new_array)


def cal_ndvi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2]))
    return ndvi

def cal_ndwi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.divide((image[:,:,1]-image[:,:,3]), (image[:,:,1]+image[:,:,3]))
    return ndwi

def cal_osavi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    osavi = np.divide(((1+0.16)*(image[:,:,3]-image[:,:,2])), (image[:,:,3]+image[:,:,2]+0.16))
    return osavi

def add_indices(data):

    data = np.transpose(data,(1,2,0))

    out_array = np.zeros((data.shape[0], data.shape[1], 7))
    ndvi = cal_ndvi(data)
    ndwi = cal_ndwi(data)
    osavi = cal_osavi(data)

    print('ndvi min: ', np.min(ndvi))
    print('ndvi max: ', np.max(ndvi))

    print('ndwi min: ', np.min(ndwi))
    print('ndwi max: ', np.max(ndwi))

    print('osavi min: ', np.min(osavi))
    print('osavi max: ', np.max(osavi))


    out_array[:,:,0] = data[:,:,0]
    out_array[:,:,1] = data[:,:,1]
    out_array[:,:,2] = data[:,:,2]
    out_array[:,:,3] = data[:,:,3]
    out_array[:,:,4] = ndvi
    out_array[:,:,5] = ndwi
    out_array[:,:,6] = osavi

    # print(out_array.shape)

    del data

    out_array = np.transpose(out_array,(2,0,1))

    # print(out_array.shape)

    return out_array


def get_train_set(args, list_ts):
    
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_inc_cloud.hdf5"

    ### UPDATE 09/01 - new datacube with small TS time series
    # if tile =='PEV':
    #     filename = "/projects/kwessel4/hls_datacube/hls-ecas-PEV-0901.hdf5"

    train_ts_set = []
    train_mask_set = []
    val_ts_set = []
    val_mask_set = []

    print('channels: ', args.channels)
    
    for ts_name in list_ts:

        ## temporary list to store data from each TS
        temp_ts_set = []
        temp_mask_set = []
    
        print("Get data from Tappan: ", ts_name)

        if "planet" in ts_name:
            tile='planet-ts32'
            filename = '/projects/kwessel4/hls_datacube/planet-tile13-train-0917.hdf5'

        if "SR" in ts_name:
            tile = 'TS01-SR'
            filename = '/projects/kwessel4/hls_datacube/Tappan01-sr-full-epoch2019.hdf5'

        ### UPDATE 09/01 - new datacube with small TS time series
        elif int(ts_name[6:8]) in [2,4,6,7,17]:
            tile ='PFV'
            filename = "/projects/kwessel4/hls_datacube/hls-PFV-epoch2019.hdf5"
        elif int(ts_name[6:8]) < 19:
            tile ='PEV'
            filename = "/projects/kwessel4/hls_datacube/hls-PEV-epoch2019.hdf5"
        elif int(ts_name[6:8]) == 21:
            tile='PDA'
            filename = "/projects/kwessel4/hls_datacube/hls-PDA-epoch2019.hdf5"
        elif int(ts_name[6:8]) < 25:
            tile ='PEA'
            filename = "/projects/kwessel4/hls_datacube/hls-PEA-epoch2019.hdf5"
        elif int(ts_name[6:8]) == 29:
            tile='PDB'
            filename = "/projects/kwessel4/hls_datacube/hls-PDB-epoch2019.hdf5"
        elif int(ts_name[6:8]) < 32:
            tile ='PCV'
            filename = "/projects/kwessel4/hls_datacube/hls-PCV-epoch2019.hdf5"
        elif int(ts_name[6:8]) >= 32:
            tile ='PGA'
            filename = "/projects/kwessel4/hls_datacube/hls-PGA-epoch2019.hdf5"


        if 'SR' not in ts_name:
            if int(ts_name[6:8]) == 25:
                args.crop_thresh = 0.05
                args.noncrop_thresh = 0.9
            elif int(ts_name[6:8]) == 6:
                args.crop_thresh = 0.05
                args.noncrop_thresh = 0.9
            elif int(ts_name[6:8]) == 29:
                args.crop_thresh = 0.05
                args.noncrop_thresh = 0.9
            elif int(ts_name[6:8]) == 21:
                args.crop_thresh = 0.4
                args.noncrop_thresh = 0.5

        print(tile)

        #### UPDATEs 09/01
        if 'SR' not in ts_name:
            with h5py.File(filename, "r") as file:
                ts_arr = file[f'{str(ts_name)}_{str(tile)}_ts'][()]
                mask_arr = file[f'{str(ts_name)}_{str(tile)}_mask'][()]
        else:
            with h5py.File(filename, "r") as file:
                ts_arr = file['Tappan01_ts'][()]

            mask_arr = np.squeeze(rxr.open_rasterio('/projects/kwessel4/nasa-multiyear-masks/epoch2019/Tappan01-2019.tif', mask=False).values)

        print("out ts arr shape: ", ts_arr.shape)
        print("out mask arr shape: ", mask_arr.shape)

        # if ts_arr.shape[0] > args.ts_length:
        #     ts_arr = get_composite(ts_arr, args.ts_length)

        print("out ts arr max pixel value: ", np.max(ts_arr))
        print("out ts arr min pixel value: ", np.min(ts_arr))

        # mask_arr = mask_arr[mask_arr != 7]
        # ts_arr = ts_arr[mask_arr != 7]

        input_size = args.img_dim
        total_ts_len = args.ts_length # L
        padding_size = args.pad_size

        # print('original ts array shape: ', ts_arr.shape)

        # nir = np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)
        # print('nir band ts array shape: ', nir.shape)

        if args.channels == 10:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:-4,:,:], ts_arr[:args.ts_length,-2:,:,:]), axis=1)
        elif args.channels == 4:
            if 'SR' not in ts_name:
                ts_arr = np.concatenate((ts_arr[:args.ts_length,1:4,:,:], np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)), axis=1)
            else:
                ts_arr = ts_arr
        elif args.channels == 7:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:4,:,:], np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)), axis=1)


        ## Added indices for 4 channels experiments
        if args.channels == 7:
            out_ts = np.zeros((ts_arr.shape[0],7,ts_arr.shape[2],ts_arr.shape[3]))
            for i in range(ts_arr.shape[0]):
                out_ts[i,:,:,:] = add_indices(ts_arr[i,:,:,:])


            ts_arr = out_ts
            del out_ts
            
        ## TEST: 12/11/2023
        if args.normalization is not None:
            if args.channels == 7:
                print('Got here with 4 channels')

                ts_arr[:,:4,:,:] = normalize_image(ts_arr[:,:4,:,:], args.normalization)

            else:

                ts_arr = normalize_image(ts_arr, args.normalization)

        print("out ts arr max pixel value after normalize: ", np.max(ts_arr))
        print("out ts arr min pixel value after normalize: ", np.min(ts_arr))

        # mask_ori_arr = mask_arr

        print("unique class in mask: ", np.unique(mask_arr, return_counts=True))

        binary_balance = args.crop_thresh
        include = True
        pct_noncrop = args.noncrop_pct

        print("percentage noncrop: ", args.noncrop_pct)

        generated_train_patches = 0
        noncrop_count = 0

        ## get train set
        while generated_train_patches < (args.num_chips):
            ts, mask = chipper(ts_arr[:,:,:,:], mask_arr, input_size=args.img_dim)

            ## generate noncrop chips
            if noncrop_count < args.num_chips*pct_noncrop:

                # first condition, tile must have valid classes
                if (ts.min() < -1 or mask.min() < 0):
                    #print('nodata condition not met: ', generated_patches)
                    continue

                ## mask does not contain nodata (class "2")
                if np.count_nonzero(mask==2) > 0:
                    #print('nodata condition not met: ', generated_patches)
                    continue

                if args.noncrop_thresh != 0 and np.count_nonzero(mask==0)< \
                            int(mask.shape[1]*
                                mask.shape[2]*
                                args.noncrop_thresh
                                ):
                    continue


                ### Visualization
                plt.figure(figsize=(20,20))
                plt.subplot(1,2,1)
                plt.title("Image")
                image = np.transpose(ts[0,0,1:4,:,:], (1,2,0))
                # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
                image = rescale_truncate(image)
                plt.imshow(image)
                plt.subplot(1,2,2)
                plt.title(f"Segmentation Label w {np.count_nonzero(mask == 0)/(mask.shape[1]*mask.shape[2])}% noncrop")
                image = np.transpose(mask[0,:,:], (0,1))
                plt.imshow(image)
                plt.savefig(f"/projects/kwessel4/dpc/output/image/training-noncrop-{str(generated_train_patches)}.png")
                plt.close()

                noncrop_count += 1

            else:

                # first condition, tile must have valid classes
                if (ts.min() < -1 or mask.min() < 0):
                    #print('nodata condition not met: ', generated_patches)
                    continue

                # second condition, mask does not contain nodata (class "2")
                if np.count_nonzero(mask==2) > 0:
                    #print('nodata condition not met: ', generated_patches)
                    continue
                    
                # condition, if include, number of labels must be at least 2
                if include and np.unique(mask).shape[0] < 2:
                    #print('mask unique values condition not met: ', generated_patches)
                    continue

                # balancing for binary classes, only applies to binary problem
                if args.crop_thresh != 0 and np.count_nonzero(mask == 1) < \
                        int(
                                mask.shape[1] *
                                mask.shape[2] *
                                args.crop_thresh
                        ):
                    #print('binary condition not met: ', generated_patches)
                    continue

                # balancing for binary classes, only applies to binary problem
                # if args.noncrop_thresh != 0 and np.count_nonzero(mask == 0) < \
                #         int(
                #                 mask.shape[1] *
                #                 mask.shape[2] *
                #                 args.noncrop_thresh
                #         ):
                #     #print('binary condition not met for background class: ', generated_patches)
                #     continue

            ts = np.squeeze(ts)

            ## TEST: 12/11/2023
            if args.rescale is not None:
                ts = rescale_image(ts,args.rescale)
            if args.standardization is not None or \
                    args.standardization != 'None':
                for frame in range(ts.shape[0]):
                    ts[frame] = standardize_image(ts[frame],args.standardization)

            ### Visualization
            # plt.figure(figsize=(20,20))
            # plt.subplot(1,2,1)
            # plt.title("Image")
            # image = np.transpose(ts[0,1:4,:,:], (1,2,0))
            # # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
            # image = rescale_truncate(image)
            # plt.imshow(image)
            # plt.subplot(1,2,2)
            # plt.title(f"Segmentation Label w {np.count_nonzero(mask == 1)/(mask.shape[1]*mask.shape[2])}")
            # image = np.transpose(mask[0,:,:], (0,1))
            # plt.imshow(image)
            # plt.savefig(f"/projects/kwessel4/dpc/output/image/training-{str(generated_train_patches)}-dpc-unet-{ts_name}-plot.png")
            # plt.close()

            
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

            # # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

            temp_ts_set.append(ts)
            temp_mask_set.append(mask)

            generated_train_patches += 1

        print(f'number of {args.noncrop_thresh} noncrop chips: ', noncrop_count)

        train_ts = np.stack(temp_ts_set, axis=0)
        train_mask = np.stack(temp_mask_set, axis=0)

        train_ts_set.append(train_ts)
        train_mask_set.append(train_mask)

        ## get validation set
        temp_ts_set = []
        temp_mask_set = []
        
        generated_val_patches = 0
        noncrop_count = 0
        while generated_val_patches < (args.num_val):
            ts, mask= chipper(ts_arr, mask_arr, input_size=args.img_dim)

            ## generate entirely noncrop chips
            if noncrop_count < args.num_val*pct_noncrop:

                # first condition, tile must have valid classes
                if (ts.min() < -0.6 or mask.min() < 0):
                    #print('nodata condition not met: ', generated_patches)
                    continue

                ## mask does not contain nodata (class "2")
                if np.count_nonzero(mask==2) > 0:
                    #print('nodata condition not met: ', generated_patches)
                    continue

                if args.noncrop_thresh != 0 and np.count_nonzero(mask==0)<\
                                                int(mask.shape[1]*
                                                    mask.shape[2]*
                                                    args.noncrop_thresh):
                    continue

                noncrop_count += 1
            else:

                # first condition, tile must have valid classes
                if (ts.min() < -0.6 or mask.min() < 0):
                    #print('nodata condition not met: ', generated_patches)
                    continue

                # second condition, mask does not contain nodata (class "7")
                if np.count_nonzero(mask==2) > 0:
                    #print('nodata condition not met: ', generated_patches)
                    continue
                    
                # condition, if include, number of labels must be at least 2
                if include and np.unique(mask).shape[0] < 2:
                    #print('mask unique values condition not met: ', generated_patches)
                    continue

                # balancing for binary classes, only applies to binary problem
                if args.crop_thresh != 0 and np.count_nonzero(mask == 1) < \
                        int(
                                mask.shape[1] *
                                mask.shape[2] *
                                args.crop_thresh
                        ):
                    #print('binary condition not met: ', generated_patches)
                    continue

                # balancing for binary classes, only applies to binary problem
                # if args.noncrop_thresh != 0 and np.count_nonzero(mask == 0) < \
                #         int(
                #                 mask.shape[1] *
                #                 mask.shape[2] *
                #                 args.noncrop_thresh
                #         ):
                #     #print('binary condition not met for background class: ', generated_patches)
                #     continue

            ts = np.squeeze(ts)

            ## TEST: 12/11/2023
            if args.rescale is not None:
                ts = rescale_image(ts,args.rescale)
            if args.standardization is not None or \
                    args.standardization != 'None':
                for frame in range(ts.shape[0]):
                    ts[frame] = standardize_image(ts[frame],args.standardization)

            mask = np.squeeze(mask)

            temp_ts_set.append(ts)
            temp_mask_set.append(mask)

            generated_val_patches += 1

        print('number of validation noncrop chips: ', noncrop_count)

        val_ts = np.stack(temp_ts_set, axis=0)
        val_mask = np.stack(temp_mask_set, axis=0)

        

        val_ts_set.append(val_ts)
        val_mask_set.append(val_mask)

    train_ts_set = np.concatenate(train_ts_set, axis=0)
    train_mask_set = np.concatenate(train_mask_set, axis=0)

    val_ts_set = np.concatenate(val_ts_set, axis=0)
    val_mask_set = np.concatenate(val_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    print(f"val ts set shape: {val_ts_set.shape}")
    print(f"val mask set shape: {val_mask_set.shape}")

    del temp_ts_set
    del temp_mask_set


    return train_ts_set, train_mask_set, val_ts_set, val_mask_set


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

    def early_stop(self, validation_loss, min_validation_loss):
        if validation_loss < min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    if args.channels == 4 and args.addindices == 'True':
        args.channels = 7

    print('channels: ', args.channels)
    

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.
    ### hls data
    #ts_name=args.dataset
    #tile='PEV'
    
    #list_ts = ['Tappan18_WV02_20170126']

    #list_ts = ['Tappan01_WV02_20181217']
    
    ### set 1 for 05-29 results
    # list_ts = [
    #             'Tappan18_WV02_20170126',
    #             'Tappan01_WV02_20181217',
    #             'Tappan19_WV03_20160617',
    #             'Tappan17_WV02_20181217',
    #             'Tappan02_WV02_20181217',
    #             'Tappan15_WV02_20160108',
    #             'Tappan16_WV02_20180508',
    #             'Tappan07_WV02_20190207',
    #             'Tappan13_WV03_20170226',
    #             'Tappan14_WV02_20161230',
    #             'Tappan23_WV02_20180119',
    #             'Tappan18_WV03_20160617'
    #             ]



    ### set 2 for 09-26

    # list_ts = [
    #             'Tappan18_WV02_20170126',
    #             'Tappan01_WV02_20181217',
    #             'Tappan15_WV02_20160108',
    #             'Tappan17_WV02_20181217',
    #             'Tappan16_WV02_20180508',
    #             'Tappan19_WV03_20160617',
    #             'Tappan20_WV02_20190127',
    #             'Tappan23_WV02_20180119',
    #             ]

    ## ETZ
    # list_ts = [
    #             'Tappan32_WV03_20160217',
    #             'Tappan01_WV02_20181217',
    #             'Tappan19_WV03_20160617',
    #             'Tappan18_WV02_20170126',
    #             'Tappan32_WV03_20170529',
    #             'Tappan24_WV02_20180119',
    #             'Tappan23_WV02_20170126',
    #             'Tappan20_WV02_20180328'
    #             ]

    # list_ts = [
    #             'Tappan32_WV03_20160217',
    #             'Tappan26_WV02_20200203',
    #             'Tappan19_WV03_20160617',
    #             'Tappan18_WV02_20170126',
    #             'Tappan32_WV03_20170529',
    #             'Tappan24_WV02_20180119',
    #             'Tappan23_WV02_20170126',
    #             'Tappan26_WV02_20171201'
    #             ]


    ### set 1 for 05-29 results
    # list_ts = [
    #             'Tappan18',
    #             'Tappan01',
    #             'Tappan19',
    #             'Tappan17',
    #             'Tappan02',
    #             'Tappan15',
    #             'Tappan16',
    #             'Tappan07',
    #             ]

    ### set 3 eetz for 12-23

    # list_ts = [
    #             'Tappan18',
    #             # 'Tappan01',
    #             # 'Tappan15',
    #             # 'Tappan17',
    #             # 'Tappan16',
    #             'Tappan19',
    #             'Tappan20',
    #             # 'Tappan23',
    #             'Tappan32',
    #             'Tappan33',
    #             # 'Tappan29',
    #             # 'Tappan25',
    #             # 'Tappan26',
    #             ]


    ## only wcas

    # list_ts = [
    #             'Tappan25',
    #             'Tappan26',
    #             ]


    ## all available tappan

    # list_ts = [
    #             # 'Tappan18',
    #             # 'Tappan01',
    #             # 'Tappan15',
    #             # 'Tappan17',
    #             # 'Tappan16',
    #             'Tappan19',
    #             'Tappan20',
    #             'Tappan23',
    #             'Tappan24',
    #             # 'Tappan06',
    #             'Tappan29',
    #             # 'Tappan25',
    #             # 'Tappan26',
    #             'Tappan21',
    #             'Tappan32'
    #             ]

    ### train with SR data (Tappan01)
    list_ts = [
                'TS01-SR'
                ]

    print('training image list: ', list_ts)

    #list_ts = ['Tappan15_WV02_20160108']

    input_size = args.img_dim

    print('time series length: ', args.ts_length)

    out_data_dir = '/scratch/mle35/dpc/data/'
    out_train_file = f'{out_data_dir}train_dpc_data_all_{len(list_ts)}ts_{args.crop_thresh}crop_{args.noncrop_pct}noncroppct_{args.channels}band.npy'
    out_val_file = f'{out_data_dir}val_dpc_data_all_{len(list_ts)}ts_{args.crop_thresh}crop_{args.noncrop_pct}noncroppct_{args.channels}band.npy'

    if os.path.isfile(out_train_file):
        ## load train set
        print("Load saved training files!")
        with open(out_train_file, 'rb') as f:
            train_set = np.load(f, allow_pickle=True)

        ## load val set
        with open(out_val_file, 'rb') as f:
            val_set = np.load(f, allow_pickle=True)


    else: # save training data out

        train_ts_set,train_mask_set,val_ts_set,val_mask_set = get_train_set(args, list_ts)

        ## train set
        train_seq = get_seq(train_ts_set, args.seq_len) #(I,L1,SL,C,H,W)
        # train_chunk = get_chunks(train_seq, args.num_seq) # (I,L2,N,SL,C,H,W); L2 = L-seq_len-num_seq
        print(f'train seq shape: {train_seq.shape}')

        ## validation set
        val_seq = get_seq(val_ts_set, args.seq_len) #(I,L1,SL,C,H,W)
        # val_chunk = get_chunks(val_seq, args.num_seq) # (I,L2,N,SL,C,H,W); L2 = L-seq_len-num_seq
        print(f'val seq shape: {val_seq.shape}') 

        (I,N,SL,C,H,W) = train_seq.shape
        train_set = tsDataset(train_seq, train_mask_set, train_ts_set)
        val_set = tsDataset(val_seq, val_mask_set, val_ts_set)

        with open(out_train_file, 'wb') as f:
            np.save(f, train_set, allow_pickle=True)

        with open(out_val_file, 'wb') as f:
            np.save(f, val_set, allow_pickle=True)


    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    val_loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=False, shuffle=False)
    train_dl = DataLoader(train_set, **loader_args)
    val_dl = DataLoader(val_set, **val_loader_args)
        
    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = args.net # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'

    # model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_1028_3band_unetvae_hls_65_2.7782891265815123e-07.pth'
    if network == 'unet':
        if input_size == 128:
            model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_64_5_8.505512028932572e-05.pth' # unet
        elif input_size == 64:
            if args.channels == 10:
                model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
            elif args.channels == 4:
                model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0315_4band_unet_hls_dim64_94_6.21471107006073e-05.pth'
            elif args.channels == 7:
                model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0315_7band_unet_hls_dim64_22_0.0002218632586300373.pth'

    elif network == 'unet-vae':
    	if input_size == 64:
            model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_64_102_6.290748715400695e-05.pth' # unet-vae
        
    elif network == 'rqunet-vae':
        if input_size == 128:
            model_checkpoint = '/projects/kwessel4/dpc/checkpoints/recon_0217_10band_unetvae_hls_128_5_6.96440190076828e-05.pth' # rqunet-vae

    # model = DPC_RNN_UNet(sample_size=input_size,
    #                 device=device,
    #                 num_seq=args.num_seq, 
    #                 seq_len=args.seq_len, 
    #                 network=network,
    #                 pred_step=1,
    #                 model_weight=model_checkpoint,
    #                 freeze=True)
    
    model = DPC_RNN(sample_size=input_size,
                    device=device,
                    num_seq=args.num_seq, 
                    seq_len=args.seq_len,
                    hidden_dim=args.hidden_dim,
                    network=network,
                    pred_step=1,
                    model_weight=model_checkpoint,
                    freeze=True,
                    segment_model=args.segment_model,
                    in_channels=args.channels)


    if torch.cuda.is_available():
        model = model.to(cuda)

    ## Apply class weights
    weights = [0.1,0.9]
    class_weights = torch.FloatTensor(weights).cuda()

    global criterion; 
    global criterion_type; 
    criterion_type = args.loss
    if criterion_type == "crossentropy":
        # criterion = models.losses.MultiTemporalCrossEntropy()
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == "focal_tverski":
        criterion = models.losses.FocalTversky()
    elif criterion_type == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif criterion_type == "dice":
        #criterion = models.losses.MultiTemporalDiceLoss()
        criterion = DiceLoss(mode='binary')

    print('Training Criterion Type: ',criterion_type)

    global iteration; iteration = 0
    best_acc = 0

    ### optimizer ###
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    ### load data ###

    print("Start training process!")

    tic = time.time()

    # setup tools
    global de_normalize; de_normalize = denorm()
    model_dir = "/scratch/mle35/dpc/train_checkpoints/"

    if network == 'unet-vae' or network == 'rqunet-vae' or network == 'unet':

        ### main loop ###
        # train_loss_lst = []
        # val_loss_lst = []
        train_loss_out = []
        val_loss_out = []

        train_losses = AverageMeter()
        val_losses = AverageMeter()

        print(f"length of dpc input training set {len(train_dl)}")
        min_loss = np.inf

        early_stopper = EarlyStopper(patience=20, min_delta=0.2)

        for epoch in range(args.start_epoch, args.epochs):
            # print('Epoch: ', epoch)
            for idx, input in enumerate(train_dl):

                input_ts = input['ts']
                input_mask = input['mask']
                ori_ts = input['ori']

                # print('original ts shape: ', ori_ts.shape)

                # (I,L2,N,SL,C,H,W) = input_ts.shape

                (I,N,SL,C,H,W) = input_ts.shape

                # input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

                train_set = satDataset(input_ts, input_mask, ori_ts)

                loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
                train_sat_dl = DataLoader(train_set, **loader_args_sat)

                output, train_loss = train_dpc(train_sat_dl, model, optimizer, network)
                # train_losses.update(train_loss.item(), I*L2)
                train_losses.update(train_loss.item(), I)

            for idx, input in enumerate(val_dl):
                input_ts_val = input['ts']
                input_mask_val = input['mask']
                ori_ts_val = input['ori']

                I_val = input_ts_val.size(0)

                # print(f'val input ts shape: {input_ts.shape}')
                # print(f'val input mask shape: {input_mask.shape}')

                # input_ts_val = rearrange(input_ts_val, "b l2 n sl c h w -> (b l2) n sl c h w")
                val_set = satDataset(input_ts_val, input_mask_val, ori_ts_val)

                loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=False, drop_last=False, shuffle=False)
                val_sat_dl = DataLoader(val_set, **loader_args_sat)
                output_val, val_loss = val_dpc(val_sat_dl, model, network)

                # saved loss value in list
                # train_loss_lst.append(train_loss)
                # val_loss_lst.append(val_loss)

                # train_losses.update(train_loss.item(), I*L2)
                # val_losses.update(val_loss.item(), I_val*L2)
                val_losses.update(val_loss.item(), I_val)

            # save check_point
            is_best = val_losses.local_avg < min_loss

            if is_best:
                min_loss = val_losses.local_avg

                # visualize predictions
                index_array = torch.argmax(output_val, dim=1)
                #index_array = output_val
                z = ori_ts_val.numpy()
                y = input_mask_val

                today = date.today()

                plt.figure(figsize=(20,20))
                plt.subplot(1,3,1)
                plt.title("Image")
                image = np.transpose(z[0,5,:3,:,:], (1,2,0))
                # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
                image = rescale_truncate(image)
                plt.imshow(image)
                # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-input.png")
                plt.subplot(1,3,2)
                plt.title("Segmentation Label")
                image = np.transpose(y[0,:,:], (0,1))
                plt.imshow(image)
                # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-label.png")
                plt.subplot(1,3,3)
                plt.title(f"Segmentation Prediction")
                image = np.transpose(index_array[0,:,:].cpu().numpy(), (0,1))
                plt.imshow(image)
                plt.savefig(f"/projects/kwessel4/dpc/output/image/best-{str(epoch)}-{str(idx)}-dpc-unet-{today}-plot.png")
                plt.close()


                # save dpc weights
                save_checkpoint({'epoch': epoch+1,
                                'net': args.net,
                                'state_dict': model.state_dict(),
                                'min_loss': min_loss,
                                'optimizer': optimizer.state_dict()}, 
                                is_best,
                                filename=os.path.join(model_dir, f'dpc-{network}-{today}-{criterion_type}_{args.segment_model}_std_{args.standardization}_{args.hidden_dim}_{np.round(min_loss,3)}_{args.crop_thresh}binary_{args.channels}band_epoch%s.pth' % str(epoch+1)),
                                keep_all=False)

            train_loss_out.append(train_losses.local_avg)
            val_loss_out.append(val_losses.local_avg)
            # print(f"train loss: {train_losses.local_avg}")
            # print(f"val loss: {val_losses.local_avg}")
            print(f"epoch: {epoch+1} train loss: {train_losses.local_avg} val loss: {val_losses.local_avg}")

            # early stopping
            if early_stopper.early_stop(val_losses.local_avg, min_loss):
                print("Stop at epoch:", epoch+1)
                break

        plt.plot(train_loss_out, color ="blue")
        plt.plot(val_loss_out, color = "red")
        plt.savefig("/projects/kwessel4/dpc/output/plot/train_loss_{today}.png")
        plt.close()

        print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
        print(f'Time required for training {time.time()-tic}')


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def _flatten_temporal_dim(preds, targets):
    # Flatten temporal dim if exists
    if preds.ndim > 2 and preds.ndim < 4:
        preds = rearrange(preds, "b t c -> (b t) c")
        targets = rearrange(targets, "b t -> (b t)")
    if preds.ndim > 4:
        preds = rearrange(preds, "b t c h w-> (b t) c h w")
        targets = rearrange(targets, "b t h w-> (b t) h w")
    return preds, targets

def train_dpc(data_loader, dpc_model, optimizer, network):
    losses = AverageMeter()
    dpc_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        tic = time.time()
        input_seq = input["x"]
        input_mask = input["mask"]

        # (B,L2,N,SL,C,H,W) = input_seq.shape

        (B,N,SL,C,H,W) = input_seq.shape

        #print('mask tensor unique value: ', torch.unique(input_mask, return_counts=True))
        #print('input mask shape: ', input_mask.shape)

        ## if using dice loss, need to change input_mask shape:
        if criterion_type == 'BCE':
            input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
            input_mask = input_mask.squeeze(0)
            input_mask = torch.permute(input_mask, (0,3,1,2))
        elif criterion_type == 'dice':
            input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
            input_mask = input_mask.squeeze(0)
            input_mask = torch.permute(input_mask, (0,3,1,2))
        else:
            input_mask = input_mask.view(input_mask.shape[1], H, W)

        input_seq = input_seq.to(cuda, dtype=torch.float32)

        if criterion_type == 'dice' or criterion_type == 'BCE':
            input_mask = input_mask.to(cuda, dtype=torch.float32)
        else:
            input_mask = input_mask.to(cuda, dtype=torch.long)

        # input_mask = input_mask.view(input_mask.shape[1], H, W)
        B = input_seq.size(0)

        # if network == 'unet-vae' or network == 'rqunet-vae-encoder' or network == 'unet':
        output, _ = dpc_model(input_seq)

        #output, input_mask = _flatten_temporal_dim(output, input_mask)
        
        #print('model output shape before loss: ', output.detach().cpu().numpy().shape)

        loss = criterion(output, input_mask)

        losses.update(loss.item(), B)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return output, losses.local_avg

def val_dpc(data_loader, dpc_model, network):
    losses = AverageMeter()
    dpc_model.eval()
    global iteration

    with torch.no_grad():
        for idx, input in enumerate(data_loader):

            tic = time.time()
            input_seq = input["x"]
            input_mask = input["mask"]

            # (B,L2,N,SL,C,H,W) = input_seq.shape

            (B,N,SL,C,H,W) = input_seq.shape

            # print(f'input mask shape : {input_mask.shape}') # 1 x batch x input_size x input_size
            if criterion_type == 'BCE' or criterion_type == 'dice':
                input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
                input_mask = input_mask.squeeze(0)
                input_mask = torch.permute(input_mask, (0,3,1,2))
            else:
                input_mask = input_mask.view(input_mask.shape[1], H, W)

            input_seq = input_seq.to(cuda, dtype=torch.float32)
            
            if criterion_type == 'dice' or criterion_type == 'BCE':
                input_mask = input_mask.to(cuda, dtype=torch.float32)
            else:
                input_mask = input_mask.to(cuda, dtype=torch.long)

            B = input_seq.size(0)

            # if network == 'unet-vae' or network == 'rqunet-vae-encoder':
            output, _ = dpc_model(input_seq)

            loss = criterion(output, input_mask)
            losses.update(loss.item(), B)

    return output, losses.local_avg

if __name__ == '__main__':
    main()

    torch.cuda.empty_cache()
    # python models/train_dpc_seg.py --net unet-vae --dataset Tappan01 --epochs 100
