#import disstl.models as models
import re
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
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from benchmod.convgru import ConvGRU_Seg
from unet3d.unet3d import UNet3D
from tqdm import tqdm
import argparse
import h5py
import logging
import cv2
import rioxarray as rxr
from datetime import date
from scipy import ndimage
from criterions.diceloss import DiceLoss

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='convgru', type=str, help='convlstm, convgru')
parser.add_argument('--dataset', default='Tappan01_WV02_20181217', type=str, help='Tappan01, Tappan05')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=3e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=16, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_chips', default=100, type=int)
parser.add_argument('--num_val', default=20, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--standardization', default='local', type=str)
parser.add_argument('--normalization', default=15000, type=float)
parser.add_argument('--rescale', default='per-ts', type=str)
parser.add_argument('--loss', default='crossentropy', type=str)
parser.add_argument('--channels', default=10, type=int)
parser.add_argument('--crop_thresh', default=0.3, type=float, help='binary balance for crop class')
parser.add_argument('--noncrop_thresh', default=0.5, type=float, help='binary balance for non-crop class')
parser.add_argument('--noncrop_pct', default=0.1, type=float, help='percentage of image chips for mainly noncrop chips')
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

def get_composite(ts_arr, ts_len=10):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[0] // ts_len

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[0], step):
        out_lst.append(ts_arr[i])

    out_array = np.stack(out_lst, axis=0)
    del ts_arr

    return out_array


def filtering_holes(mask_array):

    crop_array = mask_array.copy()
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
    
    for ts_name in list_ts:

        ## temporary list to store data from each TS
        temp_ts_set = []
        temp_mask_set = []
    
        print("Get data from Tappan: ", ts_name)

        ### UPDATE 09/01 - new datacube with small TS time series
        if int(ts_name[6:8]) in [2,4,6,7,17]:
            tile ='PFV'
            filename = "/projects/kwessel4/hls_datacube/hls-PFV-epoch2019.hdf5"
        elif int(ts_name[6:8]) < 19:
            tile ='PEV'
            filename = "/projects/kwessel4/hls_datacube/hls-PEV-epoch2019.hdf5"
        elif int(ts_name[6:8]) < 32:
            tile ='PEA'
            filename = "/projects/kwessel4/hls_datacube/hls-PEA-epoch2019.hdf5"
        elif int(ts_name[6:8]) >= 32:
            tile ='PGA'
            filename = "/projects/kwessel4/hls_datacube/hls-PGA.hdf5"

        #### UPDATEs 09/01
        with h5py.File(filename, "r") as file:
            ts_arr = file[f'{str(ts_name)}_{str(tile)}_ts'][()]
            mask_arr = file[f'{str(ts_name)}_{str(tile)}_mask'][()]

        print("out ts arr shape: ", ts_arr.shape)

        # if ts_arr.shape[0] > args.ts_length:
        #     ts_arr = get_composite(ts_arr, args.ts_length)

        print("out ts arr max pixel value: ", np.max(ts_arr))
        print("out ts arr min pixel value: ", np.min(ts_arr))

        # mask_arr = mask_arr[mask_arr != 7]
        # ts_arr = ts_arr[mask_arr != 7]

        input_size = args.img_dim
        total_ts_len = args.ts_length # L
        padding_size = args.pad_size

        print('original ts array shape: ', ts_arr.shape)

        # nir = np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)
        # print('nir band ts array shape: ', nir.shape)

        if args.channels == 10:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:-4,:,:], ts_arr[:args.ts_length,-2:,:,:]), axis=1)
        elif args.channels == 4:
            ts_arr = np.concatenate((ts_arr[:args.ts_length,1:4,:,:], np.expand_dims(ts_arr[:args.ts_length,7,:,:], axis=1)), axis=1)
            
        ## TEST: 12/11/2023
        if args.normalization is not None:
            ts_arr = normalize_image(ts_arr, args.normalization)

        print("out ts arr max pixel value after normalize: ", np.max(ts_arr))
        print("out ts arr min pixel value after normalize: ", np.min(ts_arr))

        # mask_ori_arr = mask_arr

        print("unique class in mask: ", np.unique(mask_arr, return_counts=True))

        binary_balance = args.crop_thresh
        include = True
        pct_noncrop = args.noncrop_pct

        generated_train_patches = 0
        noncrop_count = 0

        ## get train set
        while generated_train_patches < (args.num_chips):
            ts, mask = chipper(ts_arr[:,:,:,:], mask_arr, input_size=args.img_dim)

            ## generate noncrop chips
            if noncrop_count < args.num_chips*pct_noncrop:

                # first condition, tile must have valid classes
                if (ts.min() < -100 or mask.min() < 0):
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
                if (ts.min() < -100 or mask.min() < 0):
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
                if (ts.min() < -100 or mask.min() < 0):
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
                if (ts.min() < -100 or mask.min() < 0):
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

        print('number of entirely noncrop chips: ', noncrop_count)

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

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.

    #list_ts = ['Tappan18_WV02_20170126']
    # list_ts = [
    #             'Tappan18_WV02_20170126',
    #             'Tappan01_WV02_20181217',
    #             'Tappan19_WV03_20160617',
    #             'Tappan17_WV02_20181217',
    #             'Tappan02_WV02_20181217',
    #             'Tappan15_WV02_20160108',
    #             'Tappan16_WV02_20180508',
    #             'Tappan07_WV02_20190207',
    #             # 'Tappan13_WV03_20170226',
    #             # 'Tappan14_WV02_20161230',
    #             # 'Tappan23_WV02_20180119',
    #             # 'Tappan18_WV03_20160617'
    #             ]

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

    ### set 1 

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


    ### set 2 for 09-26

    list_ts = [
                'Tappan18',
                'Tappan01',
                'Tappan15',
                'Tappan17',
                'Tappan16',
                'Tappan19',
                'Tappan20',
                'Tappan23',
                'Tappan24',
                'Tappan06',
                # 'Tappan29',
                # 'Tappan25',
                # 'Tappan26',
                # 'Tappan05'
                ]

                
    #ts_name=args.dataset
    #tile='PEV'

    input_size = args.img_dim

    # out_data_dir = '/scratch/mle35/dpc/data/'
    # out_train_file = f'{out_data_dir}train_dpc_data_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'
    # out_val_file = f'{out_data_dir}val_dpc_data_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'

    # train_ts_set, train_mask_set, val_ts_set,val_mask_set = get_train_set(args, list_ts)


    out_data_dir = '/scratch/mle35/dpc/data/'
    out_train_file = f'{out_data_dir}train_benchmod_data_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'
    out_val_file = f'{out_data_dir}val_benchmod_data_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'

    out_train_m_file = f'{out_data_dir}train_benchmod_mask_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'
    out_val_m_file = f'{out_data_dir}val_benchmod_mask_{len(list_ts)}ts_{args.crop_thresh}binary_{args.channels}band.npy'


    if os.path.isfile(out_train_file):
        ## load train set
        print("Load saved training files!")
        with open(out_train_file, 'rb') as f:
            train_ts_set = np.load(f, allow_pickle=True)

        ## load val set
        with open(out_val_file, 'rb') as f:
            val_ts_set = np.load(f, allow_pickle=True)


        with open(out_train_m_file, 'rb') as f:
            train_mask_set = np.load(f, allow_pickle=True)

        ## load val set
        with open(out_val_m_file, 'rb') as f:
            val_mask_set = np.load(f, allow_pickle=True)


    else: # save training data out

        train_ts_set,train_mask_set,val_ts_set,val_mask_set = get_train_set(args, list_ts)

        with open(out_train_file, 'wb') as f:
            np.save(f, train_ts_set, allow_pickle=True)

        with open(out_val_file, 'wb') as f:
            np.save(f, val_ts_set, allow_pickle=True)

        with open(out_train_m_file, 'wb') as f:
            np.save(f, train_mask_set, allow_pickle=True)

        with open(out_val_m_file, 'wb') as f:
            np.save(f, val_mask_set, allow_pickle=True)


    ### model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_option = args.model
    if model_option == "convlstm":
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=200,
            input_dim=args.channels,
            kernel_size=(3, 3)
            )
    elif model_option == "convgru":
        model = ConvGRU_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            input_dim=args.channels,
            kernel_size=(3, 3),
            hidden_dim=180,
            )
    elif model_option == '3d-unet':
        model = UNet3D(in_channel=args.channels, n_classes=args.num_classes)

    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)

    global criterion; 
    global criterion_type;
    criterion_type = args.loss
    if criterion_type == "crossentropy":
        # criterion = models.losses.MultiTemporalCrossEntropy()
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == "focal_tverski":
        criterion = models.losses.FocalTversky()
    elif criterion_type == "dice":
        criterion = DiceLoss(mode='binary')

    ### optimizer ###
    # params = model.parameters()
    # optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0)

    segment_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    ### load data ###

    print("Start training process!")
    tic = time.time()

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
    model_dir = "/scratch/mle35/dpc/train_checkpoints/"
    
    ### main loop ###
    train_loss_lst = []
    val_loss_lst = []

    # print(f"train mask set shape: {train_mask_set.shape}")

    train_seg_set = tsDataset(train_ts_set, train_mask_set)
    val_seg_set = tsDataset(val_ts_set, val_mask_set)
    loader_args_1 = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_segment_dl = DataLoader(train_seg_set, **loader_args_1)
    val_segment_dl = DataLoader(val_seg_set, **loader_args_1)

    print(f"Length of input training set {len(train_segment_dl)}")
    print("Start segmentation training!")

    best_acc = 0
    min_loss = np.inf

    early_stopper = EarlyStopper(patience=20, min_delta=0.2)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_segment_dl, model, segment_optimizer)
        val_loss, im, mask, mask_pred = val(val_segment_dl, model, epoch)

        # saved loss value in list
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)

        print(f"epoch: {epoch+1} train loss: {train_loss} val loss: {val_loss}")
        # print(f"val loss: {val_loss}")

        # save check_point
        is_best = val_loss < min_loss

        if is_best:
            min_loss = val_loss

            # output predictions
            index_array = torch.argmax(mask_pred, dim=1)

            today = date.today()

            plt.figure(figsize=(20,20))
            plt.subplot(1,3,1)
            plt.title("Image")
            x = im
            y = mask
            image = np.transpose(x[0,5,:3,:,:].numpy(), (1,2,0))
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
            plt.savefig(f"/projects/kwessel4/dpc/output/image/best-{epoch}-{model_option}-{today}-pred.png")
            plt.close()

            # save unet segment weights
            save_checkpoint({'epoch': epoch+1,
                            'net': args.net,
                            'state_dict': model.state_dict(),
                            'min_loss': min_loss,
                            'optimizer': segment_optimizer.state_dict()}, 
                            is_best, filename=\
                                os.path.join(model_dir, \
                                    f'{model_option}_{today}_{args.channels}band_{np.round(min_loss,3)}_epoch_{str(epoch+1)}.pth'),
                                keep_all=False)
            
        # early stopping
        if early_stopper.early_stop(val_loss, min_loss):
            print("Stop at epoch:", epoch+1)
            break
        
    # plt.plot(train_loss_lst, color ="blue")
    # plt.plot(val_loss_lst, color = "red")
    # plt.savefig(f"/projects/kwessel4/dpc/output/plot/{model_option}_train_loss_{today}.png")
    # plt.close()

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    print(f'Time required for training {time.time()-tic}')


def train(data_loader, segment_model, optimizer):
    losses = AverageMeter()
    segment_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        input_im = input['ts'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,L,F,H,W) = input_im.shape
        batch = 1

        #input_mask = input_mask.view(batch,H,W)

        # print(f"features shape: {features.shape}")
        # print(f"mask shape: {input_mask.shape}")

        mask_pred = segment_model(input_im)

        # print(f"mask pred shape: {mask_pred.shape}")
        if criterion_type == 'BCE':
            input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
            input_mask = input_mask.squeeze(0)
            input_mask = torch.permute(input_mask, (0,3,1,2))
        elif criterion_type == 'dice':
            input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
            #print('input mask shape after onehot: ',input_mask.shape)
            # input_mask = input_mask.squeeze(0)
            # print('input mask shape after squeeze: ',input_mask.shape)
            input_mask = torch.permute(input_mask, (0,3,1,2))
        else:
            input_mask = input_mask.view(batch, H, W)

        if criterion_type == 'dice' or criterion_type == 'BCE':
            input_mask = input_mask.to(cuda, dtype=torch.float32)
        else:
            input_mask = input_mask.to(cuda, dtype=torch.long)

        loss = criterion(mask_pred, input_mask)

        losses.update(loss.item(), B)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.local_avg


def val(data_loader, segment_model, epoch):
    losses = AverageMeter()
    segment_model.eval()
    global iteration

    with torch.no_grad():
        for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_im = input['ts'].to(cuda, dtype=torch.float32)
            input_mask = input['mask'].to(cuda, dtype=torch.long)

            (B,L,F,H,W) = input_im.shape
            batch = 1

            #input_mask = input_mask.view(batch,H,W)

            # print(f"features shape: {features.shape}")
            # print(f"mask shape: {input_mask.shape}")

            mask_pred = segment_model(input_im)

            # print(f"mask pred shape: {mask_pred.shape}")

            if criterion_type == 'BCE':
                input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
                input_mask = input_mask.squeeze(0)
                input_mask = torch.permute(input_mask, (0,3,1,2))
            elif criterion_type == 'dice':
                input_mask = torch.nn.functional.one_hot(input_mask, num_classes=2)
                #input_mask = input_mask.squeeze(0)
                input_mask = torch.permute(input_mask, (0,3,1,2))
            else:
                input_mask = input_mask.view(batch, H, W)


            if criterion_type == 'dice' or criterion_type == 'BCE':
                input_mask = input_mask.to(cuda, dtype=torch.float32)
            else:
                input_mask = input_mask.to(cuda, dtype=torch.long)

            loss = criterion(mask_pred, input_mask)

            losses.update(loss.item(), B)

    return losses.local_avg, input['ts'], input['mask'], mask_pred


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/train_benchmodel.py --model convlstm --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model convgru --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model 3d-unet --dataset Tappan01 --img_dim 16 --epochs 50 --ts_length 16
