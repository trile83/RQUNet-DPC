import disstl.models as models
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
from sklearn import tree
import pickle
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='convgru', type=str, help='convlstm, convgru')
parser.add_argument('--dataset', default='Tappan01', type=str, help='Tappan01, Tappan05')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=3e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_chips', default=100, type=int)
parser.add_argument('--num_val', default=20, type=int)
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

def get_train_set(args, list_ts):
    
    filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_inc_cloud.hdf5"

    train_ts_set = []
    train_mask_set = []

    temp_ts_set = []
    temp_mask_set = []
    
    for ts_name in list_ts:
    
        print(ts_name)
        with h5py.File(filename, "r") as file:
            if ts_name == 'Tappan06' or ts_name == 'Tappan07':
                ts_arr = file[f'{str(ts_name)}_PFV_ts'][()]
                mask_arr = file[f'{str(ts_name)}_mask'][()]

                # ref_im_fl = f"/home/geoint/tri/hls/{str(ts_name)}_HLS.S30.T28PFV.2021179T112119.v2.0.tif"
                # ref_im = rxr.open_rasterio(ref_im_fl)
            else:
                ts_arr = file[f'{str(ts_name)}_PEV_ts'][()]
                mask_arr = file[f'{str(ts_name)}_mask'][()]

        input_size = args.img_dim
        total_ts_len = args.ts_length # L
        padding_size = args.pad_size

        ts_arr = np.concatenate((ts_arr[:args.ts_length,1:-4,:,:], ts_arr[:args.ts_length,-2:,:,:]), axis=1)

        for i in range(args.num_chips+args.num_val):
            ts, mask = chipper(ts_arr[:,:,:,:], mask_arr, input_size=args.img_dim)
            ts = np.squeeze(ts)

            if args.rescale == 'per-ts':
                ts = rescale_image(ts, args.rescale)
            else:
                for frame in range(ts.shape[0]):
                    if args.normalization is not None:
                        ts[frame] = normalize_image(ts[frame], args.normalization)

                    if args.rescale is not None:
                        ts[frame] = rescale_image(ts[frame],args.rescale)

                    if args.standardization is not None:
                        ts[frame] = standardize_image(ts[frame],args.standardization)

            mask = np.squeeze(mask)

            # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

            temp_ts_set.append(ts)
            temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_mask_set = np.stack(temp_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    return train_ts_set, train_mask_set, args.num_val*len(list_ts)


def train_decision_tree(train_ts_set, train_ts_mask):

    print(train_ts_set.shape)
    print(train_ts_mask.shape)
    ts_mean = train_ts_set.mean(axis=1)
    ts_mean = np.squeeze(ts_mean)
    ts_mean = np.transpose(ts_mean, (0,2,3,1))
    print(ts_mean.shape)

    X = ts_mean.reshape((ts_mean.shape[0]*ts_mean.shape[1]*ts_mean.shape[2], ts_mean.shape[3]))
    Y = train_ts_mask.reshape((train_ts_mask.shape[0]*train_ts_mask.shape[1]*train_ts_mask.shape[2]))

    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)

    ## save model
    filename = '/home/geoint/tri/dpc/models/checkpoints/decisiontree_model_0505.sav'
    if not os.path.isfile(filename):
        ## decision tree model
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X, Y)
        pickle.dump(clf, open(filename, 'wb'))
    else:
        clf = pickle.load(open(filename, 'rb'))

    print(clf.feature_importances_)

    # tree.plot_tree(clf)

    return clf


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

    # list_ts = ['Tappan01','Tappan07']
    list_ts = ['Tappan01']
    ts_name=args.dataset

    input_size = args.img_dim

    train_ts_set, train_mask_set, num_val = get_train_set(args, list_ts)

    model = train_decision_tree(train_ts_set, train_mask_set)


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/train_benchmodel.py --model convlstm --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model convgru --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model 3d-unet --dataset Tappan01 --img_dim 16 --epochs 50 --ts_length 16