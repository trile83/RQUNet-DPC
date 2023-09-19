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
from dpc.model_3d_unet_stride import *
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy, calc_accuracy
from tqdm import tqdm
import argparse
import h5py
import logging
import torchvision.utils as vutils
import cv2
import rioxarray as rxr
import time
from tensorboardX import SummaryWriter

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='unet', type=str, help='encoder for the DPC')
parser.add_argument('--model', default='dpc-unet', type=str, help='convlstm, dpc-unet, unet')
parser.add_argument('--dataset', default='Tappan01_WV02_20181217', type=str, help='PEV_2021, PFV_2021, or Tappan01, Tappan05')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=3e-4, type=float, help='weight decay')
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
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--num_chips', default=100, type=int)
parser.add_argument('--num_val', default=10, type=int)
parser.add_argument('--hidden_dim', default=200, type=int)
parser.add_argument('--standardization', default='local', type=str)
parser.add_argument('--normalization', default=0.0001, type=float)
parser.add_argument('--rescale', default='per-ts', type=str)
parser.add_argument('--segment_model', default='unet', type=str)

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
            if not array.any():
                print(f"i {i}")
            all_arr[j,i-num_seq] = array

        # for i in range(L1-num_seq): # same results
        #     array = windows[j,i:i+num_seq,:,:,:,:] # N, SL, C, H, W
        #     all_arr[j,i] = array

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

    # for j in range(I):
    #     all_arr[j,L2+num_seq-1,:,:,:,:] = chunks[j,L2-1,-1,:,:,:,:]

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

def get_composite(ts_arr):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[0] // 10
    # print(step)

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[0], step):
        out_lst.append(np.median(ts_arr[i:i+step], axis=0))

    out_array = np.stack(out_lst, axis=0)
    del ts_arr

    # print(out_array.shape)

    return out_array

def get_train_set(args, list_ts, tile='PEV'):
    
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"
    # filename = "/home/geoint/tri/hls_ts_video/hls_data_inc_cloud.hdf5"

    ### UPDATE 09/01 - new datacube with small TS time series
    filename = "/home/geoint/tri/hls_datacube/hls-ecas-PEV-0901.hdf5"

    train_ts_set = []
    train_mask_set = []

    temp_ts_set = []
    temp_mask_set = []
    
    for ts_name in list_ts:
    
        print(ts_name)
        # with h5py.File(filename, "r") as file:
        #     if ts_name == 'Tappan06' or ts_name == 'Tappan07':
        #         ts_arr = file[f'{str(ts_name)}_PFV_ts'][()]
        #         mask_arr = file[f'{str(ts_name)}_mask'][()]

        #         # ref_im_fl = f"/home/geoint/tri/hls/{str(ts_name)}_HLS.S30.T28PFV.2021179T112119.v2.0.tif"
        #         # ref_im = rxr.open_rasterio(ref_im_fl)
        #     else:
        #         ts_arr = file[f'{str(ts_name)}_PEV_ts'][()]
        #         mask_arr = file[f'{str(ts_name)}_mask'][()]

        #### UPDATEs 09/01
        with h5py.File(filename, "r") as file:
            ts_arr = file[f'{str(ts_name)}_{str(tile)}_ts'][()]
            mask_arr = file[f'{str(ts_name)}_{str(tile)}_mask'][()]

        print("out ts arr shape: ", ts_arr.shape)

        ts_arr = get_composite(ts_arr)

        mask_arr[mask_arr != 2] = 0
        mask_arr[mask_arr == 2] = 1

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
    ### hls data
    ts_name=args.dataset
    tile='PEV'
    list_ts = ['Tappan01_WV02_20181217']

    input_size = args.img_dim

    train_ts_set, train_mask_set, num_val = get_train_set(args, list_ts, tile)

    im_set = get_seq(train_ts_set, args.seq_len) #(I,L1,SL,C,H,W)
    print(f'window sequence shape: {im_set.shape}')

    new_set = get_chunks(im_set, args.num_seq)

    print(f'chunk sequence shape: {new_set.shape}') # (I,L2,N,SL,C,H,W); L2 = L-seq_len-num_seq

    (I,L2,N,SL,C,H,W) = new_set.shape

    train_set = tsDataset(new_set[:-num_val], train_mask_set[:-num_val], train_ts_set[:-num_val])
    val_set = tsDataset(new_set[-num_val:], train_mask_set[-num_val:], train_ts_set[-num_val:])

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
        model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
    elif network == 'unet-vae':
        model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_0217_10band_unetvae_hls_64_97_2.0649683759061257e-05.pth' # unet-vae

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
                    segment_model=args.segment_model)


    if torch.cuda.is_available():
        model = model.to(cuda)

    global criterion; 
    criterion_type = "crossentropy"
    if criterion_type == "crossentropy":
        # criterion = models.losses.MultiTemporalCrossEntropy()
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == "focal_tverski":
        criterion = models.losses.FocalTversky()
    elif criterion_type == "dice":
        criterion = models.losses.MultiTemporalDiceLoss()

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
    global img_path; img_path, model_path = set_path(args)
    model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
    global writer_train
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    if network == 'unet-vae' or network == 'rqunet-vae-encoder' or network == 'unet':

        ### main loop ###
        # train_loss_lst = []
        # val_loss_lst = []
        train_loss_out = []
        val_loss_out = []

        train_losses = AverageMeter()
        val_losses = AverageMeter()

        print(f"length of dpc input training set {len(train_dl)}")
        min_loss = np.inf

        early_stopper = EarlyStopper(patience=10, min_delta=0.2)

        for epoch in range(args.start_epoch, args.epochs):
            # print('Epoch: ', epoch)
            for idx, input in enumerate(train_dl):

                input_ts = input['ts']
                input_mask = input['mask']
                ori_ts = input['ori']

                # print('original ts shape: ', ori_ts.shape)

                (I,L2,N,SL,C,H,W) = input_ts.shape

                # input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

                train_set = satDataset(input_ts, input_mask, ori_ts)

                loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
                train_sat_dl = DataLoader(train_set, **loader_args_sat)

                output, train_loss = train_dpc(train_sat_dl, model, optimizer, network)
                train_losses.update(train_loss.item(), I*L2)

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
                val_losses.update(val_loss.item(), I_val*L2)

            # save check_point
            is_best = val_losses.local_avg < min_loss

            if is_best:
                min_loss = val_losses.local_avg

                 # visualize predictions
                index_array = torch.argmax(output_val, dim=1)
                z = ori_ts_val.numpy()
                y = input_mask_val

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
                plt.savefig(f"/home/geoint/tri/dpc/output/train-{str(epoch)}-{str(idx)}-dpc-unet-0901-pred.png")
                plt.close()


                # save dpc weights
                save_checkpoint({'epoch': epoch+1,
                                'net': args.net,
                                'state_dict': model.state_dict(),
                                'min_loss': min_loss,
                                'optimizer': optimizer.state_dict()}, 
                                is_best, filename=os.path.join(model_dir, f'dpc-{network}-encoder-0901-composite-{args.segment_model}_{args.hidden_dim}_10band_ts01_epoch%s.pth' % str(epoch+1)), keep_all=False)

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
        plt.savefig("/home/geoint/tri/dpc_test_out/train_loss_0901.png")
        plt.close()

        print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
        print(f'Time required for training {time.time()-tic}')

    else:
        for idx, input in enumerate(train_dl):

            input_ts = input['ts']
            input_mask = input['mask']

            (I,L2,N,SL,C,H,W) = input_ts.shape

            input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

            train_set = satDataset(input_ts, input_mask)
            # val_set = satDataset(input_ts[-1], input_mask[-1])

            loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=False, drop_last=False, shuffle=False)
            train_sat_dl = DataLoader(train_set, **loader_args_sat)
            val_sat_dl = DataLoader(train_set, **loader_args_sat)
        
            ### main loop ###
            for epoch in range(args.start_epoch, args.epochs):
                train_loss, train_acc, train_accuracy_list = train(train_sat_dl, model, optimizer, epoch)
                val_loss, val_acc, val_accuracy_list = validate(val_sat_dl, model, epoch)

                # save curve
                writer_train.add_scalar('global/loss', train_loss, epoch)
                writer_train.add_scalar('global/accuracy', train_acc, epoch)
                writer_val.add_scalar('global/loss', val_loss, epoch)
                writer_val.add_scalar('global/accuracy', val_acc, epoch)
                writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
                writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
                writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
                writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
                writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
                writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

                # save check_point
                is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
                save_checkpoint({'epoch': epoch+1,
                                'net': args.net,
                                'state_dict': model.state_dict(),
                                'best_acc': best_acc,
                                'optimizer': optimizer.state_dict(),
                                'iteration': iteration}, 
                                is_best, filename=os.path.join(model_path, 'hidden180-0506_new_epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

            print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train_dpc(data_loader, dpc_model, optimizer, network):
    losses = AverageMeter()
    dpc_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        tic = time.time()
        input_seq = input["x"]
        input_mask = input["mask"]

        (B,L2,N,SL,C,H,W) = input_seq.shape

        input_seq = input_seq.to(cuda, dtype=torch.float32)
        input_mask = input_mask.to(cuda, dtype=torch.long)
        
        input_mask = input_mask.view(1, H, W)
        B = input_seq.size(0)

        # if network == 'unet-vae' or network == 'rqunet-vae-encoder' or network == 'unet':
        output, _ = dpc_model(input_seq)

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

            input_seq = input_seq.to(cuda, dtype=torch.float32)
            input_mask = input_mask.to(cuda, dtype=torch.long)

            (B,L2,N,SL,C,H,W) = input_seq.shape

            # print(f'input mask shape : {input_mask.shape}') # 1 x batch x input_size x input_size
            
            input_mask = input_mask.view(input_mask.shape[1], H, W)
            B = input_seq.size(0)

            # if network == 'unet-vae' or network == 'rqunet-vae-encoder':
            output, _ = dpc_model(input_seq)

            loss = criterion(output, input_mask)
            losses.update(loss.item(), B)

    return output, losses.local_avg


def train(data_loader, model, optimizer, epoch, segment=True):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    if not segment:

        for idx, input_seq in enumerate(data_loader):
            tic = time.time()
            input_seq = input_seq['x'].to(cuda, dtype=torch.float32)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            # visualize
            if (iteration == 0) or (iteration == args.print_freq):
                if B > 2: input_seq = input_seq[0:2,:]
                writer_train.add_image('input_seq',
                                    de_normalize(vutils.make_grid(
                                        input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                        nrow=args.num_seq*args.seq_len)),
                                    iteration)
            del input_seq
            
            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
            # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.view(B*NP*SQ, B2*NS*SQ).to(cuda)
            target_flattened = target_flattened.to(int).argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            # top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            # accuracy_list[0].update(top1.item(), B)
            # accuracy_list[1].update(top3.item(), B)
            # accuracy_list[2].update(top5.item(), B)

            accu = calc_accuracy(score_flattened, target_flattened)

            losses.update(loss.item(), B)
            # accuracy.update(top1.item(), B)
            accuracy.update(accu.item(), B)

            del score_

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss

            if idx % args.print_freq == 0:
                print(f'Epoch: [{epoch}][{idx}/{len(data_loader)}]\t'
                    f'Loss {losses.val:.6f} ({losses.local_avg:.4f})\t'
                    #   'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                    #    epoch, idx, len(data_loader), top1, top3, top5, time.time()-tic, loss=losses)
                    f'Acc: {accu:.4f} T:{(time.time()-tic):.2f}\t')
                writer_train.add_scalar('local/loss', losses.val, iteration)
                writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

                iteration += 1

        return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]
    
    else:
        for idx, input in enumerate(data_loader):
            tic = time.time()
            input_seq = input['x'].to(cuda, dtype=torch.float32)
            target = input['mask'].to(cuda, dtype=torch.long)
            B = input_seq.size(0)
            output, _ = model(input_seq)

            print('output shape: ', output.shape)

            # visualize
            if (iteration == 0) or (iteration == args.print_freq):
                if B > 2: input_seq = input_seq[0:2,:]
                writer_train.add_image('input_seq', 
                                    de_normalize(vutils.make_grid(
                                        input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                        nrow=args.num_seq*args.seq_len)), 
                                    iteration)
            del input_seq

            [_, N, D] = output.size()
            output = output.view(B*N, D)
            target = target.repeat(1, N).view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            del target 

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.local_avg:.4f}) T:{3:.2f}\t'.format(
                    epoch, idx, len(data_loader), time.time()-tic,
                    loss=losses, acc=accuracy))

                total_weight = 0.0
                decay_weight = 0.0
                for m in model.parameters():
                    if m.requires_grad: decay_weight += m.norm(2).data
                    total_weight += m.norm(2).data
                print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))
                
                writer_train.add_scalar('local/loss', losses.val, iteration)
                writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

                iteration += 1

        return losses.local_avg, accuracy.local_avg


def validate(data_loader, model, epoch, segment=True):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    if not segment:
        with torch.no_grad():
            for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):
                input_seq = input['x'].to(cuda, dtype=torch.float32)
                B = input_seq.size(0)
                [score_, mask_] = model(input_seq)
                del input_seq

                if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

                # [B, P, SQ, B, N, SQ]
                score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target_.view(B*NP*SQ, B2*NS*SQ).to(cuda)
                target_flattened = target_flattened.to(int).argmax(dim=1)

                loss = criterion(score_flattened, target_flattened)
                # top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

                accu = calc_accuracy(score_flattened, target_flattened)

                losses.update(loss.item(), B)
                # accuracy.update(top1.item(), B)
                accuracy.update(accu.item(), B)

                # accuracy_list[0].update(top1.item(), B)
                # accuracy_list[1].update(top3.item(), B)
                # accuracy_list[2].update(top5.item(), B)

        print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
            #   'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
            #    epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
            'Acc: {2:.4f} \t'.format(
            epoch, args.epochs, accu, loss=losses))
        return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]

    else:
        with torch.no_grad():
            for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):
                input_seq = input['x'].to(cuda, dtype=torch.float32)
                target = input['mask'].to(cuda, dtype=torch.long)
                B = input_seq.size(0)
                output, _ = model(input_seq)

                [_, N, D] = output.size()
                output = output.view(B*N, D)
                target = target.repeat(1, N).view(-1)

                loss = criterion(output, target)
                acc = calc_accuracy(output, target)

                losses.update(loss.item(), B)
                accuracy.update(acc.item(), B)
                    
        print('Loss {loss.avg:.4f}\t'
            'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
        return losses.avg, accuracy.avg

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
    # python models/train_dpc_seg.py --net unet-vae --dataset Tappan01 --epochs 100
