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
from tqdm import tqdm
import argparse
import h5py
import logging
import cv2
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import rioxarray as rxr
import rasterio
import xarray as xr
from inference import inference
from tensorboardX import SummaryWriter
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
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
parser.add_argument('--img_dim', default=32, type=int)


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
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.min(image[i, :, :])) / \
                (np.max(image[i, :, :]) - np.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
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
    precision = report['cropland']['precision']
    recall = report['cropland']['recall']
    f1_score = report['cropland']['f1-score']
    return accuracy, precision, recall, f1_score


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ### hls data
    hls=False

    if not hls:
        ts_name = 'Tappan13'
        if ts_name == 'Tappan14' or ts_name == 'Tappan15':
            filename = "/home/geoint/tri/hls_ts_video/hls_data_1415.hdf5"
        else:
            filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"

        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            ts_arr = file[f'{str(ts_name)}_PEV_ts'][()]
            mask_arr = file[f'{str(ts_name)}_mask'][()]

    else:
        ts_name = 'PEV_2021'
        filename = "/home/geoint/tri/hls_ts_video/hls_data_all.hdf5"
        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            ts_arr = file['PEV_2021_ts'][()]
            mask_arr = file['PEV_2021_ts'][()]


        ts_arr = np.transpose(ts_arr, (0,3,1,2))
        mask_arr = ts_arr

        # ts_arr = rescale_image(ts_arr)

        if 'PFV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PFV.2021081T110731.v2.0.tif'

        if 'PEV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2022009T112441.v2.0.tif'

        ref_im = rxr.open_rasterio(ref_im_fl)

        print('reference image shape: ', ref_im.shape)

    # get RGB image
    # ts_arr = ts_arr[:,1:4,:,:]
    # ts_arr = ts_arr[:,::-1,:,:]

    seq_length = 5
    num_seq = 4
    input_size = 64
    total_ts_len = 10 # L

    padding_size = 0

    print(f'data dict {ts_name} ts shape: {ts_arr.shape}')
    print(f'data dict {ts_name} mask shape: {mask_arr.shape}')


    if ts_arr.shape[0] > 9:
        train_ts_set = ts_arr[:total_ts_len,1:-2,:,:]
    else:
        train_ts_set = ts_arr[:,1:-2,:,:]
    # train_ts_set = train_ts_set.reshape((1,T,C,H,W))
    
    
    # ignore the no-data edge of cut Tappan Square to HSL
    if not hls:
        if ts_name == 'Tappan02' or ts_name == 'Tappan04':
            train_ts_set = ts_arr[:total_ts_len,1:-2,1:-1,1:160]
        else:
            train_ts_set = ts_arr[:total_ts_len,1:-2,2:-2,2:-2]
        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_option = 'dpc-unet' # 'dpc-unet' and 'unet', convlstm

    if unet_option == 'unet':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        unet_segment = UNet_test(num_classes=2,segment=True,in_channels=10)

        ### 10 bands
        unetsegment_checkpoint = f'{str(model_dir)}unet_meanframe_ts01_1train_9band_epoch_155.pth'
        if torch.cuda.is_available():
            unet_segment = unet_segment.to(cuda)

        unet_segment.load_state_dict(torch.load(unetsegment_checkpoint)['state_dict'])

        xraster = train_ts_set[:10,:,:,:].mean(axis=0)

        # print('ts xraster min', np.min(xraster))
        # print('ts xraster max', np.max(xraster))

        if hls:
            temporary_tif = xr.where(xraster > -1000, xraster, 2000) # 2000 is the optimal value for the nodata
        else:
            temporary_tif = xr.where(xraster > -9000, xraster, 120)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=unet_segment,
            n_classes=2,
            overlap=0.5,
            batch_size=128,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            model_option=unet_option
        )
    
    elif unet_option == 'convlstm':
        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
        model = ConvLSTM_Seg(
            num_classes=2,
            input_size=(64,64),
            hidden_dim=160,
            input_dim=10,
            kernel_size=(3, 3)
            )
 
        ### 10 bands
        model_checkpoint = f'{str(model_dir)}convlstm_10band_epoch_188.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set
        # xraster = rescale_image(xraster[:10,:,1:-1,1:-1])
        # xraster = rescale_image(xraster)

        if hls:
            xraster = xr.where(xraster[:10,:,:,:] > -9000, xraster[:10,:,:,:], 1000) # 2000 is the optimal value for the nodata
        else:
            xraster = rescale_image(xraster[:10,:,1:-1,1:-1])
            # temporary_tif = xr.where(xraster > -9000, xraster, 120)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=xraster,
            model=model,
            n_classes=2,
            overlap=0.5,
            batch_size=16,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            model_option=unet_option
        )

    elif unet_option == 'dpc-unet':

        network = 'unet' # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'
        if network == 'unet':
            encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
        else:
            encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0217_10band_unetvae_hls_64_97_2.0649683759061257e-05.pth' # unet-vae

        model = DPC_RNN_UNet(sample_size=input_size,
                        device=device,
                        num_seq=4, 
                        seq_len=6, 
                        network=network,
                        pred_step=1,
                        model_weight=encoder_weight,
                        freeze=True)

        model_dir = "/home/geoint/tri/dpc/models/checkpoints/"

        ### 13 bands
        if network == 'unet':
            model_checkpoint = f'{str(model_dir)}dpc-unet_9band_epoch24.pth'
            # model_checkpoint = f'{str(model_dir)}dpc-unet-unet-encoder_10band_ts01_epoch26.pth'
        else:
            model_checkpoint = f'{str(model_dir)}dpc-unet_10band_ts01_epoch7.pth'

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        model = nn.DataParallel(model)

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
        
        # Transpose the image for channel last format
        # image = np.transpose(image, (1,2,0))

        # Remove no-data values to account for edge effects
        # temporary_tif = image.values

        xraster = rescale_image(train_ts_set[:10,:,:,:]) # previously works 02-15
        # xraster = train_ts_set[:10,:,:,:]

        # print('ts xraster min', np.min(xraster))
        
        if hls:
            temporary_tif = xr.where(xraster > -9000, xraster, 2000) # 2000 is the optimal value for the nodata
        else:
            temporary_tif = xr.where(xraster > -9000, xraster, 2000)

        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=2,
            overlap=0.5,
            batch_size=1,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            model_option=unet_option
        )

    if hls:
        ref_im = ref_im.transpose("y", "x", "band")

    if prediction.shape[0] > 1:
        prediction = np.argmax(prediction, axis=0)
    else:
        prediction = np.squeeze(
            np.where(prediction > 0.5, 1, 0).astype(np.int16)
        )

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
    plt.savefig(f"{str(data_dir)}{ts_name}-{unet_option}-new.png", dpi=300, bbox_inches='tight')

    plt.close()

    #save Tiff file output
    if hls:
        # Drop image band to allow for a merge of mask
        ref_im = ref_im.drop(
            dim="band",
            labels=ref_im.coords["band"].values[1:],
            drop=True
        )

        prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='otcb',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('otcb')
        prediction.attrs['model_name'] = (unet_option)
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
            f'{data_dir}{ts_name}-{unet_option}.tiff',
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


def predict_segment(data_loader, segment_model, optimizer):

    segment_model.eval()
    global iteration

    for idx, input in enumerate(data_loader):

        features = input['x'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,F,H,W) = features.shape
        batch = 1

        # features = features.view(B*N,SL,F,H,W)
        features = features.mean(dim=0)
        features = features.view(batch,F,H,W)
        input_mask = input_mask.view(batch,H,W)

        mask_pred, _ = segment_model(features)

    return mask_pred


if __name__ == '__main__':
    main()

    # python main.py --gpu 0 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --epochs 100
