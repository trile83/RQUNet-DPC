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
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy, calc_accuracy
from sklearn.metrics import jaccard_score, balanced_accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import argparse
import h5py
import csv
import torchvision.utils as vutils
import logging
import rioxarray as rxr
from tensorboardX import SummaryWriter

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
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
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
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i] - np.min(image[:, :, i])) / \
                (np.max(image[:, :, i]) - np.min(image[:, :, i]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
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


def get_accuracy(y_pred, y_true):

    target_names = ['non-crop','cropland']

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # get overall weighted accuracy
    accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    iou = jaccard_score(y_pred, y_true, average="micro")
    precision = report['cropland']['precision']
    recall = report['cropland']['recall']
    f1_score = report['cropland']['f1-score']
    return accuracy, precision, recall, f1_score, iou


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.
    ### hls data
    # prepare data
    ### hls data
    hls=False
    if not hls:
        filename = "/home/geoint/tri/hls_ts_video/hls_data.hdf5"

        ts_name = 'Tappan05'
        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            ts_arr = file['Tappan05_PEV_ts'][()]
            mask_arr = file['Tappan05_mask'][()]
    else:
        filename = "/home/geoint/tri/hls_ts_video/hls_data_all.hdf5"

        ts_name = 'PEV_2021'
        with h5py.File(filename, "r") as file:
            print("Keys: %s" % file.keys())
            ts_arr = file['PEV_2021_ts'][()]

        ts_arr = np.transpose(ts_arr, (0,3,1,2))
        mask_arr = ts_arr[0,0,:,:]

        # print(ts_arr.shape)

    seq_length = 6
    num_seq = 4
    input_size = 64 ## 64
    total_ts_len = 10 # L

    padding_size = 8
    
    print(f'data dict tappan01 ts shape: {ts_arr.shape}')
    print(f'data dict tappan01 mask shape: {mask_arr.shape}')

    train_ts_set = []
    train_mask_set = []

    ### get RGB image
    # ts_arr = ts_arr[:,1:4,:,:]
    # ts_arr = ts_arr[:,::-1,:,:]

    ## get different chips in the Tappan Square for multiple time series
    h_list =[10,20,30,40,50,70,80,90,100,110,150]
    w_list =[15,25,35,45,55,75,85,95,105,115,160]

    num_val = 1
    num_chips = 11 # I

    temp_ts_set = []
    temp_mask_set = []
    # for i in range(len(h_list)):
    #     ts, mask = specific_chipper(ts_arr[:,1:-2,:,:], mask_arr,h_list[i], w_list[i], input_size=input_size)

    for i in range(num_chips):
        ts, mask = chipper(ts_arr[:total_ts_len,1:-2,:,:], mask_arr, input_size=input_size)

        ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
        if not hls:
            if np.any(ts == -1): # avoid no data region in image
                continue

            # for j in range(ts.shape[0]):
            #     ts[j] = rescale_image(ts[j])

            ts = rescale_image(ts)
        else:
            if np.any(ts == -9999): # avoid no data region in image
                continue

            ts = rescale_image(ts)

        mask = mask.reshape((mask.shape[1],mask.shape[2]))

        # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    # train_ts_set = train_ts_set[:,:total_ts_len] # get the first 100 in the time series
    mask_set = np.stack(temp_mask_set, axis=0)
    train_mask_set = mask_set[:]


    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    im_set = get_seq(train_ts_set, seq_length) #(I,L1,SL,C,H,W)
    print(f'window sequence shape: {im_set.shape}')

    new_set = get_chunks(im_set, num_seq)

    print(f'chunk sequence shape: {new_set.shape}') # (I,L2,N,SL,C,H,W); L2 = L-seq_len-num_seq

    (I,L2,N,SL,C,H,W) = new_set.shape

    train_set = tsDataset(new_set[:-num_val], train_mask_set[:-num_val], train_ts_set[:-num_val])
    val_set = tsDataset(new_set[-num_val:], train_mask_set[-num_val:], train_ts_set[-num_val:])

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    val_loader_args = dict(batch_size=num_val, num_workers=4, pin_memory=True, drop_last=False, shuffle=False)
    train_dl = DataLoader(train_set, **loader_args)
    val_dl = DataLoader(val_set, **val_loader_args)
        
    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = 'unet' # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'

    # model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_1028_3band_unetvae_hls_65_2.7782891265815123e-07.pth'
    encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0129_10band_unetvae_hls_98_2.7315708488893155e-06.pth'

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
    model_checkpoint = f'{str(model_dir)}dpc-unet_9band_ts01-train1_epoch31.pth'

    model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

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
    optimizer = optim.Adam(params, lr=0.001, weight_decay=0.0001)
    args.old_lr = None

    ### load data ###

    print("Start prediction process!")

    ## visualize

    data_dir = '/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/'

    batch_size = 1

    with open(f'{data_dir}{ts_name}_dpc-unet-seg_stat_results.csv','w') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['id','frame','accuracy','precision','recall','f1-score','mIoU'])

        ### main loop ##
        print(f"length of dpc input training set {len(train_dl)}")

        for idx, input in enumerate(train_dl):

            input_ts = input['ts']
            input_mask = input['mask']
            ori_ts = input['ori']

            # print('original ts shape: ', ori_ts.shape)

            (I,L2,N,SL,C,H,W) = input_ts.shape

            input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

            train_set = satDataset(input_ts, input_mask, ori_ts)

            loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
            train_sat_dl = DataLoader(train_set, **loader_args_sat)

            output, _ = predict_dpc(train_sat_dl, model, network)

            frame = 5

            # visualize predictions
            index_array = torch.argmax(output, dim=1)
            z = ori_ts.numpy()
            y = input_mask

            # accuracy, precision, recall, f1_score, iou = get_accuracy(index_array.cpu().numpy(), y)
            # writer.writerow([idx, frame, accuracy, precision, recall, f1_score, iou])

            plt.figure(figsize=(20,20))
            plt.subplot(1,3,1)
            plt.title("Image")
            image = np.transpose(z[0,frame,1:4,:,:], (1,2,0))
            image = rescale_image(image)
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
            plt.savefig(f"/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/predict-{ts_name}-{str(idx)}-dpc-unet-pred.png")
            plt.close()

def predict_dpc(data_loader, dpc_model, network):
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

            (B,N,SL,C,H,W) = input_seq.shape
            
            input_mask = input_mask.view(1, H, W)
            B = input_seq.size(0)

            if network == 'unet-vae' or network == 'rqunet-vae-encoder' or network =='unet':
                output, _ = dpc_model(input_seq)

            # print('dpc output type: ',type(output))

            # loss = criterion(output, input_mask)
            # losses.update(loss.item(), B)

    return output, losses.local_avg


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
