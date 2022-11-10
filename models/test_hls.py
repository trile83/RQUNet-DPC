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
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
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
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ### hls data
    filename = "/home/geoint/tri/hls_ts_video/hls_data.hdf5"
    with h5py.File(filename, "r") as f:
        # print("Keys: %s" % f.keys())
        ts_arr = f['Tappan06_ts'][()]
        mask_arr = f['Tappan06_mask'][()]

    ts_name = 'TS06'

    # get RGB image
    # ts_arr = ts_arr[:,1:4,:,:]
    # ts_arr = ts_arr[:,::-1,:,:]

    seq_length = 5
    num_seq = 4
    input_size = 64
    total_ts_len = 10 # L

    print(f'data dict tappan01 ts shape: {ts_arr.shape}')
    print(f'data dict tappan01 mask shape: {mask_arr.shape}')

    train_ts_set = []
    train_mask_set = []

    ## get different chips in the Tappan Square for multiple time series
    iteration = 30

    temp_ts_set = []
    temp_mask_set = []
    for i in range(iteration):
        ts, mask = chipper(ts_arr, mask_arr, input_size=input_size)
        ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
        mask = mask.reshape((mask.shape[1],mask.shape[2]))

        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_ts_set = train_ts_set[:,:total_ts_len] # get the first 100 in the time series
    train_mask_set = np.stack(temp_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    im_set = get_seq(train_ts_set, seq_length)
    # mask_set = get_seq(train_mask_set, seq_length)

    print(f'window sequence shape: {im_set.shape}')

    new_set = get_chunks(im_set, num_seq)

    print(f'chunk sequence shape: {new_set.shape}')

    (A,L,N,SL,C,H,W) = new_set.shape

    test_set = tsDataset(new_set, train_mask_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    train_dl = DataLoader(test_set, **loader_args)
    val_dl = DataLoader(test_set, **loader_args)
        
    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # encoder_weights = '/home/geoint/tri/dpc/models/checkpoints/recon_1028_3band_unetvae_hls_65_2.7782891265815123e-07.pth'
    encoder_weights = '/home/geoint/tri/dpc/models/checkpoints/recon_0927_13band_unetvae_hls_8_0.00016154855853173791.pth'

    model = DPC_RNN_UNet(sample_size=32,
                    device=device,
                    num_seq=4, 
                    seq_len=6, 
                    # network="rqunet-vae-encoder",
                    network ="unet-vae",
                    pred_step=1,
                    model_weight=encoder_weights,
                    freeze=True)

    model_dir = "/home/geoint/tri/dpc/models/checkpoints/"

    unet_segment = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)
    # unet_segment = UNet3D(in_channel=5, n_classes=2)

    ### RGB
    # dpc_checkpoint = f'{str(model_dir)}rq_dpc_epoch222.pth'
    # unetsegment_checkpoint = f'{str(model_dir)}unetsegment_epoch222.pth'

    ### 13 bands
    dpc_checkpoint = f'{str(model_dir)}dpc_13band_epoch158.pth'
    unetsegment_checkpoint = f'{str(model_dir)}unetsegment_13band_epoch_158.pth'

    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)
        unet_segment = unet_segment.to(cuda)

    model.load_state_dict(torch.load(dpc_checkpoint)['state_dict'])
    unet_segment.load_state_dict(torch.load(unetsegment_checkpoint)['state_dict'])

    global criterion; 
    criterion_type = "crossentropy"
    if criterion_type == "crossentropy":
        # criterion = models.losses.MultiTemporalCrossEntropy()
        criterion = criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == "focal_tverski":
        criterion = models.losses.FocalTversky()
    elif criterion_type == "dice":
        criterion = models.losses.MultiTemporalDiceLoss()

    ### optimizer ###
    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.000)

    segment_optimizer = optim.Adam(unet_segment.parameters(), lr=0.0001, weight_decay=0.000)
    args.old_lr = None

    best_acc = 0
    min_loss = np.inf

    ### restart training ###
    ### load data ###

    print("Start prediction!")

    # setup tools
    global de_normalize; de_normalize = denorm()
    
    ### main loop ###

    print(f"length of dpc input training set {len(train_dl)}")

    all_feature_arr = []
    all_mask_arr = []

    for idx, input in enumerate(train_dl):

        input_ts = input['ts']
        input_mask = input['mask']

        # print(f"input ts shape: {input_ts.shape}")

        (I,L2,N,SL,C,H,W) = input_ts.shape

        input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

        # print(f"input ts shape {input_ts.shape}")
        # print(f"input mask shape {input_mask.shape}")

        test_set = satDataset(input_ts, input_mask)

        loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        train_sat_dl = DataLoader(test_set, **loader_args_sat)

        feature_arr = predict_dpc(train_sat_dl, model)
        feature_arr = rearrange(feature_arr, "(b l2) n sl c h w -> b l2 n sl c h w", l2 = L2)

        # print(f"feature arr shape: {feature_arr.shape}")

        all_feature_arr.append(feature_arr)

    all_feature_arr = torch.cat(all_feature_arr, dim=0)

    print(f"all feature arr shape before reverse: {all_feature_arr.shape}")

    all_feature_arr = reverse_chunks(all_feature_arr, num_seq)
    print(f'reverse chunk context vector shape: {all_feature_arr.shape}')
    all_feature_arr = reverse_seq(all_feature_arr, seq_length)
    print(f'reversed windown context vector shape: {all_feature_arr.shape}')

    print(f"all feature set shape after reverse: {all_feature_arr.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    print("Finished with DPC phase")

    train_seg_set = segmentDataset(all_feature_arr, train_mask_set, train_ts_set)
    loader_args_1 = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_segment_dl = DataLoader(train_seg_set, **loader_args_1)

    # print(f"Length of segmentation input set {len(train_segment_dl)}")
    print("Start segmentation!")

    ## visualize

    data_dir = '/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/'

    batch_size = 1

    with open(f'{data_dir}{ts_name}_stat_results.csv','w') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['id','accuracy','precision','recall','f1-score'])

        for idx, batch in enumerate(train_segment_dl):
            x = batch['x'].to(cuda, dtype=torch.float32)
            y = batch['mask'].numpy()
            z = batch['ts'].numpy()

            z_mean = z.mean(axis=0)

            # print(f"im shape: {z.shape}")
            # print(f"y shape: {y.shape}")

            (B,L,F,H,W) = x.shape

            x_ori = x

            x = x.view(B*L,F,H,W)
            batch_size = 1

            x_mean = x.mean(dim=0)
            x_mean = x_mean.view(batch_size,F,H,W)

            # predict mean of time series
            output = unet_segment(x_mean)
            # print(f"output shape: {output[0].shape}")
            y_pred = output[0]

            index_array = torch.argmax(y_pred, dim=1)
            # print(f"index array shape: {index_array.shape}")

            accuracy, precision, recall, f1_score = get_accuracy(index_array.cpu().numpy(), y)

            writer.writerow([idx, accuracy, precision, recall, f1_score])

            plt.figure(figsize=(20,20))
            plt.subplot(1,3,1)
            plt.title("Image")
            image = np.transpose(z[0,5,1:4,:,:], (1,2,0))
            # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
            plt.imshow(rescale_truncate(image))
            plt.subplot(1,3,2)
            plt.title("Segmentation Label")
            image = np.transpose(batch['mask'].numpy()[0,:,:], (0,1))
            plt.imshow(image)
        
            plt.subplot(1,3,3)
            plt.title(f"Segmentation w accuracy {accuracy}")
            image = np.transpose(index_array[0,:,:].cpu().numpy(), (0,1))
            plt.imshow(image)
            plt.savefig(f"{str(data_dir)}{str(idx)}_ts_{idx}_{i}_old.png")

            plt.close()


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
