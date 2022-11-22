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
import graphlearning as gl
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
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
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

        # for i in range(L-seq_length): # same results
        #     array = sequence[j,i:i+seq_length,:,:,:] # SL, C, H, W
        #     all_arr[j,i] = array

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


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ### hls data
    filename = "/home/geoint/tri/hls_ts_video/old/hls_data.hdf5"
    with h5py.File(filename, "r") as f:
        # print("Keys: %s" % f.keys())
        ts_arr = f['Tappan01_ts'][()]
        mask_arr = f['Tappan01_mask'][()]

    ts_name = 'TS01'

    seq_length = 6
    num_seq = 4
    input_size = 64 ## 64
    total_ts_len = 10 # L

    padding_size = 8
    
    # print(f'data dict tappan01 ts shape: {ts_arr.shape}')
    # print(f'data dict tappan01 mask shape: {mask_arr.shape}')

    train_ts_set = []
    train_mask_set = []

    ### get RGB image
    # ts_arr = ts_arr[:,1:4,:,:]
    # ts_arr = ts_arr[:,::-1,:,:]

    ## get different chips in the Tappan Square for multiple time series
    iteration = 10 # I

    temp_ts_set = []
    temp_mask_set = []
    for i in range(iteration):
        ts, mask = chipper(ts_arr, mask_arr, input_size=input_size)
        ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
        mask = mask.reshape((mask.shape[1],mask.shape[2]))

        # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_ts_set = train_ts_set[:,:total_ts_len] # get the first 100 in the time series
    train_mask_set = np.stack(temp_mask_set, axis=0)

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    im_set = get_seq(train_ts_set, seq_length) #(I,L1,SL,C,H,W)
    print(f'window sequence shape: {im_set.shape}')

    new_set = get_chunks(im_set, num_seq)

    print(f'chunk sequence shape: {new_set.shape}') # (I,L2,N,SL,C,H,W); L2 = L-seq_len-num_seq

    (I,L2,N,SL,C,H,W) = new_set.shape

    # rev_chunk_set = reverse_chunks(new_set, num_seq)
    # print(f'reverse chunk shape: {rev_chunk_set.shape}')
    # rev_window_set = reverse_seq(rev_chunk_set, seq_length)
    # print(f'reverse window shape: {rev_window_set.shape}')
    # print("are the reconstructed chunk set and original set equal?",np.array_equal(rev_chunk_set,im_set))
    # print("are the reconstructed ts set and original set equal?",np.array_equal(rev_window_set,train_ts_set))

    test_set = tsDataset(new_set, train_mask_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    train_dl = DataLoader(test_set, **loader_args)
    val_dl = DataLoader(test_set, **loader_args)
        
    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_1028_3band_unetvae_hls_65_2.7782891265815123e-07.pth'
    model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/recon_0927_13band_unetvae_hls_8_0.00016154855853173791.pth'

    model = DPC_RNN_UNet(sample_size=64,
                    device=device,
                    num_seq=4, 
                    seq_len=6, 
                    network='unet-vae',
                    # network="rqunet-vae-encoder",
                    pred_step=1,
                    model_weight=model_checkpoint,
                    freeze=True)

    unet_segment = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)

    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)
        unet_segment = unet_segment.to(cuda)

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
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0)
    args.old_lr = None

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else: 
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###

    print("Start training process!")

    # setup tools
    
    ### main loop ###
    print(f"length of dpc input training set {len(train_dl)}")

    all_feature_arr = []

    for idx, input in enumerate(train_dl):

        input_ts = input['ts']
        input_mask = input['mask']

        (I,L2,N,SL,C,H,W) = input_ts.shape

        input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

        # print(f"input ts shape {input_ts.shape}")
        # print(f"input mask shape {input_mask.shape}")

        test_set = satDataset(input_ts, input_mask)

        loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        train_sat_dl = DataLoader(test_set, **loader_args_sat)

        for epoch in range(len(train_sat_dl)):

            feature_arr = train_dpc(train_sat_dl, model, unet_segment, optimizer, 1, num_seq, seq_length)
            feature_arr = rearrange(feature_arr, "(b l2) n sl c h w -> b l2 n sl c h w", l2 = L2)

        # print(f"feature arr shape: {feature_arr.shape}")

        all_feature_arr.append(feature_arr)

    all_feature_arr = torch.cat(all_feature_arr, dim=0)

    print(f"all feature arr shape: {all_feature_arr.shape}")

    all_feature_arr = reverse_chunks(all_feature_arr, num_seq)
    print(f'reverse chunk context vector shape: {all_feature_arr.shape}')
    all_feature_arr = reverse_seq(all_feature_arr, seq_length)
    print(f'reversed windown context vector shape: {all_feature_arr.shape}')

    # print(f"train mask set shape: {train_mask_set.shape}")

    print("Finished with DPC training")
    print("Start Poisson segmentation!")

    y_preds = []

    for i in range(all_feature_arr.shape[0]):

        (y_pred, x ,y_true) = poisson_segment(all_feature_arr[i], train_ts_set[i], train_mask_set[i])
        y_preds.append((y_pred, x ,y_true))

    data_dir = '/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/'
    for idx, (y_pred, x, y) in enumerate(y_preds):
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1)
        plt.title("Image")
        image = np.transpose(x[5,1:4,:,:], (1,2,0))
        # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
        plt.imshow(rescale_truncate(image))
        plt.subplot(1,3,2)
        plt.title("Segmentation Label")
        image = np.transpose(y[:,:], (0,1))
        plt.imshow(image)
    
        plt.subplot(1,3,3)
        plt.title(f"Segmentation Pred")
        image = np.transpose(y_pred[:,:], (0,1))
        plt.imshow(image)
        plt.savefig(f"{str(data_dir)}{ts_name}_{str(idx)}.png")

        plt.close()

    print('Poisson Semi Supervised Finished')

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train_dpc(data_loader, dpc_model, segment_model, optimizer, epoch, num_seq, seq_length):

    dpc_model.train()
    global iteration

    image_lst = []
    feature_lst = []

    for idx, input in enumerate(data_loader):

        tic = time.time()
        # print(f'id: {idx}')
        input_seq = input["x"]
        input_mask = input["mask"]

        (B,N,SL,C,H,W) = input_seq.shape

        input_seq = input_seq.to(cuda, dtype=torch.float32)
        input_mask = input_mask.to(dtype=torch.long)
        
        input_mask = input_mask.view(input_mask.shape[0], H, W)
        B = input_seq.size(0)

        features = dpc_model(input_seq)

        image_lst.append(input_seq)
        feature_lst.append(features)

    feature_arr = torch.cat(feature_lst, dim=0)

    return feature_arr.cpu().detach()


def predict_segment(data_loader, segment_model, optimizer):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
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


def poisson_segment(feature_arr, ts, label, input_size = 64):
    '''
    feature_arr: 4D tensor TxFxHxW
    '''
    #build dataset
    # print(f'ts shape: {ts.shape}')
    X = feature_arr.mean(axis=0) # (FxHxW)

    # X = ts.mean(axis=0)
    # X = feature_arr[5]

    label_old = label

    label = label.reshape((label.shape[0]*label.shape[1]))
    X = np.transpose(X,(1,2,0))
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[2])) # (4096x128)

    rate_train_per_class = 0.2
    train_ind = gl.trainsets.generate(label, rate=rate_train_per_class)
    train_labels = label[train_ind]

    # print(f'X shape {X.shape}')
    # print(f'label shape {label.shape}')
    gl.datasets.save(X,label,'hls-timeseries',overwrite=True)

    #Build a knn graph
    k = 1000
    W = gl.weightmatrix.knn(X, k=k)
    
    #Run Poisson learning
    class_priors = gl.utils.class_priors(label)
    model = gl.ssl.poisson(W, class_priors, solver='gradient_descent')
    pred_label = model.fit_predict(train_ind, train_labels)

    accuracy = gl.ssl.ssl_accuracy(pred_label, label, len(train_ind))   
    print("Accuracy: %.2f%%"%accuracy)
    
    segmented_image = pred_label.reshape((input_size,input_size))

    return segmented_image, ts, label_old


if __name__ == '__main__':
    main()
