import disstl.models as models
import re
import torch
import torchvision
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from disstl.datasets.smart.datasets import from_cube
from disstl.datasets.transforms import ClipBands, MinMaxNormalize
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
from dpc.model_3d import *
from dpc.model_3d_unet import *
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
from tqdm import tqdm
import argparse
import pickle
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn-unet', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
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

    map_img =  np.zeros((32,32,3))
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
        Y = self.mask[index]

        return {
            'x': X,
            'mask': torch.LongTensor(Y)
        }


def get_seq(sequence, seq_length):
    '''
    sequence: long time-series input = 79
    seq_length: length of window for time-series chunk; default = 5
    return: array with all time-series chunk; prediction size =1, num_seq = 4
    '''
    (L,C,H,W) = sequence.shape
    all_arr = np.zeros((L-seq_length,seq_length,C,H,W))
    for i in range(L-seq_length):
        array = sequence[i:i+seq_length,:,:,:] # SL, C, H, W
        all_arr[i] = array

    # print(f'data array shape: {all_arr.shape}')

    return np.array(all_arr)


def get_chunks(windows, num_seq):
    '''
    windows: number of windows in 1 time-series
    number: number of window for time-series chunk; default = 4
    return: array with all time-series chunk; prediction size =1, num_seq (N) = 4; N x 5 x 5 x 32 x 32
    '''
    (L,SL,C,H,W) = windows.shape
    all_arr = np.zeros((L-num_seq,num_seq,SL,C,H,W))
    for i in range(L-num_seq):
        array = windows[i:i+num_seq,:,:,:,:] # N, SL, C, H, W
        all_arr[i] = array

    return np.array(all_arr)



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    num_seq = 4
    seq_length = 6

    # prepare data
    ### datacube BlackSky

    satellite = 'sentinel2'
    region = 'BH_R001'
    path_to_cubes ='/home/geoint/tri/dpc/models/BH_R001/region.hdf5'

    # region = 'BLA_QFABRIC_R007'
    # path_to_cubes ='E:/BlackSky/temp/cubes_new/BLA_QFABRIC_R007/region.hdf5'
    band_mins = [0, 0, 0, 0, 0]
    band_maxs = [17456, 18379, 19528, 16610, 15271]
    # band_mins = [0, 0, 0]
    # band_maxs = [17456, 18379, 19528]

    annotation_kwargs = dict(active_to_inactive_ratio=0.5,
                        n_active_sequence_draws=2,
                        n_inactive_sequence_draws=1
                        )

    transforms = torchvision.transforms.Compose([ClipBands(mins=band_mins, maxs=band_maxs),
                                                MinMaxNormalize(mins=band_mins, maxs=band_maxs)])

    region_id2idx = {region: 0}
    train_dataset_kwargs = dict(path=path_to_cubes,
                                chip_shape=[32,32],
                                stride=[16,16],
                                band_names=['red','green', 'blue', 'nir', 'swir16'],
                                # band_names=['red','green', 'blue'],
                                seq_len=5,
                                transforms=transforms,
                                spectral_indices=None,
                                with_annotations=True,
                                annotation_kwargs=annotation_kwargs,
                                satellite=satellite,
                                region_id=region,
                                region_id2idx=region_id2idx,
                                use_cache_dataset_preprocessing=False
                                )

    test_dataset_kwargs = dict(path=path_to_cubes,
                                chip_shape=[32,32],
                                stride=[16,16],
                                band_names=['red','green', 'blue', 'nir', 'swir16'],
                                # band_names=['red','green', 'blue'],
                                seq_len=79,
                                transforms=transforms,
                                spectral_indices=None,
                                with_annotations=True,
                                annotation_kwargs=annotation_kwargs,
                                satellite=satellite,
                                region_id=region,
                                region_id2idx=region_id2idx,
                                use_cache_dataset_preprocessing=False
                                )

    # from_cube returns an instance of AnnotatedDataset, defined in disstl/datasets/smart/datasets.py
    # train_ds = from_cube(**train_dataset_kwargs)
    # # val_ds = from_cube(**train_dataset_kwargs)

    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, drop_last=True, shuffle=True)
    # val_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, drop_last=True, shuffle=True)

    # print('number of training ds: ', len(train_dl))

    test_ds = from_cube(**test_dataset_kwargs)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, drop_last=True, shuffle=True)

    for idx, input in enumerate(test_dl):
        # tic = time.time()
        # print(f'id: {idx}')
        input_seq = input["x"]
        input_mask = input["binary_activity_mask"]
        # print(f'input mask shape: {input_mask.shape}')

        input_seq = torch.reshape(input_seq,(input_seq.shape[1],input_seq.shape[2],input_seq.shape[3],input_seq.shape[4]))
        input_mask = torch.reshape(input_mask,(input_mask.shape[1],1,input_mask.shape[2],input_mask.shape[3]))
        # input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        print(f'long time-series shape: {input_seq.shape}')
        print(f'input mask shape: {input_mask.shape}')

        if idx == 5:
            break

    im_set = get_seq(input_seq, seq_length)
    mask_set = get_seq(input_mask, seq_length)

    print(f'window sequence shape: {im_set.shape}')

    new_set = get_chunks(im_set, num_seq)
    new_m_set = get_chunks(mask_set, num_seq)

    print(f'chunk sequence shape: {new_set.shape}')

    test_set = satDataset(new_set, new_m_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=2, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    train_dl = DataLoader(test_set, **loader_args)
    val_dl = DataLoader(test_set, **loader_args)

    for idx, input in enumerate(train_dl):
        # tic = time.time()
        input_seq = input["x"]
        input_mask = input["mask"]
        print(f'test im shape: {input_seq.shape}')
        print(f'test mask shape: {input_mask.shape}')

        # input_seq = torch.reshape(input_seq,(input_seq.shape[1],input_seq.shape[2],input_seq.shape[3],input_seq.shape[4]))
        # input_seq = input_seq.to(cuda)
        # B = input_seq.size(0)
        # print(f'input shape: {input_seq.shape}')

        image = np.transpose(input_seq.numpy()[0,0,0,1:4,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image))
        plt.savefig("/home/geoint/tri/dpc_test_out/input_image.png")
        plt.close()

        image = np.transpose(input_mask.numpy()[0,0,0,:,:,:], (1,2,0))  
        plt.imshow(image)
        plt.savefig("/home/geoint/tri/dpc_test_out/input_mask.png")
        plt.close()

        # del image

        break
        
    ### dpc model ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/reconstruct_5band_unetvae_24_0.000947555701714009.pth'

    model = DPC_RNN_UNet(sample_size=32,
                    device=device,
                    num_seq=4, 
                    seq_len=5, 
                    network='resnet18', 
                    pred_step=1,
                    model_weight=model_checkpoint,
                    freeze=True)

    unet_segment = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)

    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)
        unet_segment = unet_segment.to(cuda)


    ### optimizer ###
    # if args.train_what == 'last':
    #     for name, param in model.module.resnet.named_parameters():
    #         param.requires_grad = False
    # else: pass # train all layers

    # print('\n===========Check Grad============')
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # print('=================================\n')


    # setup tools
    # global de_normalize; de_normalize = denorm()
    # global img_path; img_path
    model_dir = "/home/geoint/tri/dpc/models/checkpoints/"
    model_path = model_dir+"model_best_epoch198.pth.tar"
    unetsegment_path = model_dir+"unetsegment_epoch198.pth"
    # global writer_train
    

    predictions = predict(input_seq, model, unet_segment, model_path, unetsegment_path, device)

    pred = predictions[20]

    print(f"pred shape {pred.shape}")

    pred = np.transpose(pred.cpu().detach().numpy()[:,:])
    plt.imshow(pred)
    plt.savefig("/home/geoint/tri/dpc_test_out/pred.png")
    plt.close()

    

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def predict(input_seq, dpc_model, segment_model, model_saved, unetsegment_saved, device):

    checkpoint_dpc = torch.load(model_saved)
    dpc_model.load_state_dict(checkpoint_dpc['state_dict'], strict=False)
    dpc_model = dpc_model.to(cuda)

    checkpoint_segment = torch.load(unetsegment_saved, map_location=device)
    segment_model.load_state_dict(checkpoint_segment['state_dict'], strict=True)
    segment_model = segment_model.to(cuda)

    tic = time.time()
    # print(f'id: {idx}')
    # data = data["x"]
    # print(f'data shape: {data.shape}')
    # data = torch.reshape(data,(2,8,data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    # data = data.to(cuda, dtype=torch.float32)

    # input_seq = input["x"]
    # input_mask = input["mask"]
    (B,N,SL,C,H,W) = input_seq.shape

    input_seq = input_seq.to(cuda, dtype=torch.float32)
    
    B = input_seq.size(0)

    with torch.no_grad():
        features = dpc_model(input_seq)
        # features = features.view(B, N, SL, C, H, W)

        features = features.to(cuda, dtype=torch.float32)

        probs, _ = segment_model(features)

        print(probs[0])

        full_masks = torch.argmax(probs, dim=1)

    print(f"mask shape {full_masks.shape}")

    return full_masks


if __name__ == '__main__':
    main()

    # python main.py --gpu 0,1 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --epochs 100
