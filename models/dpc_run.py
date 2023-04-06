import disstl.models as models
import re
# from this import d
import torch
import torchvision
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, models
from disstl.datasets.smart.datasets import from_cube
from disstl.datasets.transforms import ClipBands, MinMaxNormalize
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
import logging
from dpc.model_3d import *
from dpc.model_3d_unet import *
from unet3d.unet3d import UNet3D
from backbone.resnet_2d3d import neq_load_customized
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
from tqdm import tqdm
import argparse
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
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
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

        # print(f"Y {Y}")

        # y = F.one_hot(Y)

        # print(f"y shape {y.shape}")

        return {
            'x': X,
            'mask': Y
        }


def get_seq(sequence, seq_length):
    '''
    sequence: long time-series input = 79
    seq_length: length of window for time-series chunk; default = 6
    return: array with all time-series chunk; prediction size =1, num_seq = 4
    '''
    (L,C,H,W) = sequence.shape
    all_arr = torch.zeros((L-seq_length,seq_length,C,H,W))
    # for i in range(seq_length, L-seq_length):
    for i in range(seq_length, L-1):
        # array = sequence[i:i+seq_length,:,:,:] # SL, C, H, W
        array = sequence[i-seq_length:i,:,:,:] # SL, C, H, W
        all_arr[i-seq_length] = array

    # print(f'data array shape: {all_arr.shape}')
    return all_arr


def get_chunks(windows, num_seq):
    '''
    TODO: match with get_seq function
    windows: number of windows in 1 time-series
    number: number of window for time-series chunk; default = 4
    return: array with all time-series chunk; prediction size =1, num_seq (N) = 4; N x 6 x 5 x 32 x 32
    '''
    (L,SL,C,H,W) = windows.shape
    all_arr = torch.zeros((L-num_seq,num_seq,SL,C,H,W))
    for i in range(num_seq, L-1):
        array = windows[i-num_seq:i,:,:,:,:] # N, SL, C, H, W
        all_arr[i-num_seq] = array

    return all_arr


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
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, drop_last=True, shuffle=False)

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
    loader_args = dict(batch_size=5, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
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

        image = np.transpose(input_seq.numpy()[2,0,0,1:4,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image))
        plt.savefig("/home/geoint/tri/dpc_test_out/image.png")
        plt.close()

        del image

        image = np.transpose(input_mask.numpy()[2,0,0,:,:,:], (1,2,0))  
        plt.imshow(image)
        plt.savefig("/home/geoint/tri/dpc_test_out/mask.png")
        plt.close()

        del image

        break
        
    ### dpc model ###
    # model = DPC_RNN(sample_size=32, 
    #                 num_seq=8, 
    #                 seq_len=1, 
    #                 network='resnet18', 
    #                 pred_step=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/reconstruct_5band_unetvae_89_1.479999627918005e-05.pth'

    model = DPC_RNN_UNet(sample_size=32,
                    device=device,
                    num_seq=4, 
                    seq_len=6, 
                    network='resnet18', 
                    pred_step=1,
                    model_weight=model_checkpoint,
                    freeze=True)

    unet_segment = UNet_VAE_old(num_classes=2,segment=True,in_channels=128)
    # unet_segment = UNet3D(in_channel=5, n_classes=2)

    # model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)
        unet_segment = unet_segment.to(cuda)

    weights = [0.01,3]
    class_weights = torch.FloatTensor(weights).cuda()

    global criterion; 
    criterion_type = "crossentropy"
    if criterion_type == "crossentropy":
        criterion = models.losses.MultiTemporalCrossEntropy(class_weights=weights)
    elif criterion_type == "focal_tverski":
        criterion = models.losses.FocalTversky()
    elif criterion_type == "dice":
        criterion = models.losses.MultiTemporalDiceLoss()

    ### optimizer ###
    # if args.train_what == 'last':
    #     for name, param in model.module.resnet.named_parameters():
    #         param.requires_grad = False
    # else: pass # train all layers

    # print('\n===========Check Grad============')
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.000)
    args.old_lr = None

    best_acc = 0
    min_loss = np.inf
    global iteration; iteration = 0

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
    
    ### main loop ###
    train_loss_lst = []
    val_loss_lst = []

    print(f"length of dpc input training set {len(train_dl)}")
    # for epoch in range(1):
    for epoch in range(args.start_epoch, args.epochs):
        
        # train_loss = train(train_dl, model, unet_segment, optimizer, epoch, num_seq, seq_length)

        feature_arr, mask_arr = train(train_dl, model, unet_segment, optimizer, epoch, num_seq, seq_length)

    print(f"feature arr shape: {feature_arr.shape}")
    print(f"mask arr shape: {mask_arr.shape}")

    # for i in range(len(mask_arr)):
    #     image = np.transpose(mask_arr[0,:,:])  
    #     plt.imshow(image)
    #     plt.savefig("/home/geoint/tri/dpc_test_out/mask_after.png")
    #     plt.close()

    print(f"mask output from dpc: {mask_arr[0,0,0,:,:]}")

    print("Finished with DPC training")

    train_seg_set = satDataset(feature_arr,  mask_arr)
    loader_args_1 = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_segment_dl = DataLoader(train_seg_set, **loader_args_1)

    # test_set_1 = satDataset(new_set[:5], new_m_set[:5])
    # test_set_2 = satDataset(new_set[-3:], new_m_set[-3:])
    # # 3. Create data loaders
    # loader_args = dict(batch_size=5, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    # train_dl_1 = DataLoader(test_set_1, **loader_args)
    # val_dl_1 = DataLoader(test_set_2, **loader_args)

    print(f"Length of segmentation input training set {len(train_segment_dl)}")
    print("Start segmentation training!")

    for idx, input in enumerate(train_segment_dl):
        # tic = time.time()
        input_seq = input["x"]
        input_mask = input["mask"]
        print(f'test im shape: {input_seq.shape}')
        print(f'test mask shape: {input_mask.shape}')

        image = np.transpose(input_seq.numpy()[0,0,0,:3,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image))
        plt.savefig("/home/geoint/tri/dpc_test_out/image_after.png")
        plt.close()

        del image

        image = np.transpose(input_mask.numpy()[0,0,0,:,:])  
        plt.imshow(image)
        plt.savefig("/home/geoint/tri/dpc_test_out/mask_after.png")
        plt.close()

        del image
        break

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_segment(train_segment_dl, unet_segment, optimizer)

        val_loss = val_segment(train_segment_dl, unet_segment, optimizer)

        # saved loss value in list
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)

        print(f"train loss: {train_loss}")
        print(f"val loss: {val_loss}")

        # save check_point

        is_best = val_loss < min_loss; min_loss = min(val_loss, min_loss)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'min_loss': min_loss,
                         'optimizer': optimizer.state_dict()}, 
                         is_best, filename=os.path.join(model_dir, 'epoch%s.pth' % str(epoch+1)), keep_all=False)

        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': unet_segment.state_dict(),
                         'min_loss': min_loss,
                         'optimizer': optimizer.state_dict()}, 
                         is_best, filename=os.path.join(model_dir, 'unetsegment_epoch%s.pth' % str(epoch+1)), keep_all=False)

    plt.plot(train_loss_lst, color ="blue")
    plt.plot(val_loss_lst, color = "red")
    plt.savefig("/home/geoint/tri/dpc_test_out/train_loss.png")
    plt.close()

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, dpc_model, segment_model, optimizer, epoch, num_seq, seq_length):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    dpc_model.train()
    #segment_model.train()
    global iteration

    image_lst = []
    feature_lst = []
    mask_lst = []

    for idx, input in enumerate(data_loader):
        tic = time.time()
        # print(f'id: {idx}')
        input_seq = input["x"]
        input_mask = input["mask"]
        (B,N,SL,C,H,W) = input_seq.shape

        input_seq = input_seq.to(cuda, dtype=torch.float32)
        input_mask = input_mask.to(dtype=torch.long)
        # input_mask = rearrange(input_mask, "b n sl c h w -> (b n) sl c h w")
        input_mask = input_mask.view(input_mask.shape[0],input_mask.shape[1],input_mask.shape[2], H, W)
        B = input_seq.size(0)
 
        # [score_, mask_] = model(input_seq)
        features = dpc_model(input_seq)
        
        # if idx == 0:
        #     print(f'input shape: {input_seq.shape}')
        #     print(f'input mask shape: {input_mask.shape}')
        #     print(f"output features shape {features.shape}")


        # features = features.to(cuda, dtype=torch.float32)

        # features= rearrange(features, "b n f h w -> b n f h w")
        # input_seq= rearrange(input_seq, "b n sl c h w -> b n sl c h w")

        # output = segment_model(features)
        # mask_pred = output[0]

        # # print(f'mask pred shape {mask_pred.shape}')
        
        # loss = criterion(mask_pred, input_mask)

        # losses.update(loss.item(), B)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if idx == 1:
            break

        image_lst.append(input_seq)
        feature_lst.append(features)
        mask_lst.append(input_mask)

    image_arr = torch.cat(image_lst, dim=0)
    feature_arr = torch.cat(feature_lst, dim=0)
    mask_arr = torch.cat(mask_lst, dim=0)

    # return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]
    # return feature_arr.cpu().detach().numpy(), mask_arr.cpu().detach().numpy()
    # return image_arr.cpu().detach(), mask_arr.cpu().detach()
    return feature_arr.cpu().detach(), mask_arr
    # return losses.local_avg

def train_segment(data_loader, segment_model, optimizer):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    segment_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        features = input['x'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,N,SL,F,H,W) = features.shape

        features = features[0,0,:,:,:,:]
        input_mask = input_mask[0,0,:,:,:]

        features = features.view(B*N*SL,F,H,W)
        input_mask = input_mask.view(B*N*SL,H,W)

        mask_pred, _ = segment_model(features)

        loss = criterion(mask_pred, input_mask)
        losses.update(loss['loss'].item(), B)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

    # return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]
    return losses.local_avg


def validate(data_loader, dpc_model, segment_model):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    dpc_model.eval()

    n_val = len(data_loader)
    # print(f"length of validation set: ", n_val)

    with torch.no_grad():
        for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_seq = input["x"]
            input_mask = input["mask"]
            (B,N,SL,C,H,W) = input_seq.shape

            input_seq = input_seq.to(cuda, dtype=torch.float32)
            input_mask = input_mask.to(cuda, dtype=torch.long)
            input_mask = rearrange(input_mask, "b n t c h w -> (b n t) c h w")
            input_mask = input_mask.view(input_mask.shape[0], H, W)
            B = input_seq.size(0)
            
            # [score_, mask_] = model(input_seq)
            features = dpc_model(input_seq)
            # features = features.view(B, N, SL, C, H, W)


            # if idx == 50:
            #     print(f'input shape: {input_seq.shape}')
            #     print(f'input mask shape: {input_mask.shape}')
            #     print(f"output features shape {features.shape}")

            del input_seq

            features = features.to(cuda, dtype=torch.float32)
            features= rearrange(features, "b n f h w -> (b n) f h w")
            mask_pred, _ = segment_model(features)
            loss = criterion(mask_pred,input_mask)
            losses.update(loss.item(), B)

    return losses.local_avg


def val_segment(data_loader, segment_model, optimizer):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    segment_model.eval()
    global iteration

    with torch.no_grad():
        for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):

            # input_seq = input["x"]
            # input_mask = input["mask"]

            # (B,N,SL,C,H,W) = input_seq.shape
            # input_seq = input_seq.to(cuda, dtype=torch.float32)
            # input_mask = input_mask.to(cuda, dtype=torch.long)
            # input_mask = rearrange(input_mask, "b n t c h w -> (b n t) c h w")
            # input_mask = input_mask.view(input_mask.shape[0], H, W)

            # input_seq = rearrange(input_seq, "b n t c h w -> (b n t) c h w")
            # mask_pred, _ = segment_model(input_seq)

            features = input['x'].to(cuda, dtype=torch.float32)
            input_mask = input['mask'].to(cuda, dtype=torch.long)

            (B,N,SL,F,H,W) = features.shape

            features = features.view(B*N*SL,F,H,W)
            input_mask = input_mask.view(B*N*SL,H,W)

            mask_pred, _ = segment_model(features)

            loss = criterion(mask_pred, input_mask)
            losses.update(loss['loss'].item(), B)

    # return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]
    return losses.local_avg


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

    # python main.py --gpu 0,1 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --epochs 100
