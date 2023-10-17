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
import logging
from dpc.model_3d import *
from dpc.model_3d_unet import *
import disstl.models as models
from unet3d.unet3d import UNet3D
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
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs to run')
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

def rescale_image(image: np.ndarray, rescale_type: str = 'per-channel'):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    # image = image.astype(np.float32)
    image = image.to(dtype=torch.float32)
    if rescale_type == 'per-image':
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - torch.min(image[i, :, :])) / \
                (torch.max(image[i, :, :]) - torch.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image


class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y
        self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]

        X = rescale_image(X)

        print(f"min X: {torch.min(X)}")
        print(f"max X: {torch.max(X)}")

        return {
            'x': X,
            'mask': Y
        }


def train_segment(data_loader, segment_model, optimizer):
    losses = AverageMeter()
    
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    segment_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        features = input['x'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,F,H,W) = features.shape

        # features = features.view(B,F,H,W)
        input_mask = torch.reshape(input_mask,(B,H,W))

        mask_pred, _ = segment_model(features)

        loss = criterion(mask_pred, input_mask)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]
    return losses.local_avg


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

    test_dataset_kwargs = dict(path=path_to_cubes,
                                chip_shape=[32,32],
                                stride=[16,16],
                                band_names=['red','green', 'blue', 'nir', 'swir16'],
                                # band_names=['red','green', 'blue'],
                                seq_len=32,
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

    unet_segment = UNet3D(in_channel=5, n_classes=2)
    global criterion; 
    # criterion = nn.CrossEntropyLoss()
    criterion = models.losses.FocalTversky()
    params = unet_segment.parameters()
    optimizer = optim.Adam(params, lr=0.0003, weight_decay=0.0)
    args.old_lr = None

    dir_checkpoint = '/home/geoint/tri/dpc/models/checkpoints/'

    # model_path ='/home/geoint/tri/dpc_test_out/checkpoints/segment_BH_R001_unet3d_focal_tverski_loss_35_0.7700747720129311.pth'
    model_path ='/home/geoint/tri/dpc_test_out/checkpoints/segment_BH_R001_unet3d_crossentropy_loss_66_0.10318317332286854.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    unet_segment.load_state_dict(torch.load(model_path, map_location=device))

    if torch.cuda.is_available():
        unet_segment = unet_segment.to(cuda)

    unet_segment.eval()
        

    num_seq = 7

    # unet_segment = UNet_VAE_old(num_classes=2,segment=True,in_channels=5)
    with torch.no_grad():
        for idx, input in enumerate(test_dl):
            input_seq = input["x"].to(cuda, dtype=torch.float32)
            input_mask = input["binary_activity_mask"]

            print(f"input seg shape {input_seq.shape}")
            print(f"input mask shape {input_mask.shape}")

            probs = unet_segment(input_seq)
            print(f"probs shape : {probs.shape}")

            full_mask = torch.argmax(probs, dim=2)
            full_mask = torch.squeeze(full_mask).cpu().numpy()
            print(f"full_mask shape {full_mask.shape}")

            ## plot images

            plt.figure(figsize=(20,20))
            plt.subplot(1,3,1)
            plt.title("Image")

            a = input_seq.cpu().numpy()[:,num_seq,:3,:,:]
            print(f"a shape: {a.shape}")

            # for num_seq in range(input_seq.shape[1]):
            image = np.transpose(a[0], (1,2,0))  
            plt.imshow(rescale_truncate(image))
            # plt.savefig(f"/home/geoint/tri/dpc_test_out/{idx}_{num_seq}_image_input.png")
            # plt.close()

            del image
            del a

            plt.subplot(1,3,2)
            plt.title("Segmentation Label")

            b = input_mask.cpu().numpy()[:,num_seq,:,:]
            print(f"b shape: {b.shape}")
            image = np.transpose(b, (1,2,0))  
            plt.imshow(image)
            # plt.savefig(f"/home/geoint/tri/dpc_test_out/{idx}_{num_seq}_mask_gt.png")
            # plt.close()

            del image
            del b

            plt.subplot(1,3,3)
            plt.title("Segmentation Prediction")

            c = full_mask[num_seq,:,:]
            print(f"c shape: {c.shape}")
            image = np.transpose(c, (0,1))  
            plt.imshow(image)
            plt.savefig(f"/home/geoint/tri/dpc_test_out/{idx}_{num_seq}_results.png")
            plt.close()

            del image
            del c

        # image = np.transpose(probs[idx,im_id,0,:,:].detach().cpu().numpy())  
        # plt.imshow(image)
        # plt.savefig(f"/home/geoint/tri/dpc_test_out/{idx}_mask_pred_1.png")
        # plt.close()

        # del image

        # image = np.transpose(probs[idx,im_id,1,:,:].detach().cpu().numpy())  
        # plt.imshow(image)
        # plt.savefig(f"/home/geoint/tri/dpc_test_out/{idx}_mask_pred_2.png")
        # plt.close()

        # del image

        # break



if __name__ == '__main__':
    main()