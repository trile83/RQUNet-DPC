import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
# from dpc.model_3d import *
from dpc.model_3d_unet import *
import argparse
import h5py
import logging
from sklearn.metrics import balanced_accuracy_score, classification_report
import rioxarray as rxr
import xarray as xr
from inference import inference
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
import graphlearning as gl

torch.cuda.empty_cache()

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

def train_dpc(input_set, dpc_model):

    dpc_model.train()
    global iteration

    input_seq = torch.tensor(input_set)
    # input_seq = torch.unsqueeze(input_seq, 0)
    print(f'input seq shape: {input_seq.shape}')
    (B,N,SL,C,H,W) = input_seq.shape
    input_seq = input_seq.to(cuda, dtype=torch.float32)
    features = dpc_model(input_seq)

    return features.cpu().detach()

def predict_dpc_sliding(xraster, dpc_model):

    prediction = inference.sliding_window_tiler(
            xraster=xraster,
            model=dpc_model,
            n_classes=2,
            overlap=0.5,
            batch_size=1,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            model_option='dpc-unet-poisson'
        )
    
    print(f"feature shape after DPC {prediction.shape}")

    return prediction


def get_feature_arr(input_set, train_mask_set, num_seq, seq_length, device, input_size, train=False):
    

    network = 'unet' # 'resnet50', 'unet-vae', 'rqunet-vae-encoder', 'unet'
    if network == 'unet':
        encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0129_10band_unet_hls_98_2.7315708488893155e-06.pth' # unet
    else:
        encoder_weight = '/home/geoint/tri/dpc/models/checkpoints/recon_0217_10band_unetvae_hls_64_97_2.0649683759061257e-05.pth' # unet-vae

    # model_dir = "/home/geoint/tri/dpc/models/checkpoints/"

    model = DPC_RNN_UNet(sample_size=input_size,
                        device=device,
                        num_seq=num_seq, 
                        seq_len=seq_length, 
                        network=network,
                        pred_step=1,
                        model_weight=encoder_weight,
                        freeze=True,
                        segment=False)

    model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)

    all_feature_arr = []

    xraster = input_set
    all_feature_arr = predict_dpc_sliding(xraster, model)

    print(f'test feature arr shape {all_feature_arr.shape}')

    return all_feature_arr


def stack_features(feature_arr, label, feats, type='train', train_ind_lst=[], train_labels_lst=[]):

    # get mean of features array
    X = feature_arr.mean(axis=0) # (FxHxW)

    label = label.reshape((label.shape[0]*label.shape[1]))
    X = np.transpose(X,(1,2,0))
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
    # print(f'X shape: {X.shape}')

    if type == 'train':
        if feats.size == 0:
            pixel_vals = np.float32(X)
        else:
            pixel_vals = np.vstack((feats,np.float32(X)))

        rate_train_per_class = 5
        train_ind = gl.trainsets.generate(label, rate=rate_train_per_class)
        train_labels = label[train_ind]

        train_ind_lst = train_ind
        train_labels_lst = train_labels

    else:
        pixel_vals = np.vstack((feats, np.float32(X)))

    return pixel_vals, train_ind_lst, train_labels_lst


def get_data(ts_name_key, input_size, train=True):
    # prepare data
    ### hls data
    filename = "/home/geoint/tri/hls_ts_video/hls_data_final.hdf5"
    with h5py.File(filename, "r") as file:
        # print("Keys: %s" % f.keys())
        ts_arr = file[f'{str(ts_name_key)}_PEV_ts'][()]
        mask_arr = file[f'{str(ts_name_key)}_mask'][()]

    total_ts_len = 10 # L

    padding_size = 0

    # ignore the no-data edge of cut Tappan Square to HSL
    if ts_name_key == 'Tappan02' or ts_name_key == 'Tappan04':
        train_ts_set = ts_arr[:total_ts_len,1:-2,1:-1,1:160]
    else:
        train_ts_set = ts_arr[:total_ts_len,1:-2,2:-2,2:-2]

    ## get different chips in the Tappan Square for multiple time series

    # print(train_ts_set.shape)
    
    max_bound = -1
    ts_set = ts_arr[:10,1:-2,1:max_bound,1:max_bound]
    # ts_set = rescale_image(ts_arr[:10,1:-2,1:-1,1:-1])

    ts_set = rescale_image(ts_set)
    mask_set = mask_arr[1:max_bound,1:max_bound]

    print(f'{ts_name_key} ts shape: {ts_set.shape}')
    print(f'{ts_name_key} mask shape: {mask_set.shape}')
    return ts_set, mask_set
    

def poisson_predict(stacked_features, train_ind, train_labels, input_size = 64):

    X = stacked_features

   #Build a knn graph
    k = 1000
    W = gl.weightmatrix.knn(X, k=k)
    
    #Run Poisson learning
    # class_priors = gl.utils.class_priors(label)
    model = gl.ssl.poisson(W, solver='conjugate_gradient') # gradient_descent
    pred_label = model.fit_predict(train_ind, train_labels)

    # accuracy = gl.ssl.ssl_accuracy(pred_label, label, len(train_ind))   
    # print("Accuracy: %.2f%%"%accuracy)
    # segmented_image = pred_label.reshape((input_size,input_size))

    return pred_label


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ### hls data
    hls=False
    use_dpc=False

    if not hls:
        ts_name = 'Tappan05'
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

        # ts_arr = rescale_image(ts_arr)

        if 'PFV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PFV.2021081T110731.v2.0.tif'

        if 'PEV' in ts_name:
            ref_im_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls/HLS.S30.T28PEV.2022009T112441.v2.0.tif'

        ref_im = rxr.open_rasterio(ref_im_fl)

        print('reference image shape: ', ref_im.shape)

    ts_name_train = 'Tappan01'
    ts_name_test = 'Tappan05'

    seq_length = 6
    num_seq = 4
    input_size = 64

    # train_set, train_mask_set, train_ts_set = get_data(ts_name_train, input_size, train=True)
    train_set, train_mask_set = get_data(ts_name_train, input_size, train=False)
    test_set, test_mask_set = get_data(ts_name_test, input_size, train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_dpc:
        train_feature_arr = get_feature_arr(train_set,train_mask_set,num_seq,seq_length,device=device,input_size=64)
        test_feature_arr = get_feature_arr(test_set,test_mask_set,num_seq,seq_length,device=device,input_size=64)
    else:
        train_feature_arr = train_set
        test_feature_arr = test_set

    print(f'max test feature {np.max(test_feature_arr)}')
    print(f'min test feature {np.min(test_feature_arr)}')

    # print(f'test feature arr: {test_feature_arr.shape}')

    (T,C,H,W) = test_feature_arr.shape

    train_ind_lst = []
    train_labels_lst = []
    feats_ini = np.asarray([])
    
    train_features, train_ind_lst, train_labels_lst = stack_features(train_feature_arr,train_mask_set, \
        feats=feats_ini, type='train', train_ind_lst=train_ind_lst, train_labels_lst=train_labels_lst)
    
    test_features, test_ind_lst, test_labels_lst = stack_features(test_feature_arr,test_mask_set, \
        feats=train_features, type='test', train_ind_lst=train_ind_lst, train_labels_lst=train_labels_lst)

    print(f'train features shape: {train_features.shape}')

    stacked_features = test_features

    del feats_ini
    del train_features

    print(f'train ind arr shape: {train_ind_lst.shape}')
    print(f'train label arr shape: {train_labels_lst.shape}')
    print(f'stacked features shape: {stacked_features.shape}')

    labels_poisson = poisson_predict(stacked_features, train_ind_lst, train_labels_lst)
    print(f'labels poisson shape: {labels_poisson.shape}')

    num_chips = train_set.shape[0]
    test_size = test_set.shape[0]

    segment_images=[]
    for i in range(num_chips+test_size):
        segment_images.append(labels_poisson[i*(H*W):(i+1)*(H*W)])

    for idx in range(len(segment_images)):
        if idx == 0:
            train_pred = segment_images[idx]
            train_pred = train_pred.reshape((H,W))
        elif idx == 1:
            test_pred = segment_images[idx]
            test_pred = test_pred.reshape((H,W))

    prediction = test_pred
    print(f'max prediction {np.max(prediction)}')
    print(f'min prediction {np.min(prediction)}')
    print(f'prediction shape: {prediction.shape}')

    if hls:
        ref_im = ref_im.transpose("y", "x", "band")

    data_dir = '/home/geoint/tri/dpc/models/output/dpc_unetvae_0927/'

    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title("Image")
    # image = np.transpose(test_feature_arr[5,:3,:,:], (1,2,0))
    image = np.transpose(train_set[5,:3,:,:], (1,2,0))
    if hls:
        image= rescale_image(xr.where(image > -9000, image, 600))
    else:
        image= rescale_image(xr.where(image > -1000, image, 150))

    plt.imshow(rescale_truncate(image))

    plt.subplot(1,3,2)
    plt.title("Label")
    image = train_mask_set
    plt.imshow(image)

    plt.subplot(1,3,3)
    plt.title(f"Prediction")
    image = train_pred
    plt.imshow(image)
    if use_dpc:
        plt.savefig(f"{str(data_dir)}{ts_name_train}-dpc-unet-poisson-train.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{str(data_dir)}{ts_name_train}-poisson-train.png", dpi=300, bbox_inches='tight')

    plt.close()

    del image


    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title("Image")
    # image = np.transpose(test_feature_arr[5,:3,:,:], (1,2,0))
    image = np.transpose(test_set[5,:3,:,:], (1,2,0))
    if hls:
        image= rescale_image(xr.where(image > -9000, image, 600))
    else:
        image= rescale_image(xr.where(image > -1000, image, 150))

    plt.imshow(rescale_truncate(image))

    plt.subplot(1,3,2)
    plt.title("Label")
    image = test_mask_set
    plt.imshow(image)

    plt.subplot(1,3,3)
    plt.title(f"Prediction")
    image = test_pred
    plt.imshow(image)
    if use_dpc:
        plt.savefig(f"{str(data_dir)}{ts_name_test}-dpc-unet-poisson-test.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{str(data_dir)}{ts_name_test}-poisson-test.png", dpi=300, bbox_inches='tight')

    plt.close()

    del image


if __name__ == '__main__':
    main()

    # python main.py --gpu 0 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --epochs 100
