import numpy as np
import rasterio as rio
import rioxarray as rxr
import re
from skimage import exposure
import glob
import os
import h5py
import matplotlib.pyplot as plt
import logging

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

def read_data(
    master_dir: str,
    tile: str ='PEA',
    mask_dir_path: str ='/home/geoint/tri/resampled_senegal_hls/filtered/'
    ):

    fl_dir = sorted(glob.glob(f"{master_dir}/output-{tile}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    mask_dict={}
    unique_name = []

    mask_dir_path = os.path.join(mask_dir_path, tile)

    mask_lst = sorted(glob.glob(f'{mask_dir_path}/*.tif'))

    for idx, file in enumerate(fl_dir):
        name = re.search(f'{tile}/(.*?).tif', file).group(1)[:22]
        # print(name)
        # if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
        #     continue
        if name not in unique_name: # check if ts name is seen or not
            anchor_name = name

            unique_name.append(anchor_name)

            ts_dict[anchor_name] = []
            mask_dict[anchor_name] = []

        if name in file:

            img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

            # print(img_data.shape)

            ts_dict[name].append(img_data)

        for mask_fl in mask_lst:
            if name in mask_fl:
                mask_data = np.squeeze(rxr.open_rasterio(mask_fl, masked=False).values)
                mask_dict[name].append(mask_data)

    output_dict = {}

    for key in ts_dict.keys():
        print(key)
        try:
            output_dict[key] = np.stack(ts_dict[key], axis=0)
        except:
            print(ts_dict[key][-1].shape)

    del ts_dict

    return output_dict, mask_dict


def get_composite(ts_arr):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[0] // 10
    #print(step)

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[0], step):
        out_lst.append(ts_arr[i])

    out_array = np.stack(out_lst, axis=0)
    del ts_arr

    print(out_array.shape)

    return out_array

## Input function to calculate NDVI

def cal_ndvi(image):

    # print(image.shape)

    image = np.transpose(image, (1,2,0))
    
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide((image[:,:,8]-image[:,:,3]), (image[:,:,8]+image[:,:,3]))
    return ndvi



if __name__ == "__main__":

    tile='PEA'
    master_dir = "/home/geoint/tri/match-hls-sen/"
    ts_dict, mask_dict = read_data(master_dir, tile)

    # for key in ts_dict.keys():
    #     print(key)
    #     print(ts_dict[key].shape)
    #     print(mask_dict[key][0].shape)

    out_dir = '/home/geoint/tri/hls_datacube'


    # #########################
    # h = h5py.File(f'{out_dir}/hls-etz-{tile}-0908.hdf5', 'w')

    # for k, v in ts_dict.items():
    #     h.create_dataset(f"{k}_{tile}_ts", data=v, compression="gzip", compression_opts=9)
    #     h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k][0], compression="gzip", compression_opts=9)
    # print(f'finished the data processing')

    for key in sorted(list(ts_dict.keys())):

        ts_arr = ts_dict[key]
        print(key)
        # print(ts_arr.shape)

        ts_arr = get_composite(ts_arr)

        for frame in range(ts_arr.shape[0]):
            ndvi = cal_ndvi(ts_arr[frame])
            ## check if high NDVI pixel (vegetation) is larger than 10% of total number of pixels
            if np.count_nonzero(ndvi > 0.05)/(ts_arr.shape[2]*ts_arr.shape[3])*100 > 10:
                print(f'At frame {frame} NDVI > 0.05, number of "green" pixels in the images')
                print(np.count_nonzero(ndvi > 0.1)/(ts_arr.shape[2]*ts_arr.shape[3])*100)
                a = np.transpose(ts_arr[frame], (1,2,0))
                a = a / 10000
                b = np.stack([a[:,:,8], a[:,:,3],a[:,:,2]], axis=2)
                #print(b.shape)


    ## Test load h5py file
    print("Test load h5py file")
    if tile != 'PEA':
        filename = f'{out_dir}/hls-ecas-{tile}-0901.hdf5'
    else:
        filename = f'{out_dir}/hls-eetz-{tile}-0908.hdf5'

    # with h5py.File(filename, "r") as file:
    #     # ts_arr = file[f'Tappan19_WV02_20170126_{tile}_ts'][()]
    #     # mask_arr = file[f'Tappan19_WV02_20170126_{tile}_mask'][()]

    #     print(sorted(list(file.keys())))

    #     for key in sorted(list(file.keys())):
    #         if 'ts' not in key:
    #             continue

    #         ts_arr = file[key]
    #         print(key)
    #         # print(ts_arr.shape)

    #         ts_arr = get_composite(ts_arr)

    #         for frame in range(ts_arr.shape[0]):
    #             ndvi = cal_ndvi(ts_arr[frame])
    #             ## check if high NDVI pixel (vegetation) is larger than 10% of total number of pixels
    #             if np.count_nonzero(ndvi > 0.05)/(ts_arr.shape[2]*ts_arr.shape[3])*100 > 10:
    #                 print(f'At frame {frame} NDVI > 0.05, number of "green" pixels in the images')
    #                 print(np.count_nonzero(ndvi > 0.1)/(ts_arr.shape[2]*ts_arr.shape[3])*100)
    #                 a = np.transpose(ts_arr[frame], (1,2,0))
    #                 a = a / 10000
    #                 b = np.stack([a[:,:,8], a[:,:,3],a[:,:,2]], axis=2)
    #                 #print(b.shape)
    #                 # plt.imshow(rescale_truncate(rescale_image(b)))
    #                 # plt.show()


    # print(ts_arr.shape)
    # print(mask_arr.shape)


    ## Create stepping composite for long timeseries

    # out_array = get_composite(ts_arr)
