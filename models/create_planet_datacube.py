import numpy as np
import tifffile
import rioxarray as rxr
import re
import glob
import os
import h5py
from skimage import exposure
import logging
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

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

def read_data(master_dir, tile='PEA', year_set='2016', mask_dir_path=''):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    unique_name = []

    count=0
    for idx, file in enumerate(fl_dir):

        if tile not in ts_dict.keys(): # check if ts name is seen or not
            ts_dict[tile] = {}

        if year_set not in ts_dict[tile].keys():
            ts_dict[tile][year_set] = []

        try:
            img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)
            #img_data = tifffile.imread(file)
            print(f"image shape: {img_data.shape}")

            print('img_data max: ', np.max(img_data))
            print('img_data min: ', np.min(img_data))

            count+=1
            ts_dict[tile][year_set].append(img_data[:-1,:,:])
        except:
            print('Cannot read this image: ', file)

    print("number of image after processing: ", count)

    output_dict = {}

    for tilename in ts_dict.keys():
        for year in ts_dict[tilename].keys():
            outname = f'{tilename}_{year}'
            if len(ts_dict[tilename][year]) > 0:
                print('time series year with at least 1 scene: ', year)
                output_dict[outname] = np.stack(ts_dict[tilename][year], axis=0)

    del ts_dict

    return output_dict


def get_composite(ts_arr):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[0] // 10
    print(step)

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[0], step):
        out_lst.append(np.median(ts_arr[i:i+step], axis=0))

    out_array = np.stack(out_lst, axis=0)
    del ts_arr

    print(out_array.shape)

    return out_array

def plot_timeseries(
    train_ts_set,
    name,
    dates,
    tile
    ):

    height = 3
    width = 4

    classes = {
            0:'purple',
            1:'yellow',
            2:'black'
            }
    # convert all colors to a list
    colors = [classes[id] for id in classes.keys()]
    colormap = pltc.ListedColormap(colors)

    plt.figure(figsize=(20,20))
    for idx in range(1,height*width-1):
        plt.subplot(height,width,idx)
        if idx < 11:
            plt.title(f'Day {dates[idx-1]}')
            image = np.transpose(train_ts_set[(idx-1),:3,:,:], (1,2,0))
            image= rescale_image(xr.where(image > -9000, image, -1000))
            plt.imshow(rescale_truncate(image))
        
        # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    tile='planet-cas'
    year="2021"

    master_dir = "/home/geoint/tri/planet-data/tile01/"
    ts_dict = read_data(master_dir, tile, year)

    for key in ts_dict.keys():
        print(key)
        print(ts_dict[key].shape)
        # print(mask_dict[key][0].shape)

    plot_timeseries(ts_dict[key], key, ['1','2','3','4','5','6','7','8','9','10'], "")

    out_dir = '/home/geoint/tri/hls_datacube'
    #########################
    filename = f'{out_dir}/{tile}-{year}-0426.hdf5'
    h = h5py.File(filename, 'w')

    for k, v in ts_dict.items():
        h.create_dataset(f"{k}_ts", data=v, compression="gzip", compression_opts=9)
    #     h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k][0], compression="gzip", compression_opts=9)
    print(f'finished the data processing')

    ## Test load h5py file
    print("Test load h5py file")

    with h5py.File(filename, "r") as file:
        ts_arr = file[f'{tile}_{year}_ts'][()]
        # mask_arr = file[f'Tappan19_WV02_20170126_{tile}_mask'][()]

        print(sorted(list(file.keys())))

    print(ts_arr.shape)