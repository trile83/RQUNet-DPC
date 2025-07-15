import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from skimage import exposure
import re
import glob
import os
import h5py
import csv
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
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

def filtering_holes(mask_array):

    # print(mask_array.shape)
    # print(np.unique(mask_array))

    # if 'Tappan14_WV02_20161230' in name:
    #     mask_array[mask_array==7]=2

    crop_array = mask_array.copy()

    crop_array[crop_array > 1] = 0
    # crop_array[crop_array == 2] = 1

    crop_array_flt = ndimage.binary_fill_holes(
        crop_array,
        structure=np.ones((3,3))
    ).astype(int)

    new_array = crop_array_flt.copy()
    new_array[mask_array == 7] = 2

    del crop_array_flt
    del mask_array

    return np.squeeze(new_array)

def read_data(
    master_dir: str='',
    tile: str='Tappan01',
    epoch: int=2019,
    ):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))

    if "-" in tile:
        stile = tile[:8]
    else:
        stile = tile

    date_lst = []
    out_lst = []

    for file in fl_dir:

        if tile == 'Tappan01':
            date = re.search(r'WV02_(.*?)_T28', file).group(1)
        elif tile == 'PEV':
            date = re.search(r'T28PEV_(.*?)T11', file).group(1)
        elif tile == 'PFA-L30':
            date = re.search(r'T28PFA.(.*?)T11', file).group(1)
        
        print(date)
        date_lst.append(date)

        img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

        out_lst.append(img_data)

    
    out_array = np.stack(out_lst, axis=0)
    print('out array shape: ', out_array.shape)


    return out_array, date_lst



def plot_timeseries(
    train_ts_set,
    name,
    dates
    ):

    height = 5
    width = 4

    classes = {
            0:'purple',
            1:'yellow',
            2:'black'
            }
    # convert all colors to a list
    colors = [classes[id] for id in classes.keys()]
    colormap = pltc.ListedColormap(colors)
    # fig, axes = plt.subplots(height, width, layout='constrained')

    # fig = plt.figure(figsize=(20,20))
    fig = plt.figure()
    for idx in range(1,height*width-3):
        # plt.subplot(height,width,idx)
        ax = fig.add_subplot(height, width, idx)
        if idx < 17:
            # plt.title(f'Day {dates[idx-1]}')

            ax.set_title(f'Day {dates[idx-1]}', x=0.5, y=0.95, size=8, color='black')
            # image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            # check_pixel_values(image)
            # image = get_rgb(image)
            image= rescale_image(xr.where(image > -9000, image, -200))
            plt.axis('off')
            plt.imshow(rescale_truncate(get_rgb(image)))

        # elif idx == 17:
        #     print('mask shape: ', mask_arr.shape)
        #     if len(mask_arr.shape) > 2:
        #         image = mask_arr[0]
        #     else:
        #         image = mask_arr

        #     ax.set_title('Label', x=0.5, y=0.95, size=8, color='black')
        #     # image = mask_arr
        #     plt.axis('off')
        #     plt.imshow(image, cmap = colormap, vmin=0, vmax=len(colors))

    # plt.subplots_adjust(wspace=0.025,hspace=0.15)

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def check_pixel_values(image):

    mask = np.where(image > -9999, True, False)

    minpix=np.min(image, initial=1.0, where=mask)
    maxpix=np.max(image, initial=1.0, where=mask)
    meanpix=np.mean(image, where=mask)
    stdpix=np.std(image, where=mask)

    # print("Min pixel value: ", np.min(image, initial=1.0, where=mask))
    # print("Max pixel value: ", np.max(image, initial=1.0, where=mask))

    # print("Mean pixel value: ", np.mean(image, where=mask))
    # print("Std pixel value: ", np.std(image, where=mask))

    return minpix,maxpix,meanpix,stdpix

def get_rgb(data):

    out_image = data.copy()

    out_image[:,:,0] = data[:,:,2]
    out_image[:,:,2] = data[:,:,0]

    return out_image

if __name__ == "__main__":

    tile = 'PFA-L30'
    epoch=2019

    #### SR data
    # master_dir = 'raster/ecas/tappan01/'

    #### S2 data
    # master_dir = 'data/ecas/tappan01/'

    #### large S2
    # master_dir = 'output/cut_s2/part_0_1'

    #### HLS.L30
    master_dir = '/home/geoint/PycharmProjects/tensorflow/out_l30_pfa'

    print('master dir: ', master_dir)
    
    # print(master_dir)

    out_array, date_lst = read_data(master_dir, tile, epoch)

    print("max pixel values: ", np.max(out_array))
    print("min pixel values: ", np.min(out_array))

    plot_timeseries(out_array, tile, date_lst)

    out_dir = '/home/geoint/tri/hls_datacube'

    ########################
    if not os.path.isfile(f'{out_dir}/{tile}-full-epoch{epoch}.hdf5'):

        h = h5py.File(f'{out_dir}/{tile}-full-epoch{epoch}.hdf5', 'w')

        # h.create_dataset(f"{tile}_ts", data=out_array, compression="gzip", compression_opts=9)

        ## large S2
        h.create_dataset(f"{tile}_ts", data=out_array, compression="gzip", compression_opts=9, chunks=(16,4,20,20))

        print(f'finished the data processing')
    else:
        print('Datacube file already exist!')


    ## Test load h5py file

    # out_dir = '/home/geoint/tri/hls_datacube'
    # print("Test load h5py file")
    # filename= f'{out_dir}/hls-PEV-full-epoch2019.hdf5'
    # # filename= f'{out_dir}/hls-PEV.hdf5'

    # with h5py.File(filename, "r") as file:

    #     key =sorted(list(file.keys()))

    #     ts_arr = file[key[0]][()]
    #     mask_arr = file[key[0]][()]

    #     print(sorted(list(file.keys())))

    # print(ts_arr.shape)
    # print(mask_arr.shape)

    # plot_timeseries(ts_arr[:10], mask_arr[0,:3], key, ['1','2','3','4','5','6','7','8','9','10','11','12'], 'ETZ')