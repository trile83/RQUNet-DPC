import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from skimage import exposure
import re
import glob
import os
import h5py
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

def filtering_holes(mask_array, name):


    if 'Tappan14_WV02_20161230' in name:
        mask_array[mask_array==7]=2

    crop_array = mask_array.copy()

    crop_array[crop_array != 2] = 0
    crop_array[crop_array == 2] = 1

    crop_array_flt = ndimage.binary_fill_holes(
        crop_array,
        structure=np.ones((3,3))
    ).astype(int)

    new_array = crop_array_flt.copy()
    new_array[mask_array == 7] = 2

    del crop_array_flt
    del mask_array

    return np.squeeze(new_array)

def read_data(master_dir, tile='PEA', mask_dir_path='/home/geoint/tri/resampled_senegal_hls/filtered/'):

    fl_dir = sorted(glob.glob(f"{master_dir}/output-{tile}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    mask_dict={}
    unique_name = []

    # mask_dir_path = os.path.join(mask_dir_path, tile)

    mask_lst = sorted(glob.glob(f'{mask_dir_path}/*.tif'))

    for idx, file in enumerate(fl_dir):
        name = re.search(f'{tile}/(.*?).tif', file).group(1)[:22]
        year_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[:4]
        date_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[4:]

        # if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
        #     continue

        if name not in unique_name: # check if ts name is seen or not
            anchor_name = name

            unique_name.append(anchor_name)

            ts_dict[anchor_name] = {}
            # mask_dict[anchor_name] = []

        if name in file:

            img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

            # ts_dict[name].append(img_data)
            if int(date_hls) not in ts_dict.keys():
                ts_dict[name][int(date_hls)] = img_data

        for mask_fl in mask_lst:
            # print(mask_fl)
            if name in mask_fl:
                # print(name)
                mask_data = np.squeeze(rxr.open_rasterio(mask_fl, masked=False).values)
                if name in ts_dict.keys():
                    # mask_data = filtering_holes(mask_data, name)
                    mask_dict[name]= mask_data

    output_dict = {}

    # for key in ts_dict.keys():
    #     # print(key)
    #     output_dict[key] = np.stack(ts_dict[key], axis=0)

    # del ts_dict

    # return output_dict, mask_dict
    return ts_dict, mask_dict


def get_composite(ts_dict):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    
    out_lst = []

    key_lst = []
    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for idx, key in enumerate(sorted(ts_dict.keys())):

        if ts_dict[key].shape[0] < 13:
            print(key, ' ', ts_dict[key].shape)
        
        if int(key) > 0 and len(out_lst) == 0:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 30 and len(out_lst) == 1:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 60 and len(out_lst) == 2:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 80 and len(out_lst) == 3:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 114 and len(out_lst) == 4:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)

        elif int(key) > 149 and len(out_lst) == 5:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 239 and len(out_lst) == 6:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 279 and len(out_lst) == 7:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 300 and len(out_lst) == 8:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)
        elif int(key) > 340 and len(out_lst) == 9:
            print(key)
            out_lst.append(ts_dict[key])
            key_lst.append(key)


    out_array = np.stack(out_lst, axis=0)
    del ts_dict

    # print(out_array.shape)

    return out_array, key_lst

def plot_timeseries(
    train_ts_set,
    mask_arr,
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
    # fig, axes = plt.subplots(height, width, layout='constrained')

    # fig = plt.figure(figsize=(20,20))
    fig = plt.figure()
    for idx in range(1,height*width):
        # plt.subplot(height,width,idx)
        ax = fig.add_subplot(height, width, idx)
        if idx < 11:
            # plt.title(f'Day {dates[idx-1]}')

            ax.set_title(f'Day {dates[idx-1]}', x=0.5, y=0.95, size=8, color='black')
            # image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            image = np.transpose(train_ts_set[(idx-1),:3,:,:], (1,2,0))
            image= rescale_image(xr.where(image > -9000, image, -1000))
            plt.axis('off')
            plt.imshow(rescale_truncate(get_rgb(image)))
        else:
            print('mask shape: ', mask_arr.shape)
            if len(mask_arr.shape) > 2:
                image = mask_arr[0]

            ax.set_title('Label', x=0.5, y=0.95, size=8, color='black')
            # image = mask_arr
            plt.axis('off')
            plt.imshow(image, cmap = colormap, vmin=0, vmax=len(colors))

    plt.subplots_adjust(wspace=0.025,hspace=0.15)

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_rgb(data):

    out_image = data.copy()

    out_image[:,:,0] = data[:,:,2]
    out_image[:,:,2] = data[:,:,0]

    return out_image

if __name__ == "__main__":

    tiles=['PGA']
    master_dir = "/home/geoint/tri/match-hls-sen/"

    for tile in tiles:
        ts_dict, mask_dict = read_data(master_dir, tile)

        output_dict = {}

        for key in ts_dict.keys():
            print(key)
            # print(len(ts_dict[key]))
            print(ts_dict[key].keys())

            ts_arr, date_lst = get_composite(ts_dict[key])

            if key not in output_dict.keys():
                output_dict[key] = ts_arr

            print('total time series length: ', len(date_lst))

            mask_arr = mask_dict[key]

            if len(ts_dict[key]) < 10:
                print(key, ' not enough frames in time series')
                continue

            plot_timeseries(ts_arr, mask_arr, key, date_lst, tile)

        out_dir = '/home/geoint/tri/hls_datacube'

    #     ########################
        # if not os.path.isfile(f'{out_dir}/hls-{tile}-0906.hdf5'):

        #     h = h5py.File(f'{out_dir}/hls-{tile}-0906.hdf5', 'w')

        #     for k, v in output_dict.items():
        #         h.create_dataset(f"{k}_{tile}_ts", data=v, compression="gzip", compression_opts=9)
        #         h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k], compression="gzip", compression_opts=9)
        #     print(f'finished the data processing')
        # else:
        #     print('Datacube file already exist!')


    ## Test load h5py file

    out_dir = '/home/geoint/tri/hls_datacube'
    print("Test load h5py file")
    filename= f'{out_dir}/old/planet-etz-2021-0426.hdf5'
    # filename= f'{out_dir}/hls-PEV.hdf5'

    with h5py.File(filename, "r") as file:

        key =sorted(list(file.keys()))

        ts_arr = file[key[0]][()]
        mask_arr = file[key[0]][()]

        print(sorted(list(file.keys())))

    print(ts_arr.shape)
    print(mask_arr.shape)

    plot_timeseries(ts_arr[:10], mask_arr[0,:3], key, ['1','2','3','4','5','6','7','8','9','10','11','12'], 'ETZ')