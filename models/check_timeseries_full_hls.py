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
import rasterio as rio

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

def read_data(
    master_dir: str='',
    tile: str='PEA',
    year: int=2016,
    ):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    unique_year = []

    for idx, file in enumerate(fl_dir):
        name = re.search(f'{master_dir}/(.*?).tif', file).group(1)

        if tile not in name:
            continue
        year_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[:4]
        date_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[4:]

        cloud_fl = name + '.Fmask.tif'

        if 'V' in tile:
            cloud_dir = "/home/geoint/tri/hls_cas_16-22_cloud"
        elif 'A' in tile:
            cloud_dir = "/home/geoint/tri/hls_etz_16-22_cloud"
        cloud_path = os.path.join(cloud_dir, cloud_fl)

        # if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
        #     continue

    
        print('year hls: ', year_hls)
        print('date hls: ', date_hls)

        if int(year_hls) == year:

            if year_hls not in unique_year: # check if ts name is seen or not

                unique_year.append(year_hls)

                ts_dict[year_hls] = {}

            try:

                img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

                cloud_mask = np.squeeze(rxr.open_rasterio(cloud_path, mask=False).values)
                temp_arr = cloud_mask % 16
                count_cloud = np.count_nonzero(temp_arr)

                print('count cloud pixels: ', count_cloud)

                if count_cloud > 1800000:
                    continue

                # print(img_data.shape)
            except:
                print('Image Error!')
                continue

            # ts_dict[name].append(img_data)
            if int(date_hls) not in ts_dict[year_hls].keys():
                ts_dict[year_hls][int(date_hls)] = img_data

    # output_dict = {}

    # for key in ts_dict.keys():
    #     print(key)
    #     output_dict[key] = np.stack(ts_dict[key], axis=0)

    # del ts_dict

    # return output_dict, mask_dict
    return ts_dict

def get_composite(ts_dict):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    
    out_lst = []

    key_lst = []
    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for idx, key in enumerate(sorted(ts_dict.keys())):
        # print(ts_dict[key].shape)
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

    print(len(out_lst))


    out_array = np.stack(out_lst, axis=0)
    del ts_dict

    # print(out_array.shape)

    return out_array, key_lst

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
            image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            image= rescale_image(xr.where(image > -9000, image, -1000))
            plt.imshow(rescale_truncate(image))
        
        # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    tiles=['PEA']
    year = 2016

    for tile in tiles:
        if 'V' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_etz_2"
        elif 'A' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015"

        # print(master_dir)
        ts_dict = read_data(master_dir, tile, year)

        output_dict = {}

        for key in ts_dict.keys():
            print(key)
            # print(len(ts_dict[key]))
            print(ts_dict[key].keys())

            ts_arr, date_lst = get_composite(ts_dict[key])

            if key not in output_dict.keys():
                output_dict[key] = ts_arr

            print('total time series length: ', len(date_lst))

            # mask_arr = mask_dict[key]

            if len(ts_dict[key]) < 10:
                print(key, ' not enough frames in time series')
                continue

            plot_timeseries(ts_arr, key, date_lst, tile)

        out_dir = '/home/geoint/tri/hls_datacube'

        # ########################
        if not os.path.isfile(f'{out_dir}/hls-{tile}-{year}-full.hdf5'):

            h = h5py.File(f'{out_dir}/hls-{tile}-{year}-full.hdf5', 'w')

            for k, v in output_dict.items():
                h.create_dataset(f"{tile}_{k}_ts", data=v, compression="gzip", compression_opts=9)

            print(f'finished the data processing')



    ## Test load h5py file
    out_dir = '/home/geoint/tri/hls_datacube'
    print("Test load h5py file")
    filename= f'{out_dir}/hls-{tile}-{year}-full.hdf5'

    with h5py.File(filename, "r") as file:
        ts_arr = file[f'{tile}_{year}_ts'][()]

        print(ts_arr.shape)

        print(sorted(list(file.keys())))

        # plot_timeseries(ts_arr, ts_arr[0], 'PEV-large', 'PEV')

    # print(ts_arr.shape)
    # print(mask_arr.shape)


    ## Create stepping composite for long timeseries

    # out_array = get_composite(ts_arr)
