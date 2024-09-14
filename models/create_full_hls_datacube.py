import numpy as np
import tifffile
import rioxarray as rxr
import xarray as xr
import re
import glob
import os
import h5py
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

def read_data(master_dir, tile='PEA', year_set='2016', mask_dir_path=''):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))
    if tile == 'PEV':
        hls_cloud_dir = '/home/geoint/tri/hls_cas_16-22_cloud/'
    elif tile == 'PFV':
        hls_cloud_dir = '/home/geoint/tri/hls_cas_16-22_cloud/'
    elif tile == 'PEA':
        hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud/'
    elif tile == 'PFA':
        hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud/'

    ts_dict = {}
    mask_dict = {}
    unique_name = []

    if mask_dir_path != "":
        mask_dir_path = os.path.join(mask_dir_path, tile)
        mask_lst = sorted(glob.glob(f'{mask_dir_path}/*.tif'))

    count=0
    for idx, file in enumerate(fl_dir):
        if tile not in file:
            continue
        name = re.search(f'T28(.*?).20', file).group(1)
        fullname = re.search(r'2015/(.*?).tif', file).group(1)
        year_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[:4]
        date_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[4:]

        cloudfile = f'{hls_cloud_dir}/{fullname}.Fmask.tif'
        print(f"cloudfile: {cloudfile}")
        year_hls = re.search(f'{tile}.(.*?)T', fullname).group(1)[:4]

        if str(year_hls) == "2015":
            print('Exclude the 2015 images', fullname)
            continue

        if year_hls != year_set:
            continue

        print(f"name : {name} at year {year_hls}")
        if name not in unique_name: # check if ts name is seen or not
            anchor_name = name
            unique_name.append(anchor_name)
            ts_dict[anchor_name] = {}

            if mask_dir_path != "":
                mask_dict[anchor_name] = []

        if year_hls not in ts_dict[name].keys():
            ts_dict[name][year_hls] = {}

        if name in file:
            try:
                #img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)
                img_data = tifffile.imread(file)
                print(f"image shape: {img_data.shape}")

                #cloud_mask = np.squeeze(rxr.open_rasterio(cloudfile, masked=False).values)
                cloud_mask = tifffile.imread(cloudfile)

                #cloud_mask = cloud_mask.reshape((cloud_mask.shape[1], cloud_mask.shape[2], cloud_mask.shape[0]))
                print(f"mask shape: {cloud_mask.shape}")
                temp_arr = cloud_mask % 16
                count_cloud = np.count_nonzero(temp_arr)

                print('count cloud pixels: ', count_cloud)

                if count_cloud > 5000000:
                    continue

                count+=1

                if int(date_hls) not in ts_dict.keys():
                    # ts_dict[name][int(date_hls)] = img_data
                    ts_dict[name][year_hls][int(date_hls)] = img_data

            except:
                print('Cannot read this image: ', fullname)

    print("number of image after processing: ", count)

    return ts_dict


def get_composite(ts_dict):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    
    out_lst = []

    # for i in ts_dict.keys():
    #     print(i)

    key_lst = []
    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for idx, key in enumerate(sorted(ts_dict.keys())):
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
    for idx in range(1,height*width):
        plt.subplot(height,width,idx)
        if idx < 11:
            plt.title(f'Day {dates[idx-1]}')
            image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            image= rescale_image(xr.where(image > -9000, image, -1000))
            plt.imshow(rescale_truncate(image))
        # else:
        #     plt.title(f'Label')
        #     image = mask_arr
        #     plt.imshow(image, cmap = colormap, vmin=0, vmax=len(colors))
        # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    tile='PFA'
    year="2016"
    if tile =="PEV":
        master_dir = "/media/geoint/Backup Plus/out_hls_cas2015/"
    elif tile == "PFV":
        master_dir = "/media/geoint/Backup Plus/out_hls_cas2015/"
    elif tile == "PEA":
        master_dir = "/media/geoint/Backup Plus/out_hls_etz2015/"
    elif tile == "PFA":
        master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015/"
    
    ts_dict = read_data(master_dir, tile, year)

    for key in ts_dict.keys():
        name = key
        print(name)
        # print(ts_dict[name].keys())
        # print(mask_dict[key][0].shape)
        for year in ts_dict[key]:
            print(year)
            ts_arr, date_lst = get_composite(ts_dict[name][year])
            for date in ts_dict[name][year].keys():
                plot_timeseries(ts_arr, name, date_lst, tile)

    out_dir = '/home/geoint/tri/hls_datacube'
    #########################
    filename = f'{out_dir}/hls-{tile}-{year}-0318.hdf5'
    h = h5py.File(filename, 'w')

    # for k, v in ts_dict.items():
    #     h.create_dataset(f"{k}_ts", data=v, compression="gzip", compression_opts=9)
    # #     h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k][0], compression="gzip", compression_opts=9)
    # print(f'finished the data processing')

    ## Test load h5py file
    # print("Test load h5py file")

    # with h5py.File(filename, "r") as file:
    #     ts_arr = file[f'{tile}_{year}_ts'][()]
    #     # mask_arr = file[f'Tappan19_WV02_20170126_{tile}_mask'][()]

    #     print(sorted(list(file.keys())))

    # print(ts_arr.shape)
    # print(mask_arr.shape)


    ## Create stepping composite for long timeseries

    # out_array = get_composite(ts_arr)