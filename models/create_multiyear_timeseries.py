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

def read_data(master_dir, epoch:int=2019, tile:str='PEA', mask_dir_path:str='/home/geoint/tri/resampled_senegal_hls/multiyear/'):

    fl_dir = sorted(glob.glob(f"{master_dir}/output-{tile}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    mask_dict={}
    unique_name = []

    # mask_dir_path = os.path.join(mask_dir_path, tile)

    for idx, file in enumerate(fl_dir):
        if "P" in tile:
            name = re.search(f'{tile}/(.*?).tif', file).group(1)[:22]
            shortname = name[:8]
            year_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[:4]
            date_hls = re.search(f'T28{tile}.(.*?)T', file).group(1)[4:]

            mask_lst = sorted(glob.glob(f'{mask_dir_path}/*.tif'))
        else:
            name = re.search(f'{tile}/(.*?).tif', file).group(1)[:22]
            year_hls = re.search(f'1105N-(.*?).tif', file).group(1)[2:]
            date_hls = re.search(f'1105N-(.*?).tif', file).group(1)[:2]
            shortname = name

            mask_lst = sorted(glob.glob(f'/home/geoint/tri/resampled_senegal_planet/*.tif'))

        # if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
        #     continue

        if tile == 'PEV' and shortname in ['Tappan02','Tappan04','Tappan17']:
            continue

        if tile == 'PFV' and shortname in ['Tappan01','Tappan05']:
            continue

        if shortname not in unique_name: # check if ts name is seen or not
            anchor_name = shortname

            unique_name.append(anchor_name)

            ts_dict[anchor_name] = {}
            # mask_dict[anchor_name] = []

        if shortname in file:

            # print('file: ', file)

            img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

            # print('img data shape: ', img_data.shape)

            # print('img_data max: ', np.max(img_data[:-1,:,:]))
            # print('img_data min: ', np.min(img_data[:-1,:,:]))

            # ts_dict[name].append(img_data)
            if int(year_hls) > (int(epoch)-4) and int(year_hls) < (int(epoch)+1):

                identifier = str(year_hls) + str(date_hls)

                # print('identifier: ', identifier)

                # if int(date_hls) not in ts_dict.keys():
                if 'P' in tile:
                    ts_dict[shortname][identifier] = img_data
                else:
                    ts_dict[shortname][identifier] = img_data[:-1,:,:]

        for mask_fl in mask_lst:
            # print(mask_fl)
            if shortname in mask_fl:
                # print(name)
                mask_data = np.squeeze(rxr.open_rasterio(mask_fl, masked=False).values)
                if shortname in ts_dict.keys():
                    
                    if "P" in tile:
                        mask_dict[shortname]= filtering_holes(mask_data)
                        # mask_dict[shortname]= mask_data
                    else:
                        mask_dict[shortname] = filtering_holes(mask_data)

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
    year_lst = [2016,2017,2018,2019]
    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames

    for idx, year in enumerate(year_lst):
        for key in sorted(ts_dict.keys()):

            ##### suitable for PEA, PEV tile
            im_date = key[4:]
            im_year = key[:4]

            # print('im date: ', im_date)

            # if ts_dict[key].shape[0] < 13:
            #     print(key, ' ', ts_dict[key].shape)

            # print('year: ', year)

            if int(im_year) == int(year):
            
                if int(im_date) > 0 and (len(out_lst)-idx*4) == 0:
                    # print("date: ", im_date)
                    out_lst.append(ts_dict[key])
                    key_lst.append(key)
                elif int(im_date) > 149 and (len(out_lst)-idx*4) == 1:
                    # print("date: ", im_date)
                    out_lst.append(ts_dict[key])
                    key_lst.append(key)
                elif int(im_date) > 240 and (len(out_lst)-idx*4) == 2:
                    # print("date: ", im_date)
                    out_lst.append(ts_dict[key])
                    key_lst.append(key)
                elif int(im_date) > 330 and (len(out_lst)-idx*4) == 3:
                    # print("date: ", im_date)
                    out_lst.append(ts_dict[key])
                    key_lst.append(key)


    try:
        out_array = np.stack(out_lst, axis=0)
        print('out array shape: ', out_array.shape)
        del ts_dict

        return out_array, key_lst
    except:
        print("Stacking Error!")
        del ts_dict



def plot_timeseries(
    train_ts_set,
    mask_arr,
    name,
    dates,
    tile
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

            ax.set_title(f'Day {dates[idx-1]}', x=0.5, y=0.90, size=5, color='black')
            # image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            image = np.transpose(train_ts_set[(idx-1),1:4,:,:], (1,2,0))
            # image = get_rgb(image)
            image= rescale_image(xr.where(image > -9000, image, -1000))
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

    plt.subplots_adjust(wspace=0.025,hspace=0.15)

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_rgb(data):

    out_image = data.copy()

    out_image[:,:,0] = data[:,:,2]
    out_image[:,:,2] = data[:,:,0]

    return out_image

if __name__ == "__main__":

    tiles=['PEV']
    epoch=2019
    master_dir = "/home/geoint/tri/match-hls-sen/"

    # for tile in tiles:
    #     ts_dict, mask_dict = read_data(master_dir,epoch,tile)

    #     output_dict = {}

    #     for key in ts_dict.keys():
    #         print('key: ', key)
    #         # print(len(ts_dict[key]))
    #         print(ts_dict[key].keys())

    #         if "P" in tile:
    #             ts_arr, date_lst = get_composite(ts_dict[key])
    #         else:
    #             out_lst = [ts_dict[key][a] for a in ts_dict[key].keys()]
    #             ts_arr = np.stack(out_lst, axis=0)
    #             print(ts_arr.shape)
    #             date_lst = list(ts_dict[key].keys())

    #         if key not in output_dict.keys():
    #             output_dict[key] = ts_arr

    #         print('total time series length: ', len(date_lst))

    #         mask_arr = mask_dict[key]

    #         if len(ts_dict[key]) < 16:
    #             print(key, ' not enough frames in time series')
    #             continue

    #         plot_timeseries(ts_arr, mask_arr, key, date_lst, tile)

    #     out_dir = '/home/geoint/tri/hls_datacube'

    #     ########################
        # if not os.path.isfile(f'{out_dir}/hls-{tile}-epoch{epoch}.hdf5'):

        #     h = h5py.File(f'{out_dir}/hls-{tile}-epoch{epoch}.hdf5', 'w')

        #     for k, v in output_dict.items():
        #         h.create_dataset(f"{k}_{tile}_ts", data=v, compression="gzip", compression_opts=9)
        #         h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k], compression="gzip", compression_opts=9)
        #     print(f'finished the data processing')
        # else:
        #     print('Datacube file already exist!')


    ## Test load h5py file

    out_dir = '/home/geoint/tri/hls_datacube'
    print("Test load h5py file")
    filename= f'{out_dir}/hls-PFV-epoch2019.hdf5'
    # filename= f'{out_dir}/hls-PEV.hdf5'

    with h5py.File(filename, "r") as file:

        key =sorted(list(file.keys()))

        ts_arr = file[key[0]][()]
        mask_arr = file[key[0]][()]

        print(sorted(list(file.keys())))

    print(ts_arr.shape)
    print(mask_arr.shape)

    # plot_timeseries(ts_arr[:10], mask_arr[0,:3], key, ['1','2','3','4','5','6','7','8','9','10','11','12'], 'ETZ')