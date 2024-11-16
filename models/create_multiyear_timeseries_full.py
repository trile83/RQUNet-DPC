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

def read_data(
    master_dir: str='',
    tile: str='PEA',
    epoch: int=2018,
    ):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    unique_year = []

    if "-" in tile:
        stile = tile[:3]
    else:
        stile = tile

    for idx, file in enumerate(fl_dir):
        name = re.search(f'{master_dir}/(.*?).tif', file).group(1)

        if stile not in name:
            continue
        year_hls = re.search(f'T28{stile}.(.*?)T', file).group(1)[:4]
        date_hls = re.search(f'T28{stile}.(.*?)T', file).group(1)[4:]

        cloud_fl = name + '.Fmask.tif'

        if tile == 'PFV-R':
            cloud_dir = "/home/geoint/tri/hls_pfv_r_cloud_mask"
        if tile == 'PFV-L':
            cloud_dir = "/home/geoint/tri/hls_pfv_l_cloud_mask"
        elif 'EV' in tile or 'FV' in tile:
            cloud_dir = "/home/geoint/tri/hls_cas_16-22_cloud"
        elif 'CV' in tile:
            cloud_dir = "/home/geoint/tri/hls_wcas_16-22_cloud"
        elif tile == 'PFA-R':
            cloud_dir = "/home/geoint/tri/hls_pfa_R_cloud_mask"
        elif 'DA' in tile:
            cloud_dir = "/home/geoint/tri/hls_pda_cloud_mask"
        elif 'EB' in tile:
            cloud_dir = "/home/geoint/tri/hls_peb_cloud_mask"
        elif 'DB' in tile:
            cloud_dir = "/home/geoint/tri/hls_pdb_cloud_mask"
        elif 'GA' in tile:
            cloud_dir = "/home/geoint/tri/hls_pga_cloud_mask"
        elif tile == 'PEA':
            cloud_dir = "/home/geoint/tri/hls_etz_16-22_cloud"
        elif tile == 'PFA':
            cloud_dir = "/home/geoint/tri/hls_etz_16-22_cloud"

        cloud_path = os.path.join(cloud_dir, cloud_fl)

        # if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
        #     continue

        if int(year_hls) > (int(epoch)-4) and int(year_hls) < (int(epoch)+1):

            print('year hls: ', year_hls)
            print('date hls: ', date_hls)

            identifier = str(year_hls) + str(date_hls)

            if epoch not in unique_year: # check if ts name is seen or not

                unique_year.append(epoch)

                ts_dict[epoch] = {}

            try:
                img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)
            except:
                print('Multispec Image Error!')
                continue

            try:

                # img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

                cloud_mask = np.squeeze(rxr.open_rasterio(cloud_path, mask=False).values)
                temp_arr = cloud_mask % 16
                count_cloud = np.count_nonzero(temp_arr)

                print('count cloud pixels: ', count_cloud)

                if 'F' in tile:
                    cloud_thresh = 7000000
                elif 'C' in tile:
                    cloud_thresh = 5000000
                else:
                    cloud_thresh = 1800000

                if count_cloud > cloud_thresh:
                    continue

                # print(img_data.shape)
            except:
                print('Cloud Mask Image Error!')
                continue

            # print('Image shape: ', img_data.shape)

            # ts_dict[name].append(img_data)
            if identifier not in ts_dict[epoch].keys():
                ts_dict[epoch][identifier] = img_data

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

            ax.set_title(f'Day {dates[idx-1]}', x=0.5, y=0.95, size=8, color='black')
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

    # plt.subplots_adjust(wspace=0.025,hspace=0.15)

    plt.savefig(f"/home/geoint/tri/match-hls-sen/test-im/{name}-{tile}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_rgb(data):

    out_image = data.copy()

    out_image[:,:,0] = data[:,:,2]
    out_image[:,:,2] = data[:,:,0]

    return out_image

if __name__ == "__main__":

    tiles=['PFV-L']
    epoch=2019

    for tile in tiles:
        if 'FV-R' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pfv_r"
        elif 'FV-L' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_cas_PFV_L"
        elif 'EV' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_cas2015"
        elif 'CV' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_wcas_2"
        elif 'FA-L' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pfa_l"
        elif tile == 'PFA-R':
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pfa_r"
        elif 'EA' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015"
        elif 'GA' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pga"
        elif 'DA' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pda"
        elif 'EB' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_peb"
        elif 'DB' in tile:
            master_dir = "/home/geoint/PycharmProjects/tensorflow/out_hls_pdb"

        print('master dir: ', master_dir)
        

        # print(master_dir)
        ts_dict = read_data(master_dir, tile, epoch)

        output_dict = {}

        for key in ts_dict.keys():
            # print(key)
            # print(len(ts_dict[key]))
            # print(ts_dict[key].keys())

            ts_arr, date_lst = get_composite(ts_dict[key])

            if key not in output_dict.keys():
                output_dict[key] = ts_arr

            print('total time series length: ', len(date_lst))

            # mask_arr = mask_dict[key]

            if len(ts_dict[key]) < 16:
                print(key, ' not enough frames in time series')
                continue

            plot_timeseries(ts_arr, key, date_lst, tile)

        out_dir = '/home/geoint/tri/hls_datacube'

        del ts_arr, key, date_lst, ts_dict

    #     ########################
        if not os.path.isfile(f'{out_dir}/hls-{tile}-full-epoch{epoch}.hdf5'):

            h = h5py.File(f'{out_dir}/hls-{tile}-full-epoch{epoch}.hdf5', 'w')

            for k, v in output_dict.items():
                h.create_dataset(f"{k}_{tile}_ts", data=v, compression="gzip", compression_opts=9)

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