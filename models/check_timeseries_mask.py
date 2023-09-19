import numpy as np
import rasterio as rio
import rioxarray as rxr
import re
import glob
import os
import h5py

def read_data(master_dir, tile='PEA', mask_dir_path='/home/geoint/tri/resampled_senegal_hls/trimmed/'):

    fl_dir = sorted(glob.glob(f"{master_dir}/output-{tile}/*.tif"))
    print("total file: ", len(fl_dir))

    ts_dict = {}
    mask_dict={}
    unique_name = []

    mask_dir_path = os.path.join(mask_dir_path, tile)

    mask_lst = sorted(glob.glob(f'{mask_dir_path}/*.tif'))

    for idx, file in enumerate(fl_dir):
        name = re.search(f'{tile}/(.*?).tif', file).group(1)[-22:]
        if name == 'Tappan18_WV03_20160307' or name =='Tappan19_WV02_20180119' or name =='Tappan19_WV02_20180119' or name =='Tappan20_WV02_20130430':
            continue
        if name not in unique_name: # check if ts name is seen or not
            anchor_name = name

            unique_name.append(anchor_name)

            ts_dict[anchor_name] = []
            mask_dict[anchor_name] = []

        if name in file:

            img_data = np.squeeze(rxr.open_rasterio(file, masked=False).values)

            ts_dict[name].append(img_data)

        for mask_fl in mask_lst:
            if name in mask_fl:
                mask_data = np.squeeze(rxr.open_rasterio(mask_fl, masked=False).values)
                mask_dict[name].append(mask_data)

    output_dict = {}

    for key in ts_dict.keys():
        output_dict[key] = np.stack(ts_dict[key], axis=0)

    del ts_dict

    return output_dict, mask_dict


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

if __name__ == "__main__":

    tile='PEA'
    master_dir = "/home/geoint/tri/match-hls-sen/"
    ts_dict, mask_dict = read_data(master_dir, tile)

    for key in ts_dict.keys():
        print(key)
        print(ts_dict[key].shape)
        print(mask_dict[key][0].shape)

    out_dir = '/home/geoint/tri/hls_datacube'
    #########################
    # h = h5py.File(f'{out_dir}/hls-etz-{tile}-0908.hdf5', 'w')

    # for k, v in ts_dict.items():
    #     h.create_dataset(f"{k}_{tile}_ts", data=v, compression="gzip", compression_opts=9)
    #     h.create_dataset(f"{k}_{tile}_mask", data=mask_dict[k][0], compression="gzip", compression_opts=9)
    # print(f'finished the data processing')


    ## Test load h5py file
    filename= f'{out_dir}/hls-etz-{tile}-0908.hdf5'

    with h5py.File(filename, "r") as file:
        ts_arr = file[f'Tappan19_WV02_20170126_{tile}_ts'][()]
        mask_arr = file[f'Tappan19_WV02_20170126_{tile}_mask'][()]

    print(ts_arr.shape)
    print(mask_arr.shape)


    ## Create stepping composite for long timeseries

    out_array = get_composite(ts_arr)
