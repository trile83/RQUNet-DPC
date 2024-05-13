import numpy as np
import tifffile
import rioxarray as rxr
import re
import glob
import os
import h5py

def read_data(master_dir, tile='PEA', year_set='2016', mask_dir_path=''):

    fl_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(fl_dir))
    if tile == 'PEV':
        hls_cloud_dir = '/home/geoint/tri/hls_cas_16-20_cloud/'
    elif tile == 'PFV':
        hls_cloud_dir = '/home/geoint/tri/hls_cas_16-20_cloud/'
    elif tile == 'PEA':
        hls_cloud_dir = '/home/geoint/tri/hls_etz_16-18_cloud/'
    elif tile == 'PFA':
        hls_cloud_dir = '/home/geoint/tri/hls_etz_16-18_cloud/'

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
            ts_dict[name][year_hls] = []

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
                ts_dict[name][year_hls].append(img_data)
            except:
                print('Cannot read this image: ', fullname)

        else:
            continue

        if mask_dir_path != "":
            for mask_fl in mask_lst:
                if name in mask_fl:
                    mask_data = np.squeeze(rxr.open_rasterio(mask_fl, masked=False).values)
                    mask_dict[name].append(mask_data)

    print("number of image after processing: ", count)

    output_dict = {}

    for tilename in ts_dict.keys():
        for year in ts_dict[tilename].keys():
            outname = f'{tilename}_{year}'
            if len(ts_dict[tilename][year]) > 0:
                print('time series year with at least 1 scene: ', year)
                output_dict[outname] = np.stack(ts_dict[tilename][year], axis=0)

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
    ts_dict, mask_dict = read_data(master_dir, tile, year)

    for key in ts_dict.keys():
        print(key)
        print(ts_dict[key].shape)
        # print(mask_dict[key][0].shape)

    out_dir = '/home/geoint/tri/hls_datacube'
    #########################
    filename = f'{out_dir}/hls-{tile}-{year}-0318.hdf5'
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

    # print(ts_arr.shape)
    # print(mask_arr.shape)


    ## Create stepping composite for long timeseries

    # out_array = get_composite(ts_arr)