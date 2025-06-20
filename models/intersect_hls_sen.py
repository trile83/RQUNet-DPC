import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.crs import CRS
import glob
import re
import os

# the first one is your raster on the right
# and the second one your red raster
# resamp_dir = '/home/geoint/tri/resampled_senegal_hls/mode' # used for inital cut of raster

tile='PFV'

resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}' # directory after trimming no-data border of cut raster

if 'EA' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015'
    hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud'
if 'GA' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015'
    hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud'
elif 'EV' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_cas2015'
    hls_cloud_dir = '/home/geoint/tri/hls_cas_16-22_cloud'
elif 'FV' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_cas_PFV_L'
    hls_cloud_dir = '/home/geoint/tri/hls_pfv_l_cloud_mask'
elif 'CV' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_wcas_1'
    hls_cloud_dir = '/home/geoint/tri/hls_wcas_16-22_cloud'
elif 'DB' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_pdb'
    hls_cloud_dir = '/home/geoint/tri/hls_pdb_cloud_mask'
elif 'DA' in tile:
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_pda'
    hls_cloud_dir = '/home/geoint/tri/hls_pda_cloud_mask'


resamp_lst = sorted(glob.glob(resamp_dir+'/*.tif'))
hls_lst = sorted(glob.glob(hls_dir+'/*.tif'))
hls_cloud_lst = sorted(glob.glob(hls_cloud_dir+'/*Fmask.tif'))

# print(resamp_lst)
# print(hls_lst)

for resamp_fl in resamp_lst:
    year_resamp = re.search(r'WV(.*?).tif', resamp_fl).group(1)[3:7]
    # print("Year WV resamp: ", year_resamp)
    print('Resampled File: ', resamp_fl)
    for hls_fl in hls_lst:

        name_hls = re.search(f'{hls_dir}/(.*?).v2.0.tif', hls_fl).group(1)
        year_hls = re.search(f'T28{tile}.(.*?)T', hls_fl).group(1)[:4]
        date_hls = re.search(f'T28{tile}.(.*?)T', hls_fl).group(1)[4:]
        # print("Year HLS: ", year_hls)
        # print("Date HLS: ", date_hls)

        cloud_fl = f'{hls_cloud_dir}/{name_hls}.v2.0.Fmask.tif'

        if tile not in hls_fl:
            continue

        # hls_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls_cas2015/HLS.S30.T28PFV.2017325T112351.v2.0.tif'

        ## match year of WV and year of HLS data
        # if year_resamp == year_hls:

        try:

            with rasterio.open(resamp_fl) as src, \
                    rasterio.open(hls_fl) as src_to_crop, \
                    rasterio.open(cloud_fl) as src_cloud_to_crop:
                
                src_affine = src.meta.get("transform")

                # name_resamp = re.search(r'/resampled_senegal_hls/mode/(.*?)_M1BS', resamp_fl).group(1)
                name_resamp = re.search(f'/resampled_senegal_hls/trimmed/{tile}/(.*?).tif', resamp_fl).group(1)
                
                # Read the first band of the "mask" raster
                band = src.read(1)
                # Use the same value on each pixel with data
                # in order to speedup the vectorization
                band[np.where(band!=src.nodata)] = 1

                geoms = []
                for geometry, raster_value in features.shapes(band, transform=src_affine):
                    # get the shape of the part of the raster
                    # not containing "nodata"
                    if raster_value == 1:
                        geoms.append(geometry)

                # crop the second raster using the
                # previously computed shapes
                try:
                    out_img, out_transform = mask(
                        dataset=src_to_crop,
                        shapes=geoms,
                        crop=True,
                    )

                    cloud_mask, out_transform_cloud = mask(
                        dataset=src_cloud_to_crop,
                        shapes=geoms,
                        crop=True,
                    )
                except:
                    continue

                # print(out_img.shape)

                ## check if cloud is less than 3000 pixels
                temp_arr = cloud_mask % 16
                count_cloud = np.count_nonzero(temp_arr)

                if count_cloud < 5000:

                    # save the result
                    # (don't forget to set the appropriate metadata)
                    # if cloud:
                    #     out_fl_name = f'/home/geoint/tri/match-hls-sen/output-{tile}-cloud/{name_hls}-{name_resamp}-cloud.tif'
                    # else:
                    #     # out_fl_name = f'/home/geoint/tri/match-hls-sen/output-PEV/{name_hls}-{name_resamp}.tif'


                    if not os.path.isdir(f'/home/geoint/tri/match-hls-sen/output-{tile}/'):
                        os.mkdir(f'/home/geoint/tri/match-hls-sen/output-{tile}/')

                    out_fl_name = f'/home/geoint/tri/match-hls-sen/output-{tile}/{name_resamp}-{name_hls}.tif'


                    if os.path.isfile(out_fl_name):
                        continue

                    with rasterio.open(
                        out_fl_name,
                        'w',
                        driver='GTiff',
                        height=out_img.shape[1],
                        width=out_img.shape[2],
                        count=src_to_crop.count,
                        dtype=out_img.dtype,
                        transform=out_transform,
                        crs=CRS.from_epsg(32628),
                    ) as dst:
                        dst.write(out_img)
        except Exception as error:
            print(error)
            # print("Year HLS: ", year_hls)
            print("Date HLS: ", date_hls)
            continue
