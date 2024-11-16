import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
import glob
import re
import os

# the first one is your raster on the right
# and the second one your red raster
# resamp_dir = '/home/geoint/tri/resampled_senegal_hls/mode' # used for inital cut of raster

tile='PDA'

if tile in ['PEV', 'PFV']:
    resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/mode/eCAS/' # directory after trimming no-data border of cut raster
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_cas2015/'
    hls_cloud_dir = '/home/geoint/tri/hls_cas_16-22_cloud'

elif tile in ['PEA','PFA','PGA']:
    resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/mode/ETZ/' # directory after trimming no-data border of cut raster
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015/'
    hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud/'

elif tile in ['PCV']:
    resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/mode/wCAS/' # directory after trimming no-data border of cut raster
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_wcas/'
    hls_cloud_dir = '/home/geoint/tri/hls_wcas_16-22_cloud/'

elif tile in ['PDB']:
    resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/mode/ETZ/' # directory after trimming no-data border of cut raster
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_pdb/'
    hls_cloud_dir = '/home/geoint/tri/hls_pdb_cloud_mask/'

elif tile in ['PDA']:
    resamp_dir = f'/home/geoint/tri/resampled_senegal_hls/mode/ETZ/' # directory after trimming no-data border of cut raster
    hls_dir = '/home/geoint/PycharmProjects/tensorflow/out_hls_pda/'
    hls_cloud_dir = '/home/geoint/tri/hls_pda_cloud_mask/'

resamp_lst = sorted(glob.glob(resamp_dir+'/*.tif'))
hls_lst = sorted(glob.glob(hls_dir+'/*.tif'))
hls_cloud_lst = sorted(glob.glob(hls_cloud_dir+'/*Fmask.tif'))

# print(resamp_lst)
# print(hls_lst)

cloud = False
if cloud:
    hls_lst = hls_cloud_lst
else:
    hls_lst = hls_lst


for resamp_fl in resamp_lst:
    year_resamp = re.search(r'WV(.*?).tif', resamp_fl).group(1)[3:7]
    print(tile)
    print("Year WV resamp: ", year_resamp)
    count=0
    for idx, hls_fl in enumerate(hls_lst):

        if tile not in hls_fl:
            # print(hls_fl)
            # print('Aha')
            continue

        if count == 0:

            if cloud:
                name_hls = re.search(f'/{hls_cloud_dir}(.*?).v2.0.Fmask.tif', hls_fl).group(1)
            else:
                name_hls = re.search(f'{hls_dir}(.*?).v2.0.tif', hls_fl).group(1)
                year_hls = re.search(f'{tile[-1]}.(.*?)T', hls_fl).group(1)[:4]
                
            cloud_fl = f'{hls_cloud_dir}/{name_hls}.v2.0.Fmask.tif'

            if year_resamp == year_hls:

                print("Year HLS: ", year_hls)

                with rasterio.open(resamp_fl) as src, \
                        rasterio.open(hls_fl) as src_to_crop, \
                        rasterio.open(cloud_fl) as src_cloud_to_crop:
                    
                    src_affine = src.meta.get("transform")

                    if tile in ['PEV','PFV']:
                        name_resamp = re.search(r'/resampled_senegal_hls/mode/eCAS/(.*?)_M1BS', resamp_fl).group(1)
                    elif tile in ['PEA','PFA','PGA','PDB','PDA']:
                        name_resamp = re.search(r'/resampled_senegal_hls/mode/ETZ/(.*?)_M1BS', resamp_fl).group(1)
                    elif tile in ['PCV']:
                        name_resamp = re.search(r'/resampled_senegal_hls/mode/wCAS/(.*?)_M1BS', resamp_fl).group(1)
                    elif tile in ['tile13']:
                        name_resamp = re.search(r'/resampled_senegal_planet/(.*?)_M1BS', resamp_fl).group(1)


                    print("Tappan: ", name_resamp)

                    # if tile == 'PEV':
                    #     name_resamp = re.search(r'/resampled_senegal_hls/trimmed/PEV/(.*?).tif', resamp_fl).group(1)
                    # elif tile == 'PFV':
                    #     name_resamp = re.search(r'/resampled_senegal_hls/trimmed/PFV/(.*?).tif', resamp_fl).group(1)
                    # elif tile == 'PEA':
                    #     name_resamp = re.search(r'/resampled_senegal_hls/trimmed/PEA/(.*?).tif', resamp_fl).group(1)
                    # elif tile == 'PFA':
                    #     name_resamp = re.search(r'/resampled_senegal_hls/trimmed/PFA/(.*?).tif', resamp_fl).group(1)


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
                        # print('Error loading file!')
                        continue

                    print(out_img.shape)

                    # save the result
                    # (don't forget to set the appropriate metadata)
                    if cloud:
                        out_fl_name = f'/home/geoint/tri/match-hls-sen/output-{tile}-cloud/{name_hls}-{name_resamp}-cloud.tif'
                    else:
                        # out_fl_name = f'/home/geoint/tri/match-hls-sen/output-PEV/{name_hls}-{name_resamp}.tif'

                        if not os.path.isdir(f'/home/geoint/tri/match-hls-sen/{tile}-1date/'):
                            os.mkdir(f'/home/geoint/tri/match-hls-sen/{tile}-1date/')

                        out_fl_name = f'/home/geoint/tri/match-hls-sen/{tile}-1date/{name_hls}-{name_resamp}.tif'

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
                    ) as dst:
                        dst.write(out_img)

                print(f'Finished save files to {out_fl_name}')
                count+=1

        else:
            break