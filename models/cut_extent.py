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

tile='all'

resamp_dir = f'/home/geoint/tri/match-hls-sen/{tile}-1date' # directory after trimming no-data border of cut raster
if tile in ['PEV', 'PFV']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/eCAS'
    hls_cloud_dir = '/home/geoint/tri/hls_cas_16-22_cloud'
elif tile in ['PEA','PFA','PGA']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/ETZ'
    hls_cloud_dir = '/home/geoint/tri/hls_etz_16-22_cloud'
elif tile in ['PCV']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/wCAS'
    hls_cloud_dir = '/home/geoint/tri/hls_wcas_16-22_cloud'

elif tile in ['PDB']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/ETZ/'
    hls_cloud_dir = '/home/geoint/tri/hls_pdb_cloud_mask/'
elif tile in ['PDA']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/ETZ/'
    hls_cloud_dir = '/home/geoint/tri/hls_pda_cloud_mask/'

elif tile in ['all']:
    hls_dir = '/home/geoint/tri/resampled_senegal_hls/multiyear/'
    hls_cloud_dir = '/home/geoint/tri/hls_pda_cloud_mask/'

resamp_lst = sorted(glob.glob(resamp_dir+'/*.tif'))
hls_lst = sorted(glob.glob(hls_dir+'/*.tif'))

# print(resamp_lst)
# print(hls_lst)

for resamp_fl in resamp_lst:

    print(resamp_fl)

    name_resamp = re.search(f'/home/geoint/tri/match-hls-sen/{tile}-1date/(.*?).tif', resamp_fl).group(1)[-22:]
    # print(name_resamp)

    for i in hls_lst:
        # print(i)
        if name_resamp[:8] in i:
            # print(i)
            hls_fl = i

    with rasterio.open(resamp_fl) as src, \
            rasterio.open(hls_fl) as src_to_crop:
        src_affine = src.meta.get("transform")

        # name_hls = re.search(r'/resampled_senegal_hls/mode/(.*?)_M1BS', resamp_fl).group(1)
        

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
        except:
            continue

        print(out_img.shape)

        # save the result
        # (don't forget to set the appropriate metadata)
        if not os.path.isdir(f'/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}/'):
            os.mkdir(f'/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}/')

        out_fl_name = f'/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}/{name_resamp}.tif'

        # if os.path.isfile(out_fl_name):
        #     continue
        
        with rasterio.open(
            # '/home/geoint/tri/match-hls-sen/result.tif',
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