import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
import glob
import re

# the first one is your raster on the right
# and the second one your red raster

tile='PEA'
resamp_dir = f'/home/geoint/tri/match-hls-sen/{tile}-1date'
hls_dir = '/home/geoint/tri/resampled_senegal_hls/mode/ETZ'
resamp_lst = sorted(glob.glob(resamp_dir+'/*.tif'))
hls_lst = sorted(glob.glob(hls_dir+'/*.tif'))

# print(resamp_lst)
# print(hls_lst)

for resamp_fl in resamp_lst:

    name_resamp = re.search(f'/match-hls-sen/{tile}-1date/HLS.S30.T28PEA.2016001T112452-(.*?).tif', resamp_fl).group(1)

    for i in hls_lst:
        if name_resamp in i:
            print(i)

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
        out_fl_name = f'/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}/{name_resamp}.tif'
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
        ) as dst:
            dst.write(out_img)