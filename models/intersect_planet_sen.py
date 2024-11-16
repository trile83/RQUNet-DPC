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

tile='tile13'

resamp_dir = f'/home/geoint/tri/resampled_senegal_planet/' # directory after trimming no-data border of cut raster

hls_dir = '/home/geoint/tri/planet-data/tile13-senegal-eetz/'

resamp_lst = sorted(glob.glob(resamp_dir+'/*.tif'))
hls_lst = sorted(glob.glob(hls_dir+'/*.tif'))

for resamp_fl in resamp_lst:
    year_resamp = re.search(r'WV(.*?).tif', resamp_fl).group(1)[3:7]
    print("Year WV resamp: ", year_resamp)
    print('Resampled File: ', resamp_fl)
    for hls_fl in hls_lst:

        name_hls = re.search(f'{hls_dir}(.*?).tif', hls_fl).group(1)
        year_hls = re.search(f'1105N-(.*?).tif', hls_fl).group(1)[2:]
        date_hls = re.search(f'1105N-(.*?).tif', hls_fl).group(1)[:2]
        print("Year HLS: ", year_hls)
        print("Date HLS: ", date_hls)


        # hls_fl = '/home/geoint/PycharmProjects/tensorflow/out_hls_cas2015/HLS.S30.T28PFV.2017325T112351.v2.0.tif'

        ## match year of WV and year of HLS data
        if int(year_resamp) == int(year_hls):

            with rasterio.open(resamp_fl) as src, \
                    rasterio.open(hls_fl) as src_to_crop:
                
                src_affine = src.meta.get("transform")

                name_resamp = re.search(f'/resampled_senegal_planet/(.*?).tif', resamp_fl).group(1)
                
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

                print('out image shape: ', out_img.shape)


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
                ) as dst:
                    dst.write(out_img)
