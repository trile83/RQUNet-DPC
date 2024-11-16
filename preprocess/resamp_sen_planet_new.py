from osgeo import gdal
import glob
import re
import os
import xarray as xr
import numpy as np


if __name__ == '__main__':

    file_dir = '/home/geoint/tri/allmasks/'

    fl_lst = sorted(glob.glob(file_dir+'/*.tif'))

    # resample the senegal mask ("mode" directory)
    for file in fl_lst:
        name = re.search(r'/allmasks/(.*?)_mask.tif', file).group(1)
        # print(name)
        if 'Tappan32' not in name:
            continue
            
        ds = gdal.Open(file)
        proj = ds.GetProjection()

        hls = gdal.Open('/home/geoint/tri/planet-data/tile13-senegal-eetz/L15-0951E-1105N-012021.tif')
        destProj = hls.GetProjection()

        input = file
        output = f"/home/geoint/tri/resampled_senegal_planet/{name}_planet_mask.tif"

        if os.path.isfile(output):
            continue

        options = gdal.WarpOptions(
            xRes=4.777314267160000405,
            yRes=4.777314267160000405,
            # creationOptions="COMPRESS=LZW",
            dstSRS=destProj,
            # resampleAlg=gdal.GRA_Mode,
            resampleAlg="mode",
            targetAlignedPixels=True
            # options='overwrite'
        )

        gdal.Warp(output, input, options=options)

    ## check resample data size
    mask_fl_lst = sorted(glob.glob("/home/geoint/tri/resampled_senegal_planet/*.tif"))
    for mask_fl in mask_fl_lst:
        mask = np.squeeze(xr.open_rasterio(mask_fl).values)
        print(mask.shape)
