from osgeo import gdal
import glob
import re
import os
import xarray as xr
import numpy as np


if __name__ == '__main__':

    file_dir = '/home/geoint/tri/nasa-multiyear-masks/epoch2015'

    fl_lst = sorted(glob.glob(file_dir+'/*.tif'))

    # resample the senegal mask ("mode" directory)
    for file in fl_lst:
        name = re.search(r'epoch2015/(.*?).tif', file).group(1)
        # print(name)
        ds = gdal.Open(file)
        proj = ds.GetProjection()

        # hls = gdal.Open('/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015/HLS.S30.T28PEA.2016001T112452.v2.0.tif')
        hls = gdal.Open('/home/geoint/PycharmProjects/tensorflow/out_hls_etz2015/HLS.S30.T28PEA.2016001T112452.v2.0.tif')
        destProj = hls.GetProjection()

        input = file
        output = f"/home/geoint/tri/resampled_senegal_hls/{name}_hls_mask.tif"

        if os.path.isfile(output):
            continue

        options = gdal.WarpOptions(
            xRes=30,
            yRes=30,
            # creationOptions="COMPRESS=LZW",
            dstSRS=destProj,
            # resampleAlg=gdal.GRA_Mode,
            resampleAlg="mode",
            targetAlignedPixels=True
            # options='overwrite'
        )

        gdal.Warp(output, input, options=options)

    ## check resample data size
    mask_fl_lst = sorted(glob.glob("/home/geoint/tri/resampled_senegal_hls/*.tif"))
    for mask_fl in mask_fl_lst:
        mask = np.squeeze(xr.open_rasterio(mask_fl).values)
        print(mask.shape)
