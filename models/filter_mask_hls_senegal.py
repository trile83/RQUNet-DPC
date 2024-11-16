import rioxarray as rxr
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import xarray as xr
import cv2
import glob
import re

def floodfill(array):
    print(array.shape)
    crop_array = array.astype(np.uint8)
    print(crop_array.dtype)
    h, w = crop_array.shape[:2]
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h + 4, w + 4), np.uint8)

    canvas[1:h + 1, 1:w + 1] = crop_array.copy()
    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)
    array_flt = (~canvas | crop_array.astype(np.uint8))
    print(np.unique(array_flt))
    plt.imshow(np.squeeze(array_flt))
    plt.show()
    plt.close()

    return array_flt

def filtering_holes(mask_file):

    raster = rxr.open_rasterio(mask_file)
    original_array = raster.values
    original_array = np.squeeze(original_array)
    crop_array = original_array.copy()
    crop_array[crop_array != 2] = 0
    crop_array[crop_array == 2] = 1

    crop_array_flt = ndimage.binary_fill_holes(
        crop_array,
        structure=np.ones((3,3))
    ).astype(int)

    # crop_array_flt = floodfill(crop_array)
    ## stack the classes back together
    new_array = crop_array_flt.copy()
    # new_array[original_array == 3] = 3
    # new_array[original_array == 4] = 4
    # new_array[original_array == 5] = 5
    # new_array[original_array == 7] = 7
    # new_array[crop_array_flt == 1] = 0
    # new_array[original_array == 1] = 1

    return np.squeeze(new_array)

def save_tif(out_array, mask_file):

    outdir='/home/geoint/tri/resampled_senegal_hls/filtered/'

    ts_name = re.search('PDB/(.*?).tif',mask_file).group(1)
    # ts_name = mask_file[-53:-9]
    print(ts_name)

    ref_im = rxr.open_rasterio(mask_file)
    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
        dim="band",
        labels=ref_im.coords["band"].values[1:],
        drop=True
    )

    out_array = xr.DataArray(
        np.expand_dims(out_array, axis=-1),
        name='otcb',
        coords=ref_im.coords,
        dims=ref_im.dims,
        attrs=ref_im.attrs
    )

    out_array.attrs['long_name'] = ('filter')
    out_array = out_array.transpose("band", "y", "x")

    # Set nodata values on mask
    nodata = out_array.rio.nodata
    prediction = out_array.where(ref_im != nodata)
    prediction.rio.write_nodata(
        255, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'{outdir}{ts_name}_mask_filtered.tif',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )


if __name__ == "__main__":

    colors = ['green','orange','brown','brown','gray','white','black']
    colormap = pltc.ListedColormap(colors)

    masks_lst = sorted(glob.glob('/home/geoint/tri/resampled_senegal_hls/trimmed/PDB/*.tif'))

    for mask_file in masks_lst:
        out_mask = filtering_holes(mask_file)
        print(out_mask.shape)
        save_tif(out_mask, mask_file)
