import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
from skimage.segmentation import quickshift, slic
import time
import scipy.ndimage as nd
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import geopandas as gpd
import json
import re
import os
import glob

def rescale_image(image: np.ndarray, rescale_type: str = 'per-image'):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    if rescale_type == 'per-image':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.min(image[i, :, :])) / \
                (np.max(image[i, :, :]) - np.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros(image.shape)
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def read_imagery(pl_file):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if img_data.ndim > 2:
        img_data = np.transpose(img_data, (1,2,0))

    return img_data, ref_im


def stack_masks(path_lst):

    out_lst = []

    for image_path in path_lst:

        img_data, ref_im = read_imagery(image_path)
        out_lst.append(img_data)

    out_array = np.stack(out_lst, axis=0)

    return out_array, ref_im


def combine_masks(mask_arrays):

    combined = np.zeros(mask_arrays[0].shape)

    for mask in mask_arrays:

        print('mask unique values', np.unique(mask))

        mask[mask!=2]=0
        mask[mask==2]=1

        combined+=mask


    combined[combined<1]=0
    combined[combined>0]=1

    return combined


def save_raster(ref_im, prediction, name, epoch):

    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
            dim="band",
            labels=ref_im.coords["band"].values[1:],
            drop=True
        )
    
    prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name='gm',
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('combined')
    prediction = prediction.transpose("band", "y", "x")

    # Set nodata values on mask
    # nodata = prediction.rio.nodata
    # prediction = prediction.where(ref_im != nodata)
    # prediction.rio.write_nodata(
    #     255, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'/home/geoint/tri/nasa-multiyear-masks/epoch{epoch}/{name}-{epoch}.tif',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )

if __name__ == '__main__':

    master_dir = '/home/geoint/tri/allmasks'
    epoch = 2023

    mask_dir = sorted(glob.glob(f"{master_dir}/*.tif"))
    print("total file: ", len(mask_dir))

    ### get mask file path

    mask_dict = {}

    for idx, file in enumerate(mask_dir):
        # print(file)
        name = re.search(f'{master_dir}/(.*?)_M1BS', file).group(1)
        shortname = re.search(f'{master_dir}/(.*?)_WV', file).group(1)
        year = int(name[-8:-4])

        # print(shortname)
        # print('year: ', year)

        if shortname not in mask_dict.keys():
            mask_dict[shortname] = []

        if year > int(epoch-4) and year < int(epoch+1):
            mask_dict[shortname].append(file)
            # print(f'{shortname} with year {year}')

    for shortname in mask_dict.keys():

        print('tappan: ', shortname)
        print('length of list: ', len(mask_dict[shortname]))

        if len(mask_dict[shortname]) < 1:
            continue

        out_arrays, ref_im = stack_masks(mask_dict[shortname])

        out_mask = combine_masks(out_arrays)

        save_raster(ref_im, out_mask, shortname, epoch)

        ## visualize

        plt.figure(figsize=(20,20))
        plt.title(f"out mask")
        plt.imshow(out_mask)
        plt.savefig(f'/home/geoint/tri/nasa-multiyear-masks/images/{shortname}-{epoch}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
