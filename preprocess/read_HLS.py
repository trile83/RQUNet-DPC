import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from osgeo import gdal

def normalize_image(image: np.ndarray, normalize: float):
    """
    Normalize image within parameter, simple scaling of values.
    Args:
        image (np.ndarray): array to normalize
        normalize (float): float value to normalize with
    Returns:
        normalized np.ndarray
    """
    image = image / normalize
    return image

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
        for i in range(image.shape[-1]):
            image[:, :, i] = (
                image[:, :, i] - np.min(image[:, :, i])) / \
                (np.max(image[:, :, i]) - np.min(image[:, :, i]))

    return image


def get_false_color(image):
    false_color = np.zeros((image.shape[0], image.shape[1], 3))
    false_color[:,:,0] = image[:,:,6]
    false_color[:,:,1] = image[:,:,2]
    false_color[:,:,2] = image[:,:,1]
    return false_color

def get_rgb(image):
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = image[:, :, 3]
    rgb[:, :, 1] = image[:, :, 2]
    rgb[:, :, 2] = image[:, :, 1]
    return rgb


if __name__ == '__main__':


    # im_files = sorted(glob.glob('/home/geoint/tri/data/HLS.S30.T28PEV.2020290T112121.v2.0.B*.tif'))

    # out_files = sorted(glob.glob('/home/geoint/tri/hls/*.tif'))

    # eCAS
    # all_fls = sorted(glob.glob('/home/geoint/tri/data/*.tif'))
    #
    # # ETZ
    all_fls = sorted(glob.glob('data_hls_etz_2/*.tif'))

    # ETZ 2015-2020
    # all_fls = sorted(glob.glob('/home/geoint/tri/data_hls_etz/*.tif'))

    seen_fls = []
    im_dict ={}
    for file in all_fls:
        name = file[-42:-6]
        long_name = file[-42:-4]
        # print(long_name)
        if name[0] != 'H' :
            continue
        elif name[-1] != 'B':
            continue
        elif name not in seen_fls:
            im_bands = sorted(glob.glob(f'data_hls_etz_2/{name}*.tif'))
            im_dict[name] = im_bands
            seen_fls.append(name)

    print(seen_fls)

    print("Done input to list!")

    for key in im_dict.keys():
        ImageList = []
        for df in im_dict[key]:
            # print(df)
            ImageList.append(df)
            name = df[-42:-8]
            print(name)

        VRT = 'OutputImage.vrt'
        gdal.BuildVRT(VRT, ImageList, separate=True, callback=gdal.TermProgress_nocb)

        # stacked the bands
        InputImage = gdal.Open(VRT, 0)  # open the VRT in read-only mode
        gdal.Translate(f'out_hls_etz_2/{name}.tif', InputImage, format='GTiff',
                       creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
                       callback=gdal.TermProgress_nocb)
        del InputImage  # close the VRT


    # build VRT
    # ImageList = []
    # for df in im_files:
    #     # ImageList = ['Band1.tif', 'Band2.tif','Band3.tif']  # or use sorted(glob.glob('*.tif')) if input images are sortable
    #     ImageList.append(df)
    #     name = df[-42:-8]

    # print(name)
    # VRT = 'OutputImage.vrt'
    # gdal.BuildVRT(VRT, ImageList, separate=True, callback=gdal.TermProgress_nocb)
    #
    # # stacked the bands
    # InputImage = gdal.Open(VRT, 0)  # open the VRT in read-only mode
    # gdal.Translate(f'out_hls/{name}.tif', InputImage, format='GTiff',
    #                creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
    #                callback=gdal.TermProgress_nocb)
    # del InputImage  # close the VRT
