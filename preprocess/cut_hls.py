import glob
import rasterio as rio, fiona
import rasterio.mask as mask
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import h5py
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from skimage import exposure

def Clipper(raster, vector):

    # Read Shapefile
    with fiona.open(vector, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rio.open(raster) as src:
        # read imagery file
        out_image, out_transform = mask.mask(src, shapes, crop=True, nodata=-10000, filled=True, invert=False)

        # Check that after the clip, the image is not empty
        test = out_image[~np.isnan(out_image)]

        if test[test > 0].shape[0] == 0:
            raise RuntimeError("Empty output")

        out_meta = src.profile
        out_meta.update({"height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

    return (out_image, out_meta)


def clip_raster(raster_file, shape_box_file):

    raster = rxr.open_rasterio(raster_file)
    shape_box = gpd.read_file(shape_box_file)
    raster_clipped = raster.rio.clip(shape_box.geometry.apply(mapping),
              # This is needed if your GDF is in a diff CRS than the raster data
              shape_box.crs)

    # print(f'raster clipped shape: {raster_clipped.shape}')
    # print(f'raster clipped type: {type(raster_clipped)}')
    # print(f'raster clipped nodata: {np.any(raster_clipped<-9000)}')

    if not np.any(raster_clipped<-9000) and raster_clipped is not None:
        return raster_clipped

def pad_3d(arr: np.ndarray, out_shape, tappan_name) -> np.ndarray:

    ### issue: pad not entirely correct on direction
    x, y, z = arr.shape
    output = np.ones(out_shape, dtype=arr.dtype)
    output = output * -1
    if tappan_name in ['Tappan01', 'Tappan05']:
        output[-x:, -y:, -z:] = arr ## need to check again
    elif tappan_name in ['Tappan02', 'Tappan04']:
        output[:x, :y, :z] = arr
    return output

def normalize_image(image: np.ndarray, normalize=255):
    """
    Normalize image within parameter, simple scaling of values.
    Args:
        image (np.ndarray): array to normalize
        normalize (float): float value to normalize with
    Returns:
        normalized np.ndarray
    """
    # image = image / normalize
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
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
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

def rescale_truncate(image): ## function to rescale image for visualization
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image)
    map_img =  np.zeros(image.shape)
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img


if __name__ == '__main__':

    im_files = sorted(glob.glob('/home/geoint/PycharmProjects/tensorflow/out_hls/*.tif'))
    # hlsmask_files = sorted(glob.glob('/home/geoint/tri/data/*Fmask.tif'))
    shape_files = sorted(glob.glob('/home/geoint/tri/nasa_senegal/shape_bound/*.shp'))
    mask_files = sorted(glob.glob('/home/geoint/tri/hls_mask/*.npy'))
    out_dir = '/home/geoint/tri/hls/'
    # out_mask_dir = '/home/geoint/tri/hls_cloud_mask/'

    data_dict = {}
    im_dict = {} # temp dictionary to store ts images

    for shape in shape_files:
        # print("shape file: ", shape)
        tappan_name = shape[-21:-13]
        # print("shape file name: ", shape[-21:-13])

        print(tappan_name)

        # if tappan_name not in data_dict.keys():
        #     data_dict[tappan_name] = {}
        #     im_dict[tappan_name] = []
        #     data_dict[tappan_name]['ts'] = []
        #     data_dict[tappan_name]['mask'] = []

        if tappan_name not in im_dict.keys():
            im_dict[tappan_name] = {}

        if tappan_name not in data_dict.keys():
            data_dict[tappan_name] = {}
            data_dict[tappan_name]['mask'] = np.zeros((335, 335))

        for idx, im_fl in enumerate(tqdm(im_files)):
            # print("tif file: ", im[-38:-4])
            tif_name = im_fl[-38:-4]
            tif_type = tif_name[11:14]
            # print(tif_type)

            if tif_type not in im_dict[tappan_name].keys():
                im_dict[tappan_name][tif_type] = []

            if tif_type not in data_dict[tappan_name].keys():
                # data_dict[tappan_name] = {}
                data_dict[tappan_name][tif_type] = {}
                data_dict[tappan_name][tif_type] = []
                # data_dict[tappan_name]['mask'] = np.zeros((335, 335))


            mask_fl = f"/home/geoint/tri/data/{tif_name}.Fmask.tif"

            raster = im_fl
            # clipped = out_dir+f'{tappan_name}_{tif_name}.tif'
            desired_shape = (13, 335, 335)

            # Clip the raster
            try:
                array_out, out_meta = Clipper(raster, shape)
                # array_out = clip_raster(raster, shape)
                array_out = np.asarray(array_out)
                # rescale for pixel value to be between 0 and 1
                array_out = normalize_image(array_out)
                # cut mask
                cloud_mask, out_mask_meta = Clipper(mask_fl, shape)
                # cloud_mask = clip_raster(mask_fl, shape)
                cloud_mask = np.asarray(cloud_mask)
                # print(f"mask cut shape: {cloud_mask.shape}")

            except:
                # print(f"{tappan_name} is not in {tif_name}")
                pass

            # if key not in data_dict.keys():
            #     im_dict[key] = []

            if array_out.shape != (13, 335, 335):  # if the array is not 335x335 resize using pad3d function
                array_out = pad_3d(array_out, desired_shape, tappan_name)
                # cloud_mask = pad_3d(cloud_mask, desired_shape)

            cloud_mask = cloud_mask.reshape((cloud_mask.shape[1], cloud_mask.shape[2], cloud_mask.shape[0]))
            # print(f"mask shape: {image.shape}")
            temp_arr = cloud_mask % 16
            count_cloud = np.count_nonzero(temp_arr)

            if count_cloud < 1000:
                # print(array_out.shape)
                image = np.transpose(array_out[1:4,:,:], (1,2,0))
                image = rescale_truncate(image)
                plt.imshow(image)
                plt.savefig(f'/home/geoint/tri/hls_rgb/{tappan_name}-{tif_name}-{idx}.png')
                plt.close()
                im_dict[tappan_name][tif_type].append(array_out)

            # if key not in data_dict.keys():
            #     data_dict[key] = {}
            #     data_dict[key]['ts'] = []
            #     data_dict[key]['mask'] = np.zeros((335, 335))

            # print("array out shape: ", array_out.shape)

    for idx, mask_file in enumerate(mask_files):
        mask_name = mask_file[-26:-18]
        # print(mask_name)
        mask = np.load(mask_file)
        mask = mask - 1
        mask = mask.astype(int)
        data_dict[mask_name]['mask'] = mask

    for key in data_dict.keys():
        for key_1 in data_dict[key].keys():
            if key_1 != 'mask':
                print(f"length of image dict {key} {key_1} :{len(im_dict[key][key_1])}")
            # data_dict[key]['ts'] = np.stack(im_dict[key], axis=0)

    for key in data_dict.keys():
        for key_1 in data_dict[key].keys():
        # print(f"length of image dict{key} :{len(im_dict[key])}")
            if key_1 != 'mask':
                if len(im_dict[key][key_1]) > 0:
                    data_dict[key][key_1] = np.stack(im_dict[key][key_1], axis=0)

    out_dir = '/home/geoint/tri/hls_ts_video'

    #########################

    h = h5py.File(f'{out_dir}/hls_data.hdf5')

    for k, v in data_dict.items():
        # h.create_dataset(k, data=np.array(v, dtype=np.float16), compression="gzip", compression_opts=9)
        h.create_dataset(f"{k}_mask", data=v['mask'], compression="gzip", compression_opts=9)
        for m, n in data_dict[k].items():
            if m != 'mask':
                h.create_dataset(f"{k}_{m}_ts", data=n, compression="gzip", compression_opts=9)


    print(f'finished the data processing step!')



