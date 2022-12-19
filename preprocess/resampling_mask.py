import rasterio as rio
from rasterio.enums import Resampling
import numpy as np
import xarray as xr
import cv2
import glob
import matplotlib.pyplot as plt


if __name__ == '__main__':

    upscale_factor = 1

    labels = sorted(glob.glob('/home/geoint/tri/nasa_senegal/new_masks/*.tif'))

    for lf in labels:

        # lf = "/home/geoint/tri/nasa_senegal/new_masks/Tappan01_WV02_20181217_M1BS_1030010089CC6D00_mask_segs_reclassified.tif"
        label = np.squeeze(xr.open_rasterio(lf).values)

        tappan_name = lf[-71:-49]

        label[label == 5] = 1  # merge burned area to other vegetation
        label[label == 7] = 1  # merge no-data area to shadow/water
        label[label == 4] = 1
        label[label == 3] = 1
        # label = label - 1

        label_res = cv2.resize(label, dsize=(335, 335), interpolation=cv2.INTER_CUBIC)

        # label_res = label_res.reshape((335,335))

        # with rio.open("/home/geoint/tri/nasa_senegal/new_masks/Tappan01_WV02_20181217_M1BS_1030010089CC6D00_mask_segs_reclassified.tif") as dataset:
        #     # resample data to target shape
        #     data = dataset.read()
        #
        #     # scale image transform
        #     transform = dataset.transform * dataset.transform.scale(
        #         (dataset.width / data.shape[-1]),
        #         (dataset.height / data.shape[-2])
        #     )
        #
        #     out_meta = dataset.profile

        resampled = '/home/geoint/tri/nasa_senegal/test_features/' + f'{tappan_name}'

        # with rio.open(resampled, "w", **out_meta) as dest:
        #     dest.write(label_res)

        np.save(resampled, label_res)

            # print(data.shape)
    #
    #
    # plt.imshow(data)
    # plt.show()