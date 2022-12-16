import logging
import numpy as np
# import tensorflow as tf
import scipy.signal.windows as w
from tqdm import tqdm
# from .tiler.tiler import Tiler, Merger
from .mosaic import from_array
from ..utils.data import normalize_image, rescale_image, \
    standardize_batch, standardize_image
from tiler import Tiler, Merger


def window2d(window_func, window_size, **kwargs):
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)


def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack(
        [np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack(
        [window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack(
        [np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack(
        [window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [window_ul, window_u, window_ur],
        [window_l, window, window_r],
        [window_bl, window_b, window_br],
    ])


def generate_patch_list(
            image_width,
            image_height,
            window_func,
            window_size,
            overlapping=False
        ):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        # max_height = int(image_height/step - 1) * step
        # max_width = int(image_width/step - 1) * step
        # print("max_height, max_width", max_height, max_width)
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        # max_height = int(image_height / step) * step
        # max_width = int(image_width / step) * step
        # print("else max_height, max_width", max_height, max_width)

    # for i in range(0, max_height, step):
    #    for j in range(0, max_width, step):
    for i in range(0, image_height-step, step):
        for j in range(0, image_width-step, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0:
                    border_x = 0
                if j == 0:
                    border_y = 0
                if i == image_height - step:
                    border_x = 2
                if j == image_width - step:
                    border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows

            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i

            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j

            # print(f'i {i} j {j} patch_height {patch_height}
            # patch_width {patch_width}')

            # Adding the patch
            patch_list.append(
                (
                    j,
                    i,
                    patch_width,
                    patch_height,
                    current_window[:patch_width, :patch_height]
                )
            )
    return patch_list
    
    
def sliding_window_tiler(
            xraster,
            model,
            n_classes: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            mean=None,
            std=None,
            window: str = 'triang'  # 'overlap-tile'
        ):
    """
    Sliding window using tiler.
    """
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = \
    #    tf.data.experimental.AutoShardPolicy.OFF
    # batch = tf.data.Dataset.from_tensor_slices(
    #    np.expand_dims(batch, axis=0))
    # batch = batch.with_options(self.options)
    # batch = function(batch, batch_size=batch_size)

    tile_size = model.layers[0].input_shape[0][1]
    tile_channels = model.layers[0].input_shape[0][-1]

    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=2,
        # overlap=overlap,
        mode=pad_style,
        constant_value=600
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(tile_size, tile_size, n_classes),
        channel_dimension=2,
        # overlap=overlap,
        mode=pad_style,
        constant_value=600
    )

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)
    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    merger = Merger(
        tiler=tiler_mask, window=window)  # #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    # Iterate over the data in batches
    for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

        # print("AFTER SELECT", batch.shape)

        # Standardize
        batch = batch / 10000.0

        # if standardization is not None:
        #    batch = standardize_batch(batch, standardization, mean, std)

        # print("AFTER STD", batch.shape)

        # Predict
        batch = model.predict(batch, batch_size=batch_size)
        batch = model(batch)
        # batch = np.moveaxis(batch, -1, 1)
        # print("AFTER PREDICT", batch.shape, batch_id)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    # prediction = merger.merge(
    # extra_padding=padding_mask, unpad=True, dtype=xraster.dtype,
    # normalize_by_weights=False)
    prediction = merger.merge(unpad=True)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(prediction > threshold, 1, 0).astype(np.int16)
        )

    """
    tiler1 = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, tile_channels),
        channel_dimension=2,
        overlap=0.50,
        #mode=pad_style,
        #constant_value=constant_value
    )

    tiler2 = Tiler(
        data_shape=xraster.shape,
        tile_shape=(tile_size, tile_size, n_classes),
        channel_dimension=2,
        overlap=(256, 256, 0),
        #mode=pad_style,
        #constant_value=constant_value
    )

    new_shape, padding = tiler1.calculate_padding()
    tiler1.recalculate(data_shape=new_shape)
    tiler2.recalculate(data_shape=new_shape)
    padded_image = np.pad(
        xraster.values, padding, mode=pad_style,
        constant_values=constant_value
    )

    merger = Merger(tiler=tiler2, window="overlap-tile")
    #merger = Merger(tiler=tiler2, logits=2)#, window="triang")
    #"hann")#"triang")#"barthann")#"overlap-tile")

    for batch_id, batch in tiler1(padded_image, batch_size=batch_size):
    #for batch_id, batch in tiler1(xraster, batch_size=batch_size):

        # remove no-data
        batch[batch < 0] = constant_value

        # Standardize
        if standardization is not None:
            print("STD")
            batch = standardize_batch(batch, standardization, mean, std)

        # preprocessing
        batch = model.predict(batch, batch_size=batch_size)

        batch = np.argmax(batch, axis=-1)

        # merge batch
        merger.add_batch(batch_id, batch_size, batch)

    #prediction = merger.merge(extra_padding=padding, dtype=xraster.dtype)
    # prediction = merger.merge(
    #    extra_padding=padding, dtype=xraster.dtype,
    # normalize_by_weights=False)
    #prediction = merger.merge(unpad=True)
    prediction = merger.merge(extra_padding=padding, dtype=padded_image.dtype)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(prediction > threshold, 1, 0).astype(np.int16)
        )

    logging.info(f"Mask info: {prediction.shape}, {prediction.min()},
    {prediction.max()}")
    """
    return prediction