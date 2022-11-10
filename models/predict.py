import disstl.models as models
import torch
import torchvision
from disstl.datasets.smart.datasets import from_cube
from disstl.datasets.transforms import ClipBands, MinMaxNormalize
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros((32,32,3))
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img


if __name__ == '__main__':


    satellite = 'sentinel2'
    region = 'BH_R001'
    path_to_cubes ='BH_R001/region.hdf5'

    # region = 'BLA_QFABRIC_R007'
    # path_to_cubes ='E:/BlackSky/temp/cubes_new/BLA_QFABRIC_R007/region.hdf5'
    band_mins = [0, 0, 0, 0, 0]
    band_maxs = [17456, 18379, 19528, 16610, 15271]
    #band_mins = [0, 0, 0]
    #band_maxs = [17456, 18379, 19528]

    annotation_kwargs = dict(active_to_inactive_ratio=0.5,
                        n_active_sequence_draws=2,
                        n_inactive_sequence_draws=1
                        )

    transforms = torchvision.transforms.Compose([ClipBands(mins=band_mins, maxs=band_maxs),
                                                MinMaxNormalize(mins=band_mins, maxs=band_maxs)])

    region_id2idx = {region: 0}
    train_dataset_kwargs = dict(path=path_to_cubes,
                                chip_shape=[32,32],
                                stride=[16,16],
                                band_names=['red','green', 'blue', 'nir', 'swir16'],
                                #band_names=['red','green', 'blue'],
                                seq_len=10,
                                transforms=transforms,
                                spectral_indices=None,
                                with_annotations=True,
                                annotation_kwargs=annotation_kwargs,
                                satellite=satellite,
                                region_id=region,
                                region_id2idx=region_id2idx,
                                use_cache_dataset_preprocessing=False
                                )

    # from_cube returns an instance of AnnotatedDataset, defined in disstl/datasets/smart/datasets.py
    train_ds = from_cube(**train_dataset_kwargs)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=10, drop_last=True, shuffle=True)

    ################
    C = 5
    NUM_CLASSES = 2 
    dir_checkpoint = 'home/mle35/cpc-cgru-bs/models/checkpoints/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'device: {device}')
    
    model_checkpoint = 'checkpoints/bidirect_seg_BH_R001_conv3d_5band_unetvae_194_0.32482007145881653.pth'
    encoder_weight = 'checkpoints/reconstruct_5band_unetvae_24_0.000947555701714009.pth'

    model = models.BiDirectionalTemporalSegmentation_RQUnet(
        genc_backbone="efficientnet-b7",
        encoder_arch = "unet",
        # encoder_dim = 1,
        # 148 - encoder with z, 192 - no z 
        # encoder_dim = 448,
        encoder_dim = 192,
        gar_dim= 128,
        channels=C,
        mlp_hidden_dim=64,
        num_classes=NUM_CLASSES,
        device=device,
        model_weight=encoder_weight,
        pretrained = True
    )
    
    
    # model = models.BiDirectionalTemporalSegmentation_StackedRQUnet(
    #     genc_backbone="efficientnet-b7",
    #     encoder_arch = "unet",
    #     encoder_dim = 192,
    #     gar_dim= 128,
    #     channels=C,
    #     mlp_hidden_dim=64,
    #     num_classes=NUM_CLASSES,
    #     pretrained = True,
    #     device = device,
    #     model_weight = encoder_weight
    # )

    # if use a stacked Unet, don't load model_checkpoint here
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    

    if torch.cuda.is_available():
        model.cuda()

    for batch in train_dl:
        x = batch['x'].cuda()
        y = batch['binary_activity_mask'].numpy()
        output = model(x)
        y_pred =output.y_pred.cpu().clone().detach().numpy()

        index_array = np.argmax(y_pred, axis=2)
        break

    #data_dir = 'output/bidirect_unetvae_stacked/'
    data_dir = 'output/bidirect_rqunetvae_0803/'

    
    for j in range(10):
        for i in range(10):
            plt.figure(figsize=(20,20))
            plt.subplot(1,3,1)
            plt.title("Image")
            image = np.transpose(batch['x'].numpy()[j,i,:3,:,:], (1,2,0))  
            plt.imshow(rescale_truncate(image))
            plt.subplot(1,3,2)
            plt.title("Segmentation Label")
            #values = np.unique(y.ravel())
            plt.imshow(y[j,i,:,:])
        
            plt.subplot(1,3,3)
            plt.title("Segmentation")
            #values = np.unique(y.ravel())
            plt.imshow(index_array[j,i,:,:])
            plt.savefig(data_dir+str(j)+ '_' + str(i))
        
        
    for batch1 in train_dl: 
        x1 = batch1['x'].cuda()
        y1 = batch1['binary_activity_mask'].numpy()
        output1 = model(x1)
        y_pred1 =output1.y_pred.cpu().clone().detach().numpy()
        
        index_array1 = np.argmax(y_pred1, axis=2)
        break
        
    i = 1  
    for idx_temp in range(10):
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1)
        plt.title("Image")
        image1 = np.transpose(batch1['x'].numpy()[idx_temp,i,:3,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image1))
        plt.subplot(1,3,2)
        plt.title("Segmentation Label")
        #values = np.unique(y.ravel())
        plt.imshow(y1[idx_temp,i,:,:])

        plt.subplot(1,3,3)
        plt.title("Segmentation")
        #values = np.unique(y.ravel())
        plt.imshow(index_array1[idx_temp,i,:,:])
        

    for i in range(10):
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1)
        plt.title("Image")
        image = np.transpose(batch1['x'].numpy()[2,i,:3,:,:], (1,2,0))  
        plt.imshow(rescale_truncate(image))
        plt.subplot(1,3,2)
        plt.title("Segmentation Label")
        #values = np.unique(y.ravel())
        plt.imshow(y1[2,i,:,:])

        plt.subplot(1,3,3)
        plt.title("Segmentation")
        #values = np.unique(y.ravel())
        plt.imshow(index_array1[2,i,:,:])
        plt.savefig(data_dir+'new_' + str(i))

