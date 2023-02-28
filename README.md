# RQUNet-DPC
 
Dense Predictive Coding [model](https://github.com/TengdaHan/DPC) and UNet [model](https://github.com/jaxony/unet-pytorch) architecture framework for segmenting satellite images time series.<br>

## Proposed Architecture and Workflow
![Model Architecture](DPC_Flowchart.png) <br>

## Dense Predictive Coding Architecture
![Dense Predictive Coding](models/asset/arch.png) <br>

## How to run the code
Create the Python environment 3.8.12 in terminal/command line for Linux OS <br>
```conda env create -f environment.yml```

To train DPC + UNet model for image segmentation, prepare the Dataset in time series format for Pytorch: T x C x H x W <br>
```python train_dpc_seg.py```

To perform prediction for small tiles of large raster, same dataset format <br>
```python predict_dpc_seg.py```

To perform window sliding prediction, run the file <br>
```python predict_sliding.py```

To run experiment DPC+Poisson segmentation <br>
```python dpc_poisson.py```





