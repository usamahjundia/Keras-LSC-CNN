# Keras-LSC-CNN

This repository contains an unofficial Keras Implementation of LSC-CNN, a crowd counting model as proposed in the paper [Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection.](https://arxiv.org/abs/1906.07538) Currently, the implementation only supports parsing weights from the pytroch pretrained model for inference purposes.

Some functions were taken from the [official pytorch implementation repository](https://github.com/val-iisc/lsc-cnn) for performing NMS.

Provided in this repository are:
1. network.py, containing the keras definition of the LSC-CNN model, and some functions for performing inference.
2. convert_lsccnn_to_keras.py, containing the code that reads the weight file of the pretrained torch models and outputs a keras model with weights from the pretrained pytorch model loaded into it.
3. main.py, as an example showing how to create the converted keras model along with how to use it for inference.

## Usage

### 1. Converting model to keras
To convert a pytorch model to keras, first you will need to: 
- head to [the official implementation repository](https://github.com/val-iisc/lsc-cnn) and download the pretrained model from there.
- extract and note the path to the .pth file of the pretrained model

And then, you can either:
- use the function *convert_torch_model_to_keras*, supplying the path to the .pth file as such:
```py
from convert_lsccnn_to_keras import convert_torch_model_to_keras
# ...
# fill in with the path to pth model
model = convert_torch_model_to_keras('/path/to/pth/file')
# this will return a keras Model. You can use it straight for inference or just save it for later
# ...
```
Or 
- Use convert_lsccnn_to_keras.py as a command line program like such and retrieve the .h5 file
```
python3 convert_lsccnn_to_keras.py --pthpath /path/to/pth/file --savedir /where/to/save/model
```

2. Doing inference with the model

network.py contains the functions you will need for inference. Function *pred_for_one_image* can be used to do prediction.
```py
from network import pred_for_one_image
# ...
model = # the lsccnn model
image = # .. make sure the image is in shape (height, width, channels) and is in RGB format
loc, h, w = pred_for_one_image(model,image)
# loc is a binary array, True if a certain pixel is decided as a center for a detected head
# h and w is the respective box height and widths for every locations
```

TODO :
- Implement NMS as a keras function if possible
- vectorize a lot of operations on NMS 
- create a prediction function for batch images