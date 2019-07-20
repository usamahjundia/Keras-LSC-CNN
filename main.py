import warnings
warnings.filterwarnings('ignore')
import network as nk
from convert_lsccnn_to_keras import convert_torch_model_to_keras as cvt
import matplotlib.pyplot as plt
import cv2
import numpy as np


# IMPORTANT : replace the path below with the path to your pth file.
## this is a keras model. you can just use it for pred or just save it for later.
## as of now, training is not supported.

lsccnn = cvt('./scale_4_epoch_24.pth')

def read_image_and_convert(impath):
    image = cv2.imread(impath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def draw_detections_on_image(image,detections,radius=4,thickness=-1):
    points, w, h = detections
    rows, cols = np.where(points)
    for x,y in zip(cols,rows):
        cv2.circle(image,(x,y),radius=radius,color=(0,255,0),thickness=thickness)

def image_show(image,figsize=(10,8)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.show()

testim = read_image_and_convert('./images/seq_001914.jpg')
image_show(testim)
detections = nk.pred_for_one_image(lsccnn,testim)
print("DONE")
draw_detections_on_image(testim,detections)
image_show(testim)
