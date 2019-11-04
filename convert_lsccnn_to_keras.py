import torch
import numpy as np
import keras
from network import build_LSCCNN_model
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(
    description="A simple program to parse the pretrained LSC-CNN model weights from the official pytorch implementation into a keras model.")
parser.add_argument('--pthpath',dest='pthpath',required=True,help="The path leading to where the .pth file is")
parser.add_argument('--savedir',dest='savedir',required=True,help="Where to save the model")

def convert_torch_model_to_keras(pthpath):
    weights_dict = torch.load(pthpath)
    if 'state_dict' in weights_dict:
    	weights_dict = weights_dict['state_dict']
    kmodel = build_LSCCNN_model()
    for layer in tqdm(kmodel.layers):
        layername = layer.name
        layerw = layername+'.weight'
        layerb = layername+'.bias'
        if layerw in weights_dict and layerb in weights_dict:
            weights_mat = weights_dict[layerw].cpu().numpy()
            weights_mat = np.transpose(weights_mat,(2,3,1,0))
            bias_mat = weights_dict[layerb].cpu().numpy()
            layer.set_weights([weights_mat,bias_mat])
    return kmodel

if __name__ == "__main__":
    args = parser.parse_args()
    model = convert_torch_model_to_keras(args.pthpath)
    model.save(args.savedir)
