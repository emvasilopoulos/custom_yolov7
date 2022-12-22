from models.yolo import Model
from utils.load_model import load_model
from utils.torch_utils import select_device
import torch
from torch import nn

if __name__ == '__main__':
    input_device = 'cpu'  # Supports also input_device = ['0', '1', ...]  for multiple CUDA devices
    input_weights = '../weights/yolov7.pt'
    cfg_path = '../cfg/my_yolov7.yaml'
    device = select_device(input_device)

    model = Model(cfg_path, 3, 1)

    start_layer_idx = 0
    end_layer_idx = 50
    subnetwork = nn.Sequential()
    for idx, layer in enumerate(list(model)[start_layer_idx: end_layer_idx + 1]):
        subnetwork.add_module("layer_{}".format(idx), layer)
    print(subnetwork)