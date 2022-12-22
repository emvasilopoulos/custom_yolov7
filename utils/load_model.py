from model.components.basic import Conv
from model.experimental import Ensemble
import torch
from torch import nn
from model.components.basic import *
from model.components.csp_net import *
from model.components.orepa import *
from model.components.repvgg import *
from model.components.swin_transformer import *
from model.components.swin_transformer_v2 import *
from model.components.transformer import *
from model.components.yolor import *
from model.components.yolov5 import *

def load_model(weights_path: str, map_location: torch.device):
    loaded_model = torch.load(weights_path, map_location=map_location)  # load
    if loaded_model.get('ema'):
        model = loaded_model['ema'].float().fuse().eval()
    else:
        model = loaded_model['model'].float().fuse().eval()

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    return model

def load_ensemble(weights: list, map_location: torch.device):
    # Loads an ensemble of model weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()

    for w in weights:
        ckpt = torch.load(w, map_location=map_location)  # load
        if ckpt.get('ema'):
            m = ckpt['ema'].float().fuse().eval()
        else:
            m = ckpt['model'].float().fuse().eval()
        model.append(m)  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    print('Ensemble created with %s\n' % weights)
    for k in ['names', 'stride']:
        setattr(model, k, getattr(model[-1], k))
    return model  # return ensemble