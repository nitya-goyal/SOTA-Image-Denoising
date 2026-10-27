"""The two architectures used in this study."""
from .nafnet import NAFNet
from .unet import UNet


def build_model(config):
    options = dict(config)
    name = options.pop('name')
    if name == 'nafnet':
        return NAFNet(**options)
    if name == 'unet':
        return UNet(**options)
    raise ValueError(f'Unknown model: {name}; choose unet or nafnet')
