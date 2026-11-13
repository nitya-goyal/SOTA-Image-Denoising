"""The architectures and attention variants used in this study."""
from .nafnet import NAFNet
from .unet import UNet


def build_model(config):
    options = dict(config)
    name = options.pop('name')
    attention = options.pop('attention', None)

    expected_attention = {
        'nafnet': 'sca',
        'nafnet_spatial': 'spatial',
        'nafnet_channel': 'channel',
        'nafnet_combined': 'combined',
    }

    if name in expected_attention:
        target_att = expected_attention[name]
        if attention is not None and attention != target_att:
            raise ValueError(
                f"Conflicting settings: model name '{name}' implies attention='{target_att}', "
                f"but attention='{attention}' was provided"
            )
        return NAFNet(attention=target_att, **options)

    if name == 'unet':
        if attention is not None:
            raise ValueError("UNet does not take an 'attention' parameter")
        return UNet(**options)

    raise ValueError(
        f"Unknown model: {name}; choose unet, nafnet, nafnet_spatial, "
        f"nafnet_channel, or nafnet_combined"
    )
