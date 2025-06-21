import torch.nn as nn
import torchvision.models as models


def create_model(cfg):
    """
    Creates a model based on the config dictionary.
    Handles input channel modifications and output layer changes.
    """
    name = cfg.model.name.lower()
    pretrained = cfg.model.pretrained
    num_classes = cfg.model.num_classes

    model_func = getattr(models, name, None)
    if model_func is None:
        raise ValueError(f"Model {name} is not found in torchvision.")

    # Instantiate the model
    model = model_func(weights='IMAGENET1K_V1' if pretrained else None)

    # Modify the first conv layer if input is 1-channel
    if name in ['efficientnet_b4', 'convnext_tiny', 'swin_v2_t']:
        kernel_size = model.features[0][0].kernel_size
        stride = model.features[0][0].stride
        padding = model.features[0][0].padding
        bias = model.features[0][0].bias is not None
        model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, bias=bias)
    elif name == 'maxvit_t':
        kernel_size = model.stem[0][0].kernel_size
        stride = model.stem[0][0].stride
        padding = model.stem[0][0].padding
        bias = model.stem[0][0].bias is not None
        model.stem[0][0] = nn.Conv2d(1, model.stem[0][0].out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding, bias=bias)
    else:
        raise ValueError(f"Model {name} is not supported. Supported models: "
                         "efficientnet_b4, convnext_tiny, swin_v2_t, maxvit_t")
        
    if name == 'efficientnet_b4':
        in_features = model.classifier[1].in_features
        bias = model.classifier[1].bias is not None
        model.classifier[1] = nn.Linear(in_features=in_features,
                                        out_features=num_classes,
                                        bias = bias)
        
    return model
