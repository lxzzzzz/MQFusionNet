import torch
import torch.nn as nn
import torchvision.models as models

class Backbones(nn.Module):
    """
    Factory class to instantiate the three heterogeneous backbones 
    as described in Section III-B of the manuscript.
    """
    def __init__(self):
        super(Backbones, self).__init__()

    @staticmethod
    def get_vgg16(pretrained=True):
        """
        VGG16 for high-resolution texture details.
        Output: 512 channels.
        """
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
        # We only need the features part (conv layers)
        return vgg.features

    @staticmethod
    def get_resnet50(pretrained=True):
        """
        ResNet50 for deep semantic features.
        Ref: "Features from ResNet50... extracted with dimensions of 2048 (14x14)"
        We use dilated convolutions in the last block to maintain spatial resolution.
        """
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        # replace_stride_with_dilation=[False, False, True] makes the last stage output 14x14
        resnet = models.resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        
        # Remove avgpool and fc, keep only feature extraction layers
        layers = list(resnet.children())[:-2] 
        return nn.Sequential(*layers)

    @staticmethod
    def get_densenet121(pretrained=True):
        """
        DenseNet121 for multi-scale adaptability.
        Output: 1024 channels.
        """
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)
        return densenet.features