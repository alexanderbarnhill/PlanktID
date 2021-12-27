import torch.nn as nn
import torchvision as tv

from options import Options


class ResNet(nn.Module):
    def __init__(self, options: Options):
        super().__init__()

        self.model_options = options.model_opts()
        self.model = self.get_model(self.model_options["size"], self.model_options["pretrained"])

    def get_model(self, size, pretrained):
        if size == 18:
            return tv.models.resnet18(pretrained=pretrained)
        if size == 34:
            return tv.models.resnet34(pretrained=pretrained)

        if size == 50:
            return tv.models.resnet50(pretrained=pretrained)