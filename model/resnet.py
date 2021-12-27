import torch.nn as nn
import torchvision as tv

from data.options import Options


class ResNet(nn.Module):
    def __init__(self, options: Options):
        super().__init__()

        self.model_options = options.model_opts()
        self.model = self.get_model(self.model_options["size"], self.model_options["pretrained"])
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.model.fc = nn.Linear(512 * 1, self.model_options["class_count"])

    def get_model(self, size, pretrained):
        if size == 18:
            return tv.models.resnet18(pretrained=pretrained)
        if size == 34:
            return tv.models.resnet34(pretrained=pretrained)
        if size == 50:
            return tv.models.resnet50(pretrained=pretrained)

    def forward(self, X):
        return self.model(X)
