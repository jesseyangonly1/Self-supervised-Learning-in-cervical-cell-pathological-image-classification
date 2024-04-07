'''
this model is for a serious of test. mostly, a ResNet50 backbone added by three Dense layers
also there is a SimCLR loss layer.
'''
import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()

        self.backbone = models.resnet18(pretrained=False, num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

