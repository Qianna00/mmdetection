from ..builder import HEADS
import torch.nn as nn
import torch
from ..builder import build_loss


@HEADS.register_module()
class DisHead(nn.Module):
    def __init__(self, in_features,
                 loss_dis=dict(type="CrossEntropyLoss",
                               use_sigmoid=False,
                               loss_weight=1.0)):
        super(DisHead, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.dis_net = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 1)
        )
        self.loss_dis = build_loss(loss_dis)

    def forward(self, inputs):

        x = inputs
        # x = torch.cat(x, dim=0)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dis_net(x)
        x = torch.sigmoid(x)

        return x

    def init_weights(self):
        for m in self.dis_net:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def loss(self, dis_score, targets):

        loss = self.loss_dis(dis_score, targets)

        return loss
