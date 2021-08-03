import torch
import torch.nn as nn
from torch.autograd.function import Function
from ..builder import LOSSES, build_loss



@LOSSES.register_module
class ContrastiveLoss(nn.Module):
    """Head for contrastive learning."""

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """
        :param pos:
        :param neg:
        :return:
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        # losses = dict()
        # losses['loss'] = self.criterion(logits, labels)
        losses = self.criterion(logits, labels)
        return losses
