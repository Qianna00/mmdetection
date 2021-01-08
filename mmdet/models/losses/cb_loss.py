import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES, build_loss


@LOSSES.register_module()
class CBLoss(nn.Module):
    def __init__(self, beta=0.9999, gamma=2.0, loss_type="focal"):
        super(CBLoss, self).__init__()
        self.beta = torch.tensor(beta).cuda()
        self.gamma = torch.tensor(gamma).cuda()
        self.loss_type = loss_type

    def forward(self, labels, logits, samples_per_cls, no_of_classes):
        # effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        # weights = (1.0 - self.beta) / np.array(effective_num)
        # weights = weights / np.sum(weights) * no_of_classes
        print(samples_per_cls)
        zero_class_index = samples_per_cls == 0
        samples_per_cls[zero_class_index] = 1
        effective_num = 1.0 - torch.pow(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        print(weights.size(), weights)
        # weights = weights / torch.sum(weights[zero_class_index]) * (no_of_classes - weights[zero_class_index].shape[0])
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels, no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

    def focal_loss(self, labels, logits, alpha, gamma):
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
