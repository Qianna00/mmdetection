import torch
import torch.nn as nn
from torch.autograd.function import Function
from ..builder import LOSSES, build_loss

import pdb


@LOSSES.register_module()
class DiscCentroidsLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim, 14, 14))
        # print(self.centroids.data)
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        # print(self.centroids.requires_grad)

    def forward(self, feat, label):
        batch_size = feat.size(0)

        #############################
        # calculate attracting loss #
        #############################
        # print(self.centroids.data)

        # feat = feat.view(batch_size, -1)
        # feat = feat.view()

        # To check the dim of centroids and features
        if feat.size()[1:] != self.centroids.size()[1:]:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.centroids.size()[1:], feat.size()[1:]))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        """loss_attract = self.disccentroidslossfunc(feat.clone(), label, self.centroids.clone(),
                                                  batch_size_tensor).squeeze()"""
        loss_attract = self.loss_attract(feat.clone(), label, batch_size)

        # print("feat:", feat)

        ############################
        # calculate repelling loss #
        #############################

        """distmat = torch.pow(feat.clone(), 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes, 14, 14) + \
                  torch.pow(self.centroids.clone(), 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).permute(1,0,2,3)
        distmat.addmm_(1, -2, feat.clone(), self.centroids.clone().t())"""
        """distmat = torch.pow(feat.clone().sum(dim=1, keepdim=True).expand(batch_size, self.num_classes, 14, 14), 2) + \
                            torch.pow(self.centroids.clone().sum(dim=1, keepdim=True).expand(self.num_classes, batch_size, 14, 14).permute(1, 0, 2, 3), 2)
        distmat = distmat - 2 * torch.matmul(feat.clone().permute(2 ,3 ,0 ,1), self.centroids.clone().permute(2, 3, 1, 0)).permute(2, 3, 0, 1)"""
        """distmat = (feat.clone().sum(dim=1, keepdim=True).expand(batch_size, self.num_classes, 14, 14)-
                   self.centroids.clone().sum(dim=1, keepdim=True).expand(self.num_classes, batch_size, 14, 14).permute(1, 0, 2, 3)).pow(2)"""
        distmat = torch.matmul(feat.clone().permute(2, 3, 0, 1), self.centroids.clone().permute(2, 3, 1, 0)).abs()
        norm_feat = torch.norm(feat.clone().permute(2, 3, 0, 1), p=2, dim=(2, 3), keepdim=True)
        norm_centroids = torch.norm(self.centroids.clone().permute(2, 3, 1, 0), p=2, dim=(2, 3), keepdim=True)
        distmat = (distmat / torch.matmul(norm_feat, norm_centroids)).permute(2, 3, 0, 1)

        classes = torch.arange(self.num_classes).long().cuda()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        distmat_neg = distmat
        distmat_neg[mask, :, :] = 0.0
        # print("distmat_neg:", distmat_neg.sum())
        # margin = 50.0
        margin = 0.2
        loss_repel = torch.clamp(margin - distmat_neg.sum(), 0.0, 1e6)

        # print(loss_repel, loss_attract)
        loss_attract = 0.01 * loss_attract
        loss_repel = 0.1 * loss_repel
        # loss = loss_attract + 0.01 * loss_repel

        return loss_attract, loss_repel

    def loss_attract(self, feats, label, batch_size):
        centroids_batch = self.centroids.clone().index_select(0, label.long())
        loss_attract = (feats - centroids_batch).pow(2).sum() / 2.0 / (batch_size * 14 * 14)
        return loss_attract


class DiscCentroidsLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / (batch_size * 14 * 14)

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())
        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids / counts.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(grad_centroids.size())
        return - grad_output * diff / (batch_size * 14 * 14), None, grad_centroids / (batch_size * 14 * 14), None

