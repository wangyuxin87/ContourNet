import torch
from torch import nn
from torch.nn import functional as F

def Focal_Loss(pred, gt):
        # print('yes!!')



        ce = nn.CrossEntropyLoss()
        alpha = 0.25
        gamma = 2
        # logp = ce(input, target)
        p = torch.sigmoid(pred)

        loss = -alpha * (1 - p) ** gamma * (gt * torch.log(p)) - \
               (1 - alpha) * p ** gamma * ((1 - gt) * torch.log(1 - p))

        return loss.mean()











        # pred =torch.sigmoid(pred)
        # pos_inds = gt.eq(1).float()
        # neg_inds = gt.lt(1).float()
        #
        # loss = 0
        #
        # pos_loss = torch.log(pred + 1e-10) * torch.pow(pred, 2) * pos_inds
        # # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        # neg_loss = torch.log(1 - pred) * torch.pow(1 - pred, 2) * neg_inds
        #
        # num_pos = pos_inds.float().sum()
        # num_neg = neg_inds.float().sum()
        #
        # pos_loss = pos_loss.sum()
        # neg_loss = neg_loss.sum()
        #
        # if num_pos == 0:
        #     loss = loss - neg_loss
        # else:
        #     # loss = loss - (pos_loss + neg_loss) / (num_pos)
        #     loss = loss - (pos_loss + neg_loss )
        # return loss * 5




        # if weight is not None and weight.sum() > 0:
        #     return (losses * weight).sum() / weight.sum()
        # else:
        #     assert losses.numel() != 0
        #     return losses.mean()