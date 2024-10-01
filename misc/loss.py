import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalSoftTargetCrossEntropy(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalSoftTargetCrossEntropy, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(8)
            self.alpha[0] =1
            self.alpha[1] =0.7
            self.alpha[2] =2
            self.alpha[3] =2.5
            self.alpha[4] =3
            self.alpha[5] =4
            self.alpha[6] =2
            self.alpha[7] = 4

            # 默认所有类别的权重都是1
        else:
            self.alpha = torch.tensor(alpha)  # 从外部传入的alpha值，应该是一个长度为7的列表或张量
        self.gamma = gamma

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(x, dim=1)
        pt = (target * p).sum(dim=1, keepdim=True)

        # 根据target计算每个样本的alpha值
        class_weights = self.alpha[target.argmax(dim=1)].cuda()
        weight = class_weights * (1. - pt).squeeze().pow(self.gamma)
        fl_soft = - weight * (target * torch.log(p + 1e-14)).sum(dim=1)
        return fl_soft.mean()
