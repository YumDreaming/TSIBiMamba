# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:40:26 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:41:39 2021

@author: user
"""

from typing import Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class TripleDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module,reduction: Optional[str] = 'mean',max_iter=1000):
        super(TripleDomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.domain_discriminator_accuracy = None
        self.crition=nn.CrossEntropyLoss()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, f_u: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t,f_u), dim=0))
        d = self.domain_discriminator(f)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        d_label_u = torch.ones((f_u.size(0), 1)).to(f_u.device)+1
        label=torch.cat((d_label_s,d_label_t,d_label_u)).squeeze().long()
        return self.crition(d,label)