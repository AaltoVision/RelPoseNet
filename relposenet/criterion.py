from abc import ABC
import torch.nn as nn


class RelPoseCriterion(nn.Module, ABC):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.q_loss = nn.MSELoss()
        self.t_loss = nn.MSELoss()

    def forward(self, q_gt, t_gt, q_est, t_est):
        loss_total = self.t_loss(t_gt, t_est) + self.alpha * self.q_loss(q_gt, q_est)
        return loss_total
