from abc import ABC
import torch.nn as nn


class RelPoseCriterion(nn.Module, ABC):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.q_loss = nn.MSELoss()
        self.t_loss = nn.MSELoss()

    def forward(self, q_gt, t_gt, q_est, t_est):
        t_loss = self.t_loss(t_est, t_gt)
        q_loss = self.q_loss(q_est, q_gt)

        loss_total = t_loss + self.alpha * q_loss
        return loss_total, t_loss.item(), q_loss.item()
