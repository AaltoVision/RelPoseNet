import torch
import torch.nn as nn
import torchvision.models as models


class RelPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.concat_layer = self._get_backbone()
        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        q_est = self.net_q_fc(self.dropout(self.concat_layer(feat)))
        t_est = self.net_t_fc(self.dropout(self.concat_layer(feat)))
        return q_est, t_est
