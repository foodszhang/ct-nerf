import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tinycudann as tcnn

from .unet import UNet

mlp_config = {
    "otype": "CutlassMLP",
    "activation": "LeakyReLU",
    "output_activation": "SoftPlus",
    "n_neurons": 256,
    "n_hidden_layers": 5,
}


def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    feat = feat.transpose(2, 3)  # [W, H]
    samples = torch.nn.functional.grid_sample(
        feat, uv, align_corners=True
    )  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


class DIF_Net(nn.Module):
    def __init__(
        self,
        num_views,
        mid_ch=4,
        image_encoding="unet",
    ):
        super().__init__()
        self.image_encoding = "unet"
        self.image_encoder = UNet(1, mid_ch)
        self.image_encoder.output_dim = mid_ch
        # self.mlp = DensityNetwork_debug(mid_ch * num_views)
        self.total_dim = mid_ch * num_views
        self.mlp = tcnn.Network(self.total_dim, 1, mlp_config)

    def forward(self, data, eval_npoint=10240, with_feat=False):
        # projection encoding
        if with_feat:
            proj_feats = [data["proj_feats"]]
        else:
            projs = data["projections"]  # B, M, C, W, H
            b, m, w, h = projs.shape
            projs = projs.reshape(b * m, 1, w, h)  # B', C, W, H
            proj_feats = self.image_encoder(projs)
            proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
            for i in range(len(proj_feats)):
                _, c_, w_, h_ = proj_feats[i].shape
                proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_)  # B, M, C, W, H

        # point-wise forward
        total_npoint = data["proj_pts"].shape[2]
        n_batch = int(np.ceil(total_npoint / eval_npoint))

        pred_list = []
        for i in range(n_batch):
            left = i * eval_npoint
            right = min((i + 1) * eval_npoint, total_npoint)
            p_pred = self.forward_points(
                proj_feats,
                {
                    "proj_pts": data["proj_pts"][..., left:right, :],
                    "pts": data["pts"][..., left:right, :],
                },
            )
            pred_list.append(p_pred)

        pred = torch.cat(pred_list, dim=2)
        return pred

    # points -> (10. 1024x10, 3)
    # proj -> (10, 1024x1, 2)
    def forward_points(self, proj_feats, data):
        n_view = proj_feats[0].shape[1]
        # 1. query view-specific features
        b = data["proj_pts"].shape[0]
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...]  # B, C, W, H

                p = data["proj_pts"][:, i, ...]  # B, N, 2
                p_feats = index_2d(feat, p)  # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.cat(p_list, dim=1)  # B, C, N, M

        proj_feats = p_feats

        p_feats = p_feats.permute(0, 2, 1)
        x = [self.mlp(p_feat) for p_feat in p_feats]
        x = torch.cat(x, dim=1)  # B, C, N, M
        # return outputs.view(b, -1)
        x = x.reshape(b, 1, -1)
        return x
