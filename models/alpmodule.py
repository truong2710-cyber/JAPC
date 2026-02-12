"""
ALPModule (multi-rater version of the 1st ALPModule)
- Supports multi-rater support masks: sup_y = [way(1), shot, rater, h, w]
- Adds optional prototype calibration via:
  (1) proto_attention: p_i <- p_i * att(p_i, p_i - p0)
  (2) residual_mlp:    p_i <- p_i + MLP([p0, p_i - p0])
- Keeps original proto extraction logic from the 1st file for:
  mode in {'mask', 'gridconv', 'gridconv+'}
- Outputs per-rater logits (stacked along dim=0): [R, 1, H, W]
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode="bilinear",
                 use_mlp=True, use_proto_attention=True):
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode

        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)

        self.use_mlp = use_mlp
        self.use_proto_attention = use_proto_attention

        if self.use_mlp:
            self.residual_mlp_fg = self.get_residual_mlp()
            self.residual_mlp_bg = self.get_residual_mlp()

        if self.use_proto_attention:
            self.proto_attention = self.get_proto_attention()

        # for external access (optional)
        self.proto_calib_loss = None

    # -------------------------
    #  Calibration submodules
    # -------------------------
    def get_residual_mlp(self):
        """
        Residual MLP: input [p0, delta] -> residual (C)
        """
        proto_dim = 256
        hidden_dim = 256

        mlp = nn.Sequential(
            nn.Linear(proto_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proto_dim),
        )

        # init last layer to zero so residual starts at 0
        last = mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        return mlp

    def get_proto_attention(self):
        """
        Scalar attention: weight = sigmoid( <Q(p_i), K(delta)> / sqrt(C) )
        """
        proto_dim = 256

        class ProtoAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q = nn.Linear(dim, dim)
                self.k = nn.Linear(dim, dim)

            def forward(self, p_i, delta):
                q = self.q(p_i)
                k = self.k(delta)
                att_raw = (q * k).sum(dim=1, keepdim=True) / math.sqrt(q.size(1))
                return torch.sigmoid(att_raw)

        att = ProtoAttention(proto_dim)
        # keep biases near 0
        for lin in [att.q, att.k]:
            if isinstance(lin, nn.Linear) and lin.bias is not None:
                nn.init.zeros_(lin.bias)
        return att

    # -------------------------
    #  Utils
    # -------------------------
    def safe_norm(self, x, p=2, dim=1, eps=1e-4):
        x_norm = torch.norm(x, p=p, dim=dim)
        x_norm = torch.max(x_norm, torch.ones_like(x_norm, device=x_norm.device) * eps)
        x = x.div(x_norm.unsqueeze(1).expand_as(x))
        return x

    # -------------------------
    #  Proto extraction
    # -------------------------
    def get_proto(self, qry, sup_x, sup_y, mode, thresh, isval=False, val_wsize=None):
        """
        Returns prototypes in the same *shape semantics* as the 1st ALPModule:
          - mode='mask'     -> proto: [1, C]   (mean over shots)
          - mode='gridconv' -> protos: [N, C]  (local protos)
          - mode='gridconv+'-> protos: [N, C]  (local + per-shot global concatenated then normed)
        """
        # qry:   [1, C, H, W]
        # sup_x: [S, C, H, W]
        # sup_y: [S, 1, H, W]
        if mode == "mask":
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # [S, C]
            proto = proto.mean(dim=0, keepdim=True)  # [1, C]
            return proto

        elif mode == "gridconv":
            nch = qry.shape[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)  # [1, S, HW, C]
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)  # [1, 1, S*HW, C]

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)  # [1,1,S*HW]

            protos = n_sup_x[sup_y_g > thresh, :]  # [N, C]
            return protos

        elif mode == "gridconv+":
            nch = qry.shape[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)  # [1,S,HW,C]
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)  # [1,1,S*HW,C]

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]  # local: [N, C]

            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # [S, C]
            allp = torch.cat([protos, glb_proto], dim=0)  # [N+S, C]
            return allp

        else:
            raise NotImplementedError(f"Unsupported mode in 1st ALP: {mode}")

    # -------------------------
    #  Forward (multi-rater wrapper + calibration)
    # -------------------------
    def forward(self, mol, qry, sup_x, sup_y, s_init_seed, mode, thresh,
                isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """
        Multi-rater API (aligned with your multi-rater 2nd version):
          qry:   [way(1), nb(1), C, H, W]
          sup_x: [way(1), shot, nb(1), C, H, W]
          sup_y: [way(1), shot, rater, H, W]
        Output:
          pred_stack: [R, 1, H, W]
        Notes:
          - mol, s_init_seed are accepted for signature compatibility but unused here (1st ALP has no seeds path).
        """
        # squeeze to match the original 1st ALP internal shapes
        qry = qry.squeeze(1)                      # [1, C, H, W]
        sup_x = sup_x.squeeze(0).squeeze(1)       # [S, C, H, W]
        sup_y = sup_y.squeeze(0)                  # [S, R, H, W]

        if sup_y.dim() == 3:
            # fallback: [S,H,W] -> treat as single rater
            sup_y = sup_y.unsqueeze(1)

        S, R, H, W = sup_y.shape

        # 1) extract per-rater prototypes (using 1st ALP logic)
        per_rater_protos = []
        per_rater_counts = []
        for r in range(R):
            sup_y_r = sup_y[:, r:r+1, :, :]  # [S,1,H,W]
            proto_r = self.get_proto(qry, sup_x, sup_y_r, mode, thresh, isval, val_wsize)
            per_rater_protos.append(proto_r)
            per_rater_counts.append(proto_r.shape[0] if proto_r.dim() == 2 else 1)

        per_rater_protos = torch.cat(per_rater_protos, dim=0)  # [total_proto, C]

        # 2) consensus prototype p0 (mean over raters mask)
        consensus_mask = sup_y.mean(dim=1, keepdim=True)  # [S,1,H,W]
        p0 = self.get_proto(qry, sup_x, consensus_mask, mode, thresh, isval, val_wsize)
        p0 = torch.mean(p0, dim=0, keepdim=True)  # [1,C] regardless of proto count

        # 3) calibrate prototypes
        p0_rep = p0.expand(per_rater_protos.size(0), -1)      # [total_proto, C]
        delta = per_rater_protos - p0_rep                     # [total_proto, C]

        if self.use_proto_attention:
            w = self.proto_attention(per_rater_protos, delta)  # [total_proto,1]
            calibrated_protos = per_rater_protos * w
        elif self.use_mlp:
            # choose fg/bg MLP based on mode
            residual_mlp = self.residual_mlp_fg if mode in ["mask", "gridconv+"] else self.residual_mlp_bg
            mlp_in = torch.cat([p0_rep, delta], dim=1)         # [total_proto, 2C]
            calibrated_protos = per_rater_protos + residual_mlp(mlp_in)
        else:
            calibrated_protos = per_rater_protos

        # store calib loss for training hooks
        try:
            self.proto_calib_loss = torch.mean((per_rater_protos - calibrated_protos) ** 2)
        except Exception:
            self.proto_calib_loss = None

        # split back to per-rater lists
        per_rater_calibrated = []
        idx = 0
        for c in per_rater_counts:
            per_rater_calibrated.append(calibrated_protos[idx:idx + c])
            idx += c

        # 4) per-rater prediction (same style as your multi-rater 2nd: conv2d + softmax-weighted sum)
        qry_n = self.safe_norm(qry)
        per_rater_preds = []
        per_rater_assigns = []
        raw_sims_list = []

        for cal in per_rater_calibrated:
            if cal.numel() == 0:
                per_rater_preds.append(torch.zeros_like(qry[:, :1, :, :]))
                per_rater_assigns.append(torch.zeros(1, qry.size(2), qry.size(3), device=qry.device).float())
                raw_sims_list.append(None)
                continue

            if mode == "mask":
                pred = F.cosine_similarity(qry, cal[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]
                raw_sims_list.append(pred)
                pred = pred.unsqueeze(1)  # [1,1,H,W]
            else:
                pro_n = self.safe_norm(cal)  # [N,C]
                dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20  # [1, N, H, W]
                pred = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)  # [1,1,H,W]
                assign = dists.argmax(dim=1).float().detach()  # [1,H,W]

                raw_sims_list.append(dists.detach() if vis_sim else None)
                per_rater_assigns.append(assign)
            
            per_rater_preds.append(pred)
            
        pred_stack = torch.cat(per_rater_preds, dim=0)  # [R,1,H,W]

        if mode == "mask":
            vis_dict = {"proto_assign": None}
        else:
            vis_dict = {"proto_assign": torch.cat(per_rater_assigns, dim=0)}
        if self.proto_calib_loss is not None:
            vis_dict["proto_calib_loss"] = self.proto_calib_loss
        if vis_sim:
            vis_dict["raw_local_sims"] = raw_sims_list

        return pred_stack, per_rater_assigns, vis_dict
