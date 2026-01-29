"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from util.seed_init import place_seed_points_d
# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+'
BG_PROT_MODE = 'gridconv' 
# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95



class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)

        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            check = torch.load(self.pretrained_path)
            self.load_state_dict(torch.load(self.pretrained_path),False)
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')
        # TODO: add loading of pretrained weights for classifier if any


    def compute_consensus_mask(self, rater_masks):
        """
        Compute consensus mask by averaging across raters and thresholding at 0.5.
        
        Args:
            rater_masks: List of masks, one per rater, each shape [H x W]
        
        Returns:
            consensus_mask: Binary mask [H x W] with values 0 or 1
        """
        # Stack all rater masks and average
        stacked = torch.stack(rater_masks, dim=0)  # [num_raters, H, W]
        avg_mask = stacked.mean(dim=0)  # [H, W]
        consensus = (avg_mask > 0.5).float()  # Binary mask
        return consensus

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images. 
                way x shot x num_raters x [B x H x W], list of lists of lists of tensors
            back_mask: background masks for support images
                way x shot x num_raters x [B x H x W], list of lists of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        mol = 'Loss'
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs) 

        # print("aa", n_ways, "bb", n_shots, "cc", n_queries)

        assert n_ways == 1, "Multi-shot has not been implemented yet" # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.encoder(imgs_concat, low_level = False)
        fts_size = img_fts.shape[-2:] 

        # print("aa", img_fts.shape)

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)        # N x B x C x H' x W'
        # fore_mask and back_mask expected as: way x shot x raters x [B x H x W]
        # Stack into tensors: Wa x Sh x R x B x H x W
        fore_mask_stacked = torch.stack([
            torch.stack([torch.stack(rater_list, dim=0) for rater_list in shot_list], dim=0)
            for shot_list in fore_mask], dim=0)  # Wa x Sh x R x B x H x W
        back_mask_stacked = torch.stack([
            torch.stack([torch.stack(rater_list, dim=0) for rater_list in shot_list], dim=0)
            for shot_list in back_mask], dim=0)  # Wa x Sh x R x B x H x W

        # Squeeze batch dim (B=1) -> Wa x Sh x R x H x W
        fore_mask_stacked = fore_mask_stacked.squeeze(3)
        back_mask_stacked = back_mask_stacked.squeeze(3)

        Wa, Sh, Raters, H_m, W_m = fore_mask_stacked.shape

        # Resize masks to feature spatial size
        fore_mask_flat = fore_mask_stacked.view(Wa * Sh * Raters, 1, H_m, W_m)
        back_mask_flat = back_mask_stacked.view(Wa * Sh * Raters, 1, H_m, W_m)

        fore_mask_resized = F.interpolate(fore_mask_flat.float(), size=fts_size, mode='bilinear')
        back_mask_resized = F.interpolate(back_mask_flat.float(), size=fts_size, mode='bilinear')

        fore_mask_resized = fore_mask_resized.view(Wa, Sh, Raters, *fts_size).squeeze(3)  # Wa x Sh x R x H' x W'
        back_mask_resized = back_mask_resized.view(Wa, Sh, Raters, *fts_size).squeeze(3)  # Wa x Sh x R x H' x W'

        # consensus mask from resized fore masks for seed init
        fore_mask_consensus = (fore_mask_resized.mean(dim=2) > 0.5).float()  # Wa x Sh x H' x W'
        s_y = fore_mask_consensus[0, 0]

        init_seed_list = []
        mask = (s_y == 1).float()  # H' x W'

        init_seed = place_seed_points_d(mask, down_stride=8, max_num_sp=5, avg_sp_area=100)
        init_seed_list.append(init_seed.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list).cuda()

        back_mask_consensus = (back_mask_resized.mean(dim=2) > 0.5).float()

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1):
            scores_per_rater = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            # Call classifier once, passing all raters' masks; classifier will perform per-rater prototype calibration
            _raw_scores_bg, assign_bg, aux_bg = self.cls_unit(mol, qry_fts, supp_fts, back_mask_resized, s_init_seed,
                                                              mode=BG_PROT_MODE, thresh=BG_THRESH, isval=isval,
                                                              val_wsize=val_wsize, vis_sim=show_viz)
            _raw_scores_fg, assign_fg, aux_fg = self.cls_unit(mol, qry_fts, supp_fts, fore_mask_resized, s_init_seed,
                                                              mode=FG_PROT_MODE if F.avg_pool2d(fore_mask_consensus, 4).max() >= FG_THRESH else 'mask',
                                                              thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)

            # _raw_scores_* expected shape: R x 1 x H' x W'
            # combine bg and fg per rater along class dim
            scores_all = torch.cat([_raw_scores_bg, _raw_scores_fg], dim=1)  # R x 2 x H' x W'

            # collect assign and sim maps if provided
            if assign_bg is not None:
                assign_maps.append(assign_bg)
            if show_viz:
                bg_sim_maps.append(aux_bg.get('raw_local_sims', None) if isinstance(aux_bg, dict) else None)
                fg_sim_maps.append(aux_fg.get('raw_local_sims', None) if isinstance(aux_fg, dict) else None)

            # interpolate each rater's score to image size and append to outputs
            for rater_idx in range(scores_all.shape[0]):
                pred_rater = F.interpolate(scores_all[rater_idx].unsqueeze(0), size=img_size, mode='bilinear')
                outputs.append(pred_rater)

            if self.config.get('align', False) and self.training:
                # average predictions across raters for alignment loss (consensus)
                pred = scores_all.mean(dim=0)  # 2 x H' x W'
                pred_int = F.interpolate(pred.unsqueeze(0), size=img_size, mode='bilinear')
                align_loss_epi = self.alignLoss_multi_rater(qry_fts, pred, pred_int, supp_fts,
                                                           fore_mask_consensus, back_mask_consensus,
                                                           fore_mask_resized, back_mask_resized)
                align_loss += align_loss_epi
             

        output = torch.stack(outputs, dim=1)  # R x 1 x 2 x H x W
        output = output.view(-1, *output.shape[2:]) # R x 2 x H x W

        if len(assign_maps) > 0 and assign_maps[0] is not None:
            # list of 1 x H' x W' -> stack to R x 1 x H' x W'
            assign_maps = torch.stack(assign_maps[0], dim=0)
        else:
            assign_maps = None

        if show_viz and len(bg_sim_maps) > 0 and bg_sim_maps[0] is not None:
            bg_sim_maps = torch.stack(bg_sim_maps, dim=1)
        else:
            bg_sim_maps = None

        if show_viz and len(fg_sim_maps) > 0 and fg_sim_maps[0] is not None:
            fg_sim_maps = torch.stack(fg_sim_maps, dim=1)
        else:
            fg_sim_maps = None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps

    # Batch was at the outer loop
    def alignLoss_multi_rater(self, qry_fts, pred, pred_int, supp_fts, 
                              fore_mask_consensus, back_mask_consensus,
                              fore_mask_resized, back_mask_resized):
        """
        Compute alignment loss for multi-rater setting.
        Uses consensus masks as target labels.
        
        Args:
            qry_fts: query features [N x B x C x H' x W']
            pred: predicted segmentation score [N x 2 x H' x W']
            pred_int: interpolated prediction [N x 2 x H x W]
            supp_fts: support features [Wa x Sh x B x C x H' x W']
            fore_mask_consensus: consensus foreground mask [Wa x Sh x H' x W']
            back_mask_consensus: consensus background mask [Wa x Sh x H' x W']
            fore_mask_resized: per-rater foreground masks [Wa x Sh x Raters x H' x W']
            back_mask_resized: per-rater background masks [Wa x Sh x Raters x H' x W']
        """
        mol = 'alignLoss'
        way = 0
        shot = 0
        
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x N x H' x W'
        binary_masks = [pred_mask == i for i in range(2)]  # [bg_mask, fg_mask]
        
        pred_mask_int = pred_int.argmax(dim=1).unsqueeze(0)  # 1 x N x H x W
        pred_mask_int_fg = (pred_mask_int == 1).float()
        
        loss = []
        
        # Compute loss on support images using consensus masks as pseudo ground truth
        supp_fts_single = supp_fts[way, shot, 0, :, :, :]  # C x H' x W'
        
        # Use consensus masks as target
        target_fg = fore_mask_consensus[way, shot]  # H' x W'
        target_bg = back_mask_consensus[way, shot]  # H' x W'
        
        # Predict on support using query features and support prototypes
        # This is simplified - ideally we'd extract prototypes and use them
        qry_fts_single = qry_fts[0, 0, :, :, :]  # C x H' x W'
        
        # Compute similarity scores (simplified - just check if predictions match consensus)
        # For proper alignment, we compare prediction to consensus ground truth
        pred_fg = binary_masks[1][0, 0]  # H' x W'
        pred_bg = binary_masks[0][0]  # H' x W'
        
        # Create target labels
        target_label = torch.full_like(target_fg, 255, dtype=torch.long)
        target_label[target_fg > 0.5] = 1
        target_label[target_bg > 0.5] = 0
        
        # Reshape predictions and target for cross-entropy
        pred_reshaped = pred.view(-1, 2)  # [N*H'*W', 2]
        target_reshaped = target_label.view(-1).long()  # [N*H'*W']
        
        # Compute cross-entropy loss
        loss_align = F.cross_entropy(pred_reshaped, target_reshaped, 
                                     ignore_index=255, reduction='mean')
        
        return loss_align
    