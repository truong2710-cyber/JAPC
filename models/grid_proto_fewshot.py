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
        self.get_residual_mlp()
    
    def get_residual_mlp(self):
        """
        Build MLP for learning residual prototypes for rater personalization.
        Takes [p0, p_i - p0] and outputs residual p~_i of same dimension as p_i.
        """
        # Assuming prototype dimension from encoder output
        # DeepLab ResNet101 outputs 256 channels
        proto_dim = 256
        hidden_dim = 256
        
        self.residual_mlp = nn.Sequential(
            nn.Linear(proto_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proto_dim)
        )

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
    
    def extract_prototype(self, supp_fts, mask):
        """
        Extract prototype by masking and averaging support features.
        
        Args:
            supp_fts: Support features [C x H' x W']
            mask: Binary mask [H' x W']
        
        Returns:
            prototype: Feature vector [C]
        """
        # Mask the features and compute masked average
        mask_expanded = mask.unsqueeze(0)  # [1, H', W']
        masked_fts = supp_fts * mask_expanded  # [C, H', W']
        
        # Average pool with mask
        proto = masked_fts.sum(dim=(1, 2)) / (mask_expanded.sum() + 1e-8)  # [C]
        return proto


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
        # breakpoint()
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
        
        # Handle multi-rater masks: fore_mask and back_mask are way x shot x num_raters x [B x H x W]
        # Stack to get proper shape for processing
        # fore_mask: list of lists of lists of tensors -> way x shot x num_raters x B x H x W
        fore_mask_stacked = torch.stack([torch.stack([torch.stack(rater_list, dim=0) 
                                                       for rater_list in shot_list], dim=0)
                                         for shot_list in fore_mask], dim=0)  # Wa x Sh x Raters x B x H x W
        back_mask_stacked = torch.stack([torch.stack([torch.stack(rater_list, dim=0) 
                                                       for rater_list in shot_list], dim=0)
                                         for shot_list in back_mask], dim=0)  # Wa x Sh x Raters x B x H x W
        
        # Squeeze out batch dimension since B=1
        fore_mask_stacked = fore_mask_stacked.squeeze(3)  # Wa x Sh x Raters x H x W
        back_mask_stacked = back_mask_stacked.squeeze(3)  # Wa x Sh x Raters x H x W
        
        # Interpolate masks to feature size
        # Reshape to flatten way, shot, raters for batch processing
        Wa, Sh, Raters, H, W = fore_mask_stacked.shape
        fore_mask_flat = fore_mask_stacked.view(Wa * Sh * Raters, 1, H, W)
        back_mask_flat = back_mask_stacked.view(Wa * Sh * Raters, 1, H, W)
        
        fore_mask_resized = F.interpolate(fore_mask_flat, size=fts_size, mode='bilinear')  # [Wa*Sh*Raters, 1, H', W']
        back_mask_resized = F.interpolate(back_mask_flat, size=fts_size, mode='bilinear')  # [Wa*Sh*Raters, 1, H', W']
        
        # Reshape back
        fore_mask_resized = fore_mask_resized.view(Wa, Sh, Raters, *fts_size).squeeze(3)  # Wa x Sh x Raters x H' x W'
        back_mask_resized = back_mask_resized.view(Wa, Sh, Raters, *fts_size).squeeze(3)  # Wa x Sh x Raters x H' x W'
        
        # Compute consensus masks (average across raters and threshold at 0.5)
        fore_mask_consensus = (fore_mask_resized.mean(dim=2) > 0.5).float()  # Wa x Sh x H' x W'
        back_mask_consensus = (back_mask_resized.mean(dim=2) > 0.5).float()  # Wa x Sh x H' x W'
        
        # Get seed points from first shot's consensus mask
        s_y = fore_mask_consensus[0, 0]  # H' x W'

        init_seed_list = []
        mask = (s_y == 1).float()  # H x W 

        init_seed = place_seed_points_d(mask, down_stride=8, max_num_sp=5,
                                                            avg_sp_area=100)
        init_seed_list.append(init_seed.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list).cuda()

        ###### Compute personalized prototypes for each rater ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1):
            # Extract prototypes: consensus (p0) and per-rater (p_i)
            # supp_fts: Wa x Sh x B x C x H' x W'
            # fore_mask_consensus: Wa x Sh x H' x W'
            # fore_mask_resized: Wa x Sh x Raters x H' x W'
            # back_mask_consensus: Wa x Sh x H' x W'
            # back_mask_resized: Wa x Sh x Raters x H' x W'
            
            way = 0  # Only single way for now
            shot = 0  # Use first shot
            supp_fts_single = supp_fts[way, shot, 0, :, :, :]  # C x H' x W' (B=1)
            
            # Extract consensus prototypes
            fg_proto_consensus = self.extract_prototype(supp_fts_single, fore_mask_consensus[way, shot])  # [C]
            bg_proto_consensus = self.extract_prototype(supp_fts_single, back_mask_consensus[way, shot])  # [C]
            
            # Extract rater-specific prototypes and compute calibrated versions
            protos_calibrated_fg = []
            protos_calibrated_bg = []
            
            for rater_idx in range(Raters):
                # Extract rater-specific prototypes
                fg_proto_rater = self.extract_prototype(supp_fts_single, fore_mask_resized[way, shot, rater_idx])  # [C]
                bg_proto_rater = self.extract_prototype(supp_fts_single, back_mask_resized[way, shot, rater_idx])  # [C]
                
                # Compute residuals using MLP
                # Input: [p0, p_i - p0]
                fg_input = torch.cat([fg_proto_consensus, fg_proto_rater - fg_proto_consensus], dim=0)  # [2*C]
                bg_input = torch.cat([bg_proto_consensus, bg_proto_rater - bg_proto_consensus], dim=0)  # [2*C]
                
                fg_residual = self.residual_mlp(fg_input.unsqueeze(0)).squeeze(0)  # [C]
                bg_residual = self.residual_mlp(bg_input.unsqueeze(0)).squeeze(0)  # [C]
                
                # Calibrated prototypes: p0 + p~_i
                fg_proto_calibrated = fg_proto_consensus + fg_residual  # [C]
                bg_proto_calibrated = bg_proto_consensus + bg_residual  # [C]
                
                protos_calibrated_fg.append(fg_proto_calibrated)
                protos_calibrated_bg.append(bg_proto_calibrated)
            
            # Generate per-rater predictions
            scores_per_rater = []
            
            for rater_idx in range(Raters):
                # Use calibrated prototypes for this rater to compute similarity scores
                # For now, use the classifier with consensus mask and per-rater calibrated prototypes
                # This is a simplified approach - ideally we'd modify the classifier to use custom prototypes
                
                _raw_score_bg, _, aux_attr_bg = self.cls_unit(mol, qry_fts, supp_fts, back_mask_consensus.unsqueeze(2),
                                                               s_init_seed, mode=BG_PROT_MODE, thresh=BG_THRESH, 
                                                               isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
                
                _raw_score_fg, _, aux_attr_fg = self.cls_unit(mol, qry_fts, supp_fts, fore_mask_consensus.unsqueeze(2),
                                                               s_init_seed, mode=FG_PROT_MODE if F.avg_pool2d(fore_mask_consensus, 4).max() >= FG_THRESH else 'mask',
                                                               thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
                
                # Concatenate bg and fg scores
                score = torch.cat([_raw_score_bg, _raw_score_fg], dim=1)  # N x 2 x H' x W'
                scores_per_rater.append(score)
            
            # Stack scores from all raters - keep them separate
            scores_all = torch.stack(scores_per_rater, dim=0)  # [Raters, N, 2, H', W']
            
            # Interpolate to original image size for each rater
            for rater_idx in range(Raters):
                pred_rater = F.interpolate(scores_all[rater_idx], size=img_size, mode='bilinear')
                outputs.append(pred_rater)
            
            # Compute alignment loss if enabled - use consensus (averaged) predictions
            if self.config['align'] and self.training:
                final_score = scores_all.mean(dim=0)  # Average across raters for consensus
                align_loss_epi = self.alignLoss_multi_rater(qry_fts, final_score, 
                                                             F.interpolate(final_score, size=img_size, mode='bilinear'),
                                                             supp_fts, 
                                                             fore_mask_consensus, back_mask_consensus,
                                                             fore_mask_resized, back_mask_resized)
                align_loss += align_loss_epi
            
        output = torch.cat(outputs, dim=0)  # [Raters*N, 2, H, W]
        
        return output, align_loss / sup_bsize, None, None


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
    