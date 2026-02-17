"""Util functions
Extended from original PANet code
TODO: move part of dataset configurations to data_utils
"""
import random
import torch
import numpy as np
import operator

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'SABS': {
        'pa_all': set( [1,2,3,6]  ),
        0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
        1: set( [2,3] ), # lower_abdomen
    },
    'C0': {
        'pa_all': set(range(1, 4)),
        0: set([2,3]),
        1: set([1,3]),
        2: set([1,2]),
    },
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]), # lower_abdomen
    },
}

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()

def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)

def compose_wt_simple(is_wce, data_name):
    """
    Weights for cross-entropy loss
    """
    if is_wce:
        if data_name in ['SABS', 'SABS_Superpix', 'C0', 'C0_Superpix', 'CHAOST2', 'CHAOST2_Superpix','CMR_Superpix','CMR', 'CURVAS','CURVAS_Superpix', 'CURVASPDAC','CURVASPDAC_Superpix'] or data_name.startswith('QUBIQ'):
            return torch.FloatTensor([0.05, 1.0]).cuda()
        else:
            raise NotImplementedError
    else:
        return torch.FloatTensor([1.0, 1.0]).cuda()


class CircularList(list):
    """
    Helper for spliting training and validation scans
    Originally: https://stackoverflow.com/questions/8951020/pythonic-circular-list/8951224
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)


def visualize_multi_rater(support_images, support_masks, query_images, query_labels, save_path=None):
    """
    Visualize support images with multi-rater fg masks overlayed, and query images with multi-rater labels.
    
    Args:
        support_images: List of support images (C x H x W), each as tensor
        support_masks: List of lists - each inner list contains rater mask dicts with 'fg_mask' and 'bg_mask'
        query_images: List of query images (C x H x W), each as tensor
        query_labels: List of lists - n_queries x raters x (H x W) label tensors
        save_path: Optional path to save visualization
    
    Returns:
        visualization_image: numpy array combining support and query visualizations
    """
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    
    # Color palette for different raters
    rater_colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
    ]
    
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor
    
    def normalize_image(img):
        """Normalize image to 0-255 range"""
        img = tensor_to_numpy(img)
        
        # Handle CHW format -> HWC
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        # Remove singleton dimensions
        while img.ndim > 3 and (img.shape[0] == 1 or img.shape[-1] == 1):
            img = np.squeeze(img, axis=0) if img.shape[0] == 1 else np.squeeze(img, axis=-1)
        
        # Convert single channel to 3 channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Normalize to 0-1 range
        if img.max() > 1:
            img = img / (img.max() + 1e-8)
        img = np.clip(img, 0, 1)
        
        return (img * 255).astype(np.uint8)
    
    # Process support images
    # Process support images
    support_viz = []
    for shot_idx, shot_masks in enumerate(support_masks):
        img = normalize_image(support_images[shot_idx])
        
        # Ensure image is HWC format with 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Ensure correct shape
        assert img.ndim == 3 and img.shape[-1] == 3, f"Image shape should be (H, W, 3), got {img.shape}"
        
        # Overlay each rater's fg mask with different color
        img_with_masks = img.copy().astype(np.float32)
        
        for rater_idx, mask_dict in enumerate(shot_masks):
            fg_mask = tensor_to_numpy(mask_dict['fg_mask'])
            
            # Squeeze to 2D (H, W)
            while fg_mask.ndim > 2:
                fg_mask = np.squeeze(fg_mask)
            
            # Ensure mask is 2D
            assert fg_mask.ndim == 2, f"Mask should be 2D, got shape {fg_mask.shape}"
            
            # Ensure mask is float in [0, 1]
            fg_mask = fg_mask.astype(np.float32)
            fg_mask_max = fg_mask.max()
            if fg_mask_max > 1:
                fg_mask = fg_mask / (fg_mask_max + 1e-8)
            fg_mask_binary = (fg_mask > 0.5)  # Boolean mask
            
            # Skip if no mask pixels
            if not fg_mask_binary.any():
                continue
            
            # Resize mask if needed to match image
            if fg_mask.shape != img_with_masks.shape[:2]:
                import cv2
                fg_mask_binary = cv2.resize((fg_mask_binary.astype(np.uint8) * 255), 
                                           (img_with_masks.shape[1], img_with_masks.shape[0])) > 127

            # Apply color overlay
            color = rater_colors[rater_idx % len(rater_colors)]
            for c in range(3):
                img_with_masks[fg_mask_binary, c] = img_with_masks[fg_mask_binary, c] * 0.5 + color[c] * 0.5
        
        img_with_masks = np.clip(img_with_masks, 0, 255).astype(np.uint8)
        support_viz.append(img_with_masks)
    
    # Process query images with multi-rater labels
    query_viz = []
    for query_idx, query_img in enumerate(query_images):
        img = normalize_image(query_img)
        
        # Ensure image is HWC format with 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Ensure correct shape
        assert img.ndim == 3 and img.shape[-1] == 3, f"Image shape should be (H, W, 3), got {img.shape}"
        
        # Overlay all rater labels for this query
        img_with_labels = img.copy().astype(np.float32)
        query_labels_for_raters = query_labels[query_idx]  # List of labels for each rater
        
        # Iterate through all raters for this query
        for rater_idx, query_label in enumerate(query_labels_for_raters):
            query_label = tensor_to_numpy(query_label)
            
            # Squeeze to 2D (H, W)
            while query_label.ndim > 2:
                query_label = np.squeeze(query_label)
            
            # Ensure label is 2D
            assert query_label.ndim == 2, f"Label should be 2D, got shape {query_label.shape}"
            
            # Ensure label is float in [0, 1]
            query_label = query_label.astype(np.float32)
            label_max = query_label.max()
            if label_max > 1:
                query_label = query_label / (label_max + 1e-8)
            
            query_label_binary = (query_label > 0.5)  # Boolean mask
            
            # Skip if no label pixels
            if not query_label_binary.any():
                continue
            
            # Resize label if needed to match image
            if query_label.shape != img_with_labels.shape[:2]:
                import cv2
                query_label_binary = cv2.resize((query_label_binary.astype(np.uint8) * 255), 
                                               (img_with_labels.shape[1], img_with_labels.shape[0])) > 127
            
            # Apply color overlay for this rater
            color = rater_colors[rater_idx % len(rater_colors)]
            for c in range(3):
                img_with_labels[query_label_binary, c] = img_with_labels[query_label_binary, c] * 0.5 + color[c] * 0.5
        
        img_with_labels = np.clip(img_with_labels, 0, 255).astype(np.uint8)
        query_viz.append(img_with_labels)
    
    # Combine all visualizations
    # Concatenate horizontally: all support shots, then all query images
    if support_viz:
        support_row = np.hstack(support_viz)
    else:
        support_row = None
    
    if query_viz:
        query_row = np.hstack(query_viz)
    else:
        query_row = None
    
    if support_row is not None and query_row is not None:
        # Resize query row to match support row height if needed
        if support_row.shape[0] != query_row.shape[0]:
            scale = support_row.shape[0] / query_row.shape[0]
            query_row = cv2.resize(query_row, (int(query_row.shape[1] * scale), support_row.shape[0]))
        
        viz_image = np.vstack([support_row, query_row])
    elif support_row is not None:
        viz_image = support_row
    else:
        viz_image = query_row
    
    # Save if path provided
    if save_path is not None:
        cv2.imwrite(save_path, viz_image)
    
    return viz_image


