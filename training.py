"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple, visualize_multi_rater
from util.metric import Metric
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_ssl_upload import ex
import tqdm
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

def visualize_batch(sample_batched, query_images, query_labels, i_iter, _run, _log):
    """
    Visualize multi-rater support and query data
    """
    try:
        # Prepare data for visualization (extract from batch)
        support_images = [[shot.cpu() for shot in way]
                          for way in sample_batched['support_images']]
        
        # Convert to CPU and prepare for viz
        support_imgs_cpu = [img.detach().cpu() for img in support_images[0]]
        support_masks_cpu = [[{k: v.detach().cpu() for k, v in mask_dict.items()} 
                             for mask_dict in shot_masks] 
                            for shot_masks in sample_batched['support_masks']]
        query_imgs_cpu = [img.detach().cpu() for img in query_images]
        # query_labels_cpu: n_queries x raters x [H x W]
        query_labels_cpu = [[label.detach().cpu() for label in query_rater_labels] 
                            for query_rater_labels in query_labels]
        
        # Create visualization
        viz_image = visualize_multi_rater(
            support_imgs_cpu,
            support_masks_cpu,
            query_imgs_cpu,
            query_labels_cpu,
            save_path=os.path.join(f'{_run.observers[0].dir}/snapshots', f'viz_{i_iter + 1}.png')
        )
        _log.info(f'Saved visualization at iteration {i_iter + 1}')
    except Exception as e:
        _log.warning(f'Visualization failed at iteration {i_iter + 1}: {e}')


def visualize_pred_and_label(support_images, support_masks, query_images, query_pred_reshaped, query_labels, i_iter, _run, _log):
    """
    Create a single figure showing for the first query (index 0) and each rater:
    - support image (first support shot)
    - support label (fg mask)
    - query image
    - query prediction (argmax over channels)
    - query label (rater)

    The figure is saved to the experiment snapshots directory.
    """
    # pick first query for visualization
    q_idx = 0

    num_raters = query_pred_reshaped.shape[0]

    # Prepare support images and masks (CPU tensors)
    support_imgs_cpu = [img.detach().cpu() for img in support_images[0]] if support_images and len(support_images) > 0 else []
    support_masks_cpu = [[{k: v.detach().cpu() for k, v in mask_dict.items()} for mask_dict in shot_masks]
                            for shot_masks in support_masks]

    # Helper to convert tensors/arrays to HxW or HxWx3 numpy arrays for imshow
    def _prepare_image(x):
        # x may be a torch tensor or numpy array
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = to01(x)
            x = t2n(x)
        a = x
        # remove batch dim if present
        if a.ndim == 4 and a.shape[0] == 1:
            a = np.squeeze(a, axis=0)
        # if CHW -> HWC
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        # if single channel HxW keep as is
        return a

    # Prepare query image
    query_img_tensor = query_images[q_idx]
    query_img_np = _prepare_image(query_img_tensor)

    # Prepare predicted and label masks for Dice computation
    pred_masks = []
    for r in range(num_raters):
        pm = query_pred_reshaped[r, q_idx].argmax(0).detach().cpu().numpy()
        if pm.ndim == 3 and pm.shape[0] == 1:
            pm = np.squeeze(pm, axis=0)
        pred_masks.append((pm == 1).astype(np.uint8))

    label_masks = []
    for k in range(num_raters):
        ql = query_labels[q_idx][k]
        if isinstance(ql, torch.Tensor):
            ql = ql.detach().cpu().numpy()
        if ql.ndim == 3 and ql.shape[0] > 1:
            ql = ql.argmax(0)
        if ql.ndim == 3 and ql.shape[0] == 1:
            ql = np.squeeze(ql, axis=0)
        label_masks.append((ql == 1).astype(np.uint8))

    # Compute pairwise Dice: rows are pred rater, cols are label rater
    dice_matrix = np.zeros((num_raters, num_raters), dtype=float)
    for i in range(num_raters):
        for j in range(num_raters):
            p = pred_masks[i]
            g = label_masks[j]
            inter = (p & g).sum()
            psum = p.sum()
            gsum = g.sum()
            if psum == 0 and gsum == 0:
                dice = 1.0
            else:
                dice = 2.0 * inter / (psum + gsum + 1e-6)
            dice_matrix[i, j] = dice

    # Hungarian matching between prediction set and label set using Dice as similarity.
    # We use cost = 1 - dice for linear_sum_assignment (minimize cost). If SciPy
    # is unavailable, fall back to a greedy matching.
    matched_label_for_pred = -np.ones(num_raters, dtype=int)
    personalized = np.zeros(num_raters, dtype=bool)
    if num_raters > 0:
        if linear_sum_assignment is not None:
            cost = 1.0 - dice_matrix
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                matched_label_for_pred[r] = c
        else:
            # greedy fallback: repeatedly pick best remaining dice
            dm = dice_matrix.copy()
            assigned_preds = set()
            assigned_labels = set()
            for _ in range(num_raters):
                idx = dm.argmax()
                r = int(idx // num_raters)
                c = int(idx % num_raters)
                if r in assigned_preds or c in assigned_labels:
                    dm[r, c] = -1.0
                    continue
                matched_label_for_pred[r] = c
                assigned_preds.add(r)
                assigned_labels.add(c)
                dm[r, :] = -1.0
                dm[:, c] = -1.0
        personalized = (matched_label_for_pred == np.arange(num_raters))

    # Check if predictions across raters are identical (binary masks)
    pred_identical = {i: [] for i in range(num_raters)}
    for i in range(num_raters):
        for j in range(i + 1, num_raters):
            if np.array_equal(pred_masks[i], pred_masks[j]):
                pred_identical[i].append(j)
                pred_identical[j].append(i)

    # Prepare figure: 5 rows (support img, support label, query img, query pred, query label) x num_raters columns
    fig, axes = plt.subplots(nrows=5, ncols=num_raters, figsize=(3 * num_raters, 15))
    if num_raters == 1:
        axes = axes.reshape(5, 1)

    for r in range(num_raters):
        # Support image (use first support shot)
        if support_imgs_cpu:
            sup_img_np = _prepare_image(support_imgs_cpu[0])
            axes[0, r].imshow(sup_img_np)
            axes[0, r].set_title(f'Rater {r}: Support Image')
            axes[0, r].axis('off')

            # Support label (fg mask)
            try:
                sup_mask = support_masks_cpu[0][r]['fg_mask'].numpy()
                # sup_mask may have channel dim
                if sup_mask.ndim == 3 and sup_mask.shape[0] == 1:
                    sup_mask = np.squeeze(sup_mask, axis=0)
                axes[1, r].imshow(sup_img_np)
                axes[1, r].imshow(sup_mask, cmap='Reds', alpha=0.5)
                axes[1, r].set_title(f'Rater {r}: Support Label')
                axes[1, r].axis('off')
            except Exception:
                axes[1, r].text(0.5, 0.5, 'No support label', horizontalalignment='center')
                axes[1, r].axis('off')
        else:
            axes[0, r].text(0.5, 0.5, 'No support image', horizontalalignment='center')
            axes[0, r].axis('off')
            axes[1, r].axis('off')

        # Query image
        axes[2, r].imshow(query_img_np)
        axes[2, r].set_title(f'Rater {r}: Query Image')
        axes[2, r].axis('off')

        # Query prediction: take argmax over channels
        try:
            pred_mask = query_pred_reshaped[r, q_idx].argmax(0).detach().cpu().numpy()
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = np.squeeze(pred_mask, axis=0)
            axes[3, r].imshow(query_img_np)
            axes[3, r].imshow(pred_mask, cmap='bwr', alpha=0.5)
            axes[3, r].set_title(f'Rater {r}: Query Pred')
            axes[3, r].axis('off')
        except Exception:
            axes[3, r].text(0.5, 0.5, 'No pred', horizontalalignment='center')
            axes[3, r].axis('off')

        # Query label for this rater
        try:
            q_label = query_labels[q_idx][r]
            if isinstance(q_label, torch.Tensor):
                q_label = q_label.detach().cpu().numpy()
            if q_label.ndim == 3 and q_label.shape[0] == 1:
                q_label = np.squeeze(q_label, axis=0)
            axes[4, r].imshow(query_img_np)
            axes[4, r].imshow(q_label, cmap='Reds', alpha=0.5)
            axes[4, r].set_title(f'Rater {r}: Query Label')
            axes[4, r].axis('off')
            # Annotate Dice: self vs max other, personalization, identical preds
            try:
                # show dice against matched label (if matching available)
                matched_label = int(matched_label_for_pred[r]) if matched_label_for_pred is not None and matched_label_for_pred[r] >= 0 else r
                self_dice = dice_matrix[r, matched_label]
                if num_raters > 1:
                    others = np.delete(dice_matrix[:, matched_label], r)
                    max_other = float(np.max(others))
                else:
                    max_other = 0.0
                is_personalized = bool(personalized[r])
                matched_str = f'{matched_label}'
                identical_list = pred_identical.get(r, [])
                identical_str = ','.join(str(x) for x in identical_list) if identical_list else 'None'
                text = f'matched_label: {matched_str}\ndice: {self_dice:.3f}\nmax_other: {max_other:.3f}\npersonalized: {is_personalized}\nidentical_preds: {identical_str}'
                axes[4, r].text(0.5, -0.18, text, transform=axes[4, r].transAxes, ha='center', va='top')
            except Exception:
                pass
        except Exception:
            axes[4, r].text(0.5, 0.5, 'No label', horizontalalignment='center')
            axes[4, r].axis('off')

    plt.tight_layout()
    save_path = os.path.join(f'{_run.observers[0].dir}/snapshots', f'viz_pred_{i_iter}.png')
    fig.savefig(save_path)
    plt.close(fig)
    _log.info(f'Saved prediction visualization at iteration {i_iter}')


def compute_query_loss(query_pred_reshaped, query_labels, criterion):
    """Compute cross-entropy query loss averaged over raters.

    Args:
        query_pred_reshaped: Tensor [R, N, C, H, W]
        query_labels: list of length N, each a list of R label tensors [H, W] or [1, H, W]
        criterion: loss function (CrossEntropyLoss)

    Returns:
        scalar tensor loss
    """
    num_raters = query_pred_reshaped.shape[0]
    num_queries = query_pred_reshaped.shape[1]
    query_loss = 0.0
    for rater_idx in range(num_raters):
        pred_rater = query_pred_reshaped[rater_idx]  # [N, C, H, W]
        # collect labels for this rater into tensor [N, H, W]
        labels_rater = torch.stack([query_labels[q_idx][rater_idx] for q_idx in range(num_queries)], dim=0)
        if labels_rater.dim() == 4:
            labels_rater = labels_rater.squeeze(1)
        loss_rater = criterion(pred_rater, labels_rater)
        query_loss += loss_rater
    return query_loss / num_raters


def compute_bound_loss(query_pred_reshaped, query_labels, eps=1e-6):
    """Compute bound loss: Dice on intersection + Dice on union between preds and labels.

    Uses soft probabilities for predictions to remain differentiable:
      - intersection_pred = prod_r p_r
      - union_pred = 1 - prod_r (1 - p_r)

    For labels (binary) we compute exact intersection (all raters) and union (any rater).

    Args:
        query_pred_reshaped: Tensor [R, N, 2, H, W]
        query_labels: list length N of lists length R of label tensors

    Returns:
        scalar tensor loss (averaged over queries)
    """
    R, N, C, H, W = query_pred_reshaped.shape

    for q in range(N):
        # build predicted foreground probabilities per rater: [R, H, W]
        p_list = []
        for r in range(R):
            logits = query_pred_reshaped[r, q]  # [2, H, W]
            probs = F.softmax(logits, dim=0)
            p_fg = probs[1]
            p_list.append(p_fg)
        preds_stack = torch.stack(p_list, dim=0)  # [R, H, W]

        # intersection_pred and union_pred (soft)
        inter_pred = torch.prod(preds_stack, dim=0)
        union_pred = 1.0 - torch.prod(1.0 - preds_stack, dim=0)

        # build label stack [R, H, W]
        lbls = []
        for r in range(R):
            lbl = query_labels[q][r]
            if lbl.dim() == 3 and lbl.shape[0] > 1:
                lbl = lbl.argmax(0)
            if lbl.dim() == 3 and lbl.shape[0] == 1:
                lbl = lbl.squeeze(0)
            lbl = lbl.float()
            lbls.append(lbl)
        labels_stack = torch.stack(lbls, dim=0)

        inter_lbl = torch.prod(labels_stack, dim=0)
        union_lbl = (labels_stack.sum(dim=0) > 0).float()

        # Dice losses
        def dice_loss(pred_map, target_map):
            inter = (pred_map * target_map).sum()
            sums = pred_map.sum() + target_map.sum()
            # if both empty, treat as zero loss
            if sums.item() == 0:
                return torch.tensor(0.0, device=pred_map.device)
            dice = (2.0 * inter + eps) / (sums + eps)
            return 1.0 - dice

        loss_inter = dice_loss(inter_pred, inter_lbl)
        loss_union = dice_loss(union_pred, union_lbl)
        total_loss = (loss_inter + loss_union) / 2.0

    return total_loss / float(N)


def freeze_encoder(model, trainable=False, freeze_bn=True, keep_bn_affine=False):
    """Freeze encoder parameters and optionally control BatchNorm behavior.

    Args:
        model: model instance with `encoder` attribute (e.g., `FewShotSeg`)
        trainable: set mode for encoder parameters
        freeze_bn: if True, put encoder in `eval()` mode to disable BN running stat updates
        keep_bn_affine: if True, keep BN affine parameters (`weight`/`bias`) trainable

    Effects:
        - sets `requires_grad = trainable` for all `model.encoder` parameters
        - if `keep_bn_affine`, re-enables `requires_grad=True` for BN affine params and sets encoder to `train()`
        - otherwise, if `freeze_bn` is True, sets encoder to `eval()` to freeze BN stats
    """
    # turn on/off grads for all encoder parameters
    if not hasattr(model, 'encoder'):
        return
    for p in model.encoder.parameters():
        p.requires_grad = trainable

    # handle BatchNorm params
    if keep_bn_affine:
        for m in model.encoder.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, 'weight', None) is not None:
                    m.weight.requires_grad = True
                if getattr(m, 'bias', None) is not None:
                    m.bias.requires_grad = True
        # keep BN in train mode so affine params and running stats can update
        model.encoder.train()
    else:
        if freeze_bn:
            model.encoder.eval()


def freeze_mlp(model, trainable: bool):
    """
    Enable/disable gradient updates for any residual MLPs found on `model`.

    Strategy (robust):
      - Try attributes `residual_mlp_fg`, `residual_mlp_bg`, `residual_mlp`.
      - Also search `named_modules()` and `named_parameters()` for keys containing 'residual_mlp'.
    """
    found = False

    # search named modules for residual mlp
    for name, mod in model.named_modules():
        if 'residual_mlp' in name:
            for p in mod.parameters():
                p.requires_grad = trainable
            if trainable:
                mod.train()
            else:
                mod.eval()
            found = True

    # finally, set any parameter whose name contains 'residual_mlp'
    for name, p in model.named_parameters():
        if 'residual_mlp' in name:
            p.requires_grad = trainable
            found = True

    if found:
        print(f'### Set residual MLP trainable={trainable} ###')
    else:
        raise KeyError('No residual MLP found in model to set trainable state')

@ex.automain
def main(_run, _config, _log):

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True 
    cudnn.benchmark = True 
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config["reload_model_path"], cfg=_config['model'])

    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'CURVAS_Superpix':
        baseset_name = 'CURVAS'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')


    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name, 
        base_dir = _config['path'][data_name], 
        idx_split = _config['eval_fold'], 
        mode = 'train',
        min_fg = str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms = tr_transforms, 
        nsup = _config['task']['n_shots'], 
        scan_per_load = _config['scan_per_load'], 
        exclude_list = _config["exclude_cls_list"], 
        superpix_scale = _config["superpix_scale"], 
        test_lbs = test_labels,
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
        num_raters=_config["num_pseudo_raters"],
        mild=_config["mild_aug"]
    )
    

    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    param_group = []
    for k,v in model.named_parameters():
        if 'cls_unit' in k:
            if 'residual_mlp' in k or 'proto_attention' in k:
                param_group +=[{'params':v,'lr':_config['optim']['lr'],'momentum':_config['optim']['momentum'],'weight_decay':_config['optim']['weight_decay']}]
            else:
                param_group +=[{'params':v,'lr':_config['optim']['lr']*0.0001,'momentum':_config['optim']['momentum'],'weight_decay':_config['optim']['weight_decay']}]
        else :
            param_group +=[{'params':v,'lr':_config['optim']['lr'],'momentum':_config['optim']['momentum'],'weight_decay':_config['optim']['weight_decay']}]
    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(param_group)
    else:
        raise NotImplementedError


    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)


    i_iter = 0  # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading


    log_loss = {'loss': 0, 'align_loss': 0, 'bound_loss': 0, 'calib_loss': 0}

    _log.info('###### Training ######')
    if _config['freeze_encoder']:
        freeze_encoder(model, trainable=False)
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            if i_iter == _config['train_milestones']:
                _log.info('###### Milestone reached, unfreeze the encoder ######')
                # unfreeze encoder
                freeze_encoder(model, trainable=True, freeze_bn=False, keep_bn_affine=True)
            # Prepare input
            i_iter += 1 
            # add writers
            support_images = [[shot.cuda() for shot in way]
                              for way in sample_batched['support_images']]

            # Get masks from list: sample_batched['support_masks'] is [shot0_masks, shot1_masks, ...]
            # Each shot_masks is [rater0_dict, rater1_dict, ...] where each dict has 'fg_mask' and 'bg_mask'
            support_fg_mask = [[[rater_mask['fg_mask'].float().cuda() for rater_mask in shot_masks] 
                               for shot_masks in sample_batched['support_masks']]]
            support_bg_mask = [[[rater_mask['bg_mask'].float().cuda() for rater_mask in shot_masks] 
                               for shot_masks in sample_batched['support_masks']]]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            
            # Get labels from list: query_labels is n_queries x raters x [H x W]
            # Convert to cuda but keep the multi-rater structure
            query_labels = [[query_label.long().cuda() for query_label in query_rater_labels] 
                           for query_rater_labels in sample_batched['query_labels']]

            # Visualize multi-rater data every N iterations
            # if (i_iter - 1) % (_config['save_snapshot_every']) == 0:
            #     visualize_batch(sample_batched, query_images, query_labels, i_iter, _run, _log)

            optimizer.zero_grad()
            # FIXME: in the model definition, filter out the failure case where pseudolabel falls outside of image or too small to calculate a prototype
            try: 
                query_pred, align_loss, proto_calib_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            except Exception as e:
                import traceback
                print("Faulty batch detected, skipping")
                print("Error:", e)
                traceback.print_exc()
                continue
            # breakpoint()
            
            # query_pred shape: [Raters*N, 2, H, W]
            # query_labels shape: n_queries x raters x [H x W]
            # Reshape query_pred to [Raters, N, 2, H, W]
            num_raters = len(query_labels[0]) if query_labels else 1
            num_queries = len(query_labels)
            query_pred_reshaped = query_pred.view(num_raters, num_queries, *query_pred.shape[1:])

            # Visualize predictions and labels every N iterations
            if (i_iter - 1) % 1000 == 0:
                try:
                    visualize_pred_and_label(
                        support_images,
                        sample_batched['support_masks'],
                        query_images,
                        query_pred_reshaped,
                        query_labels,
                        i_iter,
                        _run,
                        _log,
                    )
                except Exception as e:
                    _log.warning(f'Pred visualization failed at iteration {i_iter}: {e}')
            
            # Compute query cross-entropy loss (averaged across raters)
            query_loss = compute_query_loss(query_pred_reshaped, query_labels, criterion)

            if _config['use_bound']:
                # Compute bound loss (Dice on intersection + Dice on union)
                b_loss = compute_bound_loss(query_pred_reshaped, query_labels)
            else:
                b_loss = 0.0

            # Total loss = query CE + alignment loss + bound loss
            loss = query_loss + align_loss + _config['bound_wt'] * b_loss + _config['calib_wt'] * proto_calib_loss
        
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Log loss
            q_loss_val = query_loss.detach().cpu().numpy()
            align_loss_val = align_loss.detach().cpu().numpy() if isinstance(align_loss, torch.Tensor) else align_loss
            proto_calib_loss_val = proto_calib_loss.detach().cpu().numpy() if isinstance(proto_calib_loss, torch.Tensor) else proto_calib_loss
            if _config['use_bound']:
                b_loss_val = b_loss.detach().cpu().numpy() if isinstance(b_loss, torch.Tensor) else 0

            _run.log_scalar('loss', float(q_loss_val))
            _run.log_scalar('align_loss', float(align_loss_val))
            _run.log_scalar('calib_loss', float(proto_calib_loss_val))
            if _config['use_bound']:
                _run.log_scalar('bound_loss', float(b_loss_val))
            log_loss['loss'] += float(q_loss_val)
            log_loss['align_loss'] += float(align_loss_val)
            if _config['use_bound']:
                log_loss['bound_loss'] += float(b_loss_val)
            log_loss['calib_loss'] += float(proto_calib_loss_val)

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']
                bound_loss = log_loss['bound_loss'] / _config['print_interval']
                calib_loss = log_loss['calib_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0
                log_loss['bound_loss'] = 0
                log_loss['calib_loss'] = 0

                if _config['use_bound']:
                    print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, bound_loss: {bound_loss}, calib_loss: {calib_loss}')
                else:
                    print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, calib_loss: {calib_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0 or (i_iter + 1) == _config['n_steps']:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up

