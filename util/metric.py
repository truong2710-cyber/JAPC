"""
Metrics for computing evalutation results
Modified from vanilla PANet code by Wang et al.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_scans:
            number of test scans
    """
    def __init__(self, max_label=20, n_scans=None, debug_viz_dir='./debug_viz'):
        self.labels = list(range(max_label + 1))  # all class labels [0 1 2 3 4 ]
        self.n_scans = 1 if n_scans is None else n_scans # 4

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_scans)] # [[],[],[],[]]
        self.fp_lst = [[] for _ in range(self.n_scans)] # [[],[],[],[]]
        self.fn_lst = [[] for _ in range(self.n_scans)] # [[],[],[],[]]

        # optional directory to save pred/target visualizations for debugging
        self.debug_viz_dir = debug_viz_dir
        if self.debug_viz_dir:
            os.makedirs(self.debug_viz_dir, exist_ok=True)

    def reset(self):
        """
        Reset accumulated evaluation. 
        """
        # assert self.n_scans == 1, 'Should not reset accumulated result when we are not doing one-time batch-wise validation'
        del self.tp_lst, self.fp_lst, self.fn_lst
        # n_scans x n_slices x [R (variable) x num_classes]
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]


    def _stack_scan_entries(self, entries):
        """
        Flatten a list of per-sample entries where each entry can be either
        a 1D array (L,) or a 2D array (R, L). Returns a 2D array of shape
        (sum_R_over_samples, L) where each row corresponds to one rater-sample.
        """
        if len(entries) == 0:
            return np.empty((0, len(self.labels)))
        stacked = []
        for e in entries:
            arr = np.asarray(e)
            if arr.ndim == 1:
                stacked.append(arr[np.newaxis, :])
            elif arr.ndim == 2:
                stacked.append(arr.reshape(-1, arr.shape[-1]))
            else:
                raise ValueError("Unexpected entry ndim in metric stack: %d" % arr.ndim)
        return np.vstack(stacked)

    def visualize_pair(self, pred2d, target2d, out_path):
        """Save a side-by-side visualization of predicted and target 2D label maps."""
        pred2d = np.asarray(pred2d)
        target2d = np.asarray(target2d)
        cmap = plt.get_cmap('tab20')
        max_label = int(max(np.nanmax(pred2d) if pred2d.size else 0, np.nanmax(target2d) if target2d.size else 0))
        nlabels = max(1, max_label + 1)

        def label_to_rgb(label_arr):
            h, w = label_arr.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for lab in range(nlabels):
                mask = (label_arr == lab)
                color = cmap(lab % 20)[:3]
                rgb[mask] = color
            rgb[label_arr == 255] = 0.0
            return rgb

        pred_rgb = label_to_rgb(pred2d)
        targ_rgb = label_to_rgb(target2d)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(pred_rgb)
        axs[0].set_title('pred')
        axs[0].axis('off')
        axs[1].imshow(targ_rgb)
        axs[1].set_title('target')
        axs[1].axis('off')
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def record(self, pred, target, labels=None, n_scan=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is R x H x W 256 256
            target:
                target mask array, expected shape is R x H x W 256 256
            labels:
                only count specific label, used when knowing all possible labels in advance 2
        """
        # allow pred/target to be either HxW or R x H x W (multi-rater)
        if self.n_scans == 1:
            n_scan = 0

        if labels is None:
            labs = self.labels
        else:
            labs = [0,] + labels

        def _counts_for_pair(pred2d, target2d):
            tp_arr = np.full(len(self.labels), np.nan)
            fp_arr = np.full(len(self.labels), np.nan)
            fn_arr = np.full(len(self.labels), np.nan)
            for j, label in enumerate(labs):
                idx = np.where(np.logical_and(pred2d == j, target2d != 255))
                pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
                idx2 = np.where(target2d == j)
                target_idx_j = set(zip(idx2[0].tolist(), idx2[1].tolist()))

                tp_arr[label] = len(pred_idx_j & target_idx_j)
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)

            return tp_arr, fp_arr, fn_arr

        pred = np.asarray(pred)
        target = np.asarray(target)

        if pred.ndim == 3:
            R = pred.shape[0]
            tp_rows = []
            fp_rows = []
            fn_rows = []
            for r in range(R):
                pred2d = pred[r]
                if target.ndim == 3 and target.shape[0] == R:
                    target2d = target[r]
                else:
                    target2d = target
                tp_r, fp_r, fn_r = _counts_for_pair(pred2d, target2d)
                # # optional visualization for debugging
                # if getattr(self, 'debug_viz_dir', None):
                #     entry_idx = len(self.tp_lst[n_scan])
                #     fname = os.path.join(self.debug_viz_dir, f'scan{n_scan}_entry{entry_idx}_r{r}.png')
                #     try:
                #         self.visualize_pair(pred2d, target2d, fname)
                #     except Exception:
                #         pass
                tp_rows.append(tp_r)
                fp_rows.append(fp_r)
                fn_rows.append(fn_r)
            self.tp_lst[n_scan].append(np.vstack(tp_rows))
            self.fp_lst[n_scan].append(np.vstack(fp_rows))
            self.fn_lst[n_scan].append(np.vstack(fn_rows))
        else:
            tp_arr, fp_arr, fn_arr = _counts_for_pair(pred, target)
            self.tp_lst[n_scan].append(tp_arr)
            self.fp_lst[n_scan].append(fp_arr)
            self.fn_lst[n_scan].append(fn_arr)

    def get_mIoU(self, labels=None, n_scan=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            # For each scan: compute per-sample-per-rater IoU then average
            mIoU_class_list = []
            for _scan in range(self.n_scans):
                tp_all = self._stack_scan_entries(self.tp_lst[_scan]) # [sum_R of scan x num_classes]
                fp_all = self._stack_scan_entries(self.fp_lst[_scan]) # [sum_R of scan x num_classes]
                fn_all = self._stack_scan_entries(self.fn_lst[_scan]) # [sum_R of scan x num_classes]

                denom = tp_all + fp_all + fn_all
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou_per_entry = tp_all / denom
                # mean across entries (each entry is one rater-sample)
                mIoU_class_scan = np.nanmean(iou_per_entry, axis=0)
                mIoU_class_list.append(mIoU_class_scan.take(labels))

            mIoU_class = np.vstack(mIoU_class_list)
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_all = self._stack_scan_entries(self.tp_lst[n_scan])
            fp_all = self._stack_scan_entries(self.fp_lst[n_scan])
            fn_all = self._stack_scan_entries(self.fn_lst[n_scan])

            denom = tp_all + fp_all + fn_all
            with np.errstate(divide='ignore', invalid='ignore'):
                iou_per_entry = tp_all / denom
            mIoU_class = np.nanmean(iou_per_entry, axis=0).take(labels)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mDice(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute mean Dice score (in 3D scan level)

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples 1 4
        if n_scan is None:
            mDice_class_list = []
            for _scan in range(self.n_scans):
                tp_all = self._stack_scan_entries(self.tp_lst[_scan])
                fp_all = self._stack_scan_entries(self.fp_lst[_scan])
                fn_all = self._stack_scan_entries(self.fn_lst[_scan])

                denom = 2 * tp_all + fp_all + fn_all
                with np.errstate(divide='ignore', invalid='ignore'):
                    dice_per_entry = 2 * tp_all / denom
                mDice_class_scan = np.nanmean(dice_per_entry, axis=0)
                mDice_class_list.append(mDice_class_scan.take(labels))

            mDice_class = np.vstack(mDice_class_list)
            mDice = mDice_class.mean(axis=1)
            if not give_raw:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0))
            else:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0), mDice_class)

        else:
            tp_all = self._stack_scan_entries(self.tp_lst[n_scan])
            fp_all = self._stack_scan_entries(self.fp_lst[n_scan])
            fn_all = self._stack_scan_entries(self.fn_lst[n_scan])

            denom = 2 * tp_all + fp_all + fn_all
            with np.errstate(divide='ignore', invalid='ignore'):
                dice_per_entry = 2 * tp_all / denom
            mDice_class = np.nanmean(dice_per_entry, axis=0).take(labels)
            mDice = mDice_class.mean()

            return mDice_class, mDice

    def get_mPrecRecall(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute precision and recall

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            mPrec_list = []
            mRec_list = []
            for _scan in range(self.n_scans):
                tp_all = self._stack_scan_entries(self.tp_lst[_scan])
                fp_all = self._stack_scan_entries(self.fp_lst[_scan])
                fn_all = self._stack_scan_entries(self.fn_lst[_scan])

                with np.errstate(divide='ignore', invalid='ignore'):
                    prec_per_entry = tp_all / (tp_all + fp_all)
                    rec_per_entry = tp_all / (tp_all + fn_all)

                mPrec_class_scan = np.nanmean(prec_per_entry, axis=0)
                mRec_class_scan = np.nanmean(rec_per_entry, axis=0)
                mPrec_list.append(mPrec_class_scan.take(labels))
                mRec_list.append(mRec_class_scan.take(labels))

            mPrec_class = np.vstack(mPrec_list)
            mRec_class = np.vstack(mRec_list)

            mPrec = mPrec_class.mean(axis=1)
            mRec  = mRec_class.mean(axis=1)
            if not give_raw:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0))
            else:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0), mPrec_class, mRec_class)


        else:
            tp_all = self._stack_scan_entries(self.tp_lst[n_scan])
            fp_all = self._stack_scan_entries(self.fp_lst[n_scan])
            fn_all = self._stack_scan_entries(self.fn_lst[n_scan])

            with np.errstate(divide='ignore', invalid='ignore'):
                prec_per_entry = tp_all / (tp_all + fp_all)
                rec_per_entry = tp_all / (tp_all + fn_all)

            mPrec_class = np.nanmean(prec_per_entry, axis=0).take(labels)
            mPrec = mPrec_class.mean()

            mRec_class = np.nanmean(rec_per_entry, axis=0).take(labels)
            mRec = mRec_class.mean()

            return (mPrec_class, mPrec, mRec_class, mRec)

    def get_mIoU_binary(self, n_scan=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            # For each scan: compute per-entry (rater-sample) binary IoU then average
            mIoU_class_list = []
            for _scan in range(self.n_scans):
                tp_all = self._stack_scan_entries(self.tp_lst[_scan])  # (entries, L)
                fp_all = self._stack_scan_entries(self.fp_lst[_scan])
                fn_all = self._stack_scan_entries(self.fn_lst[_scan])

                if tp_all.size == 0:
                    mIoU_class_list.append(np.array([np.nan, np.nan]))
                    continue

                # background per-entry
                bg_tp = tp_all[:, 0]
                bg_fp = fp_all[:, 0]
                bg_fn = fn_all[:, 0]
                # foreground aggregated across classes per-entry
                fg_tp = np.nansum(tp_all[:, 1:], axis=1)
                fg_fp = np.nansum(fp_all[:, 1:], axis=1)
                fg_fn = np.nansum(fn_all[:, 1:], axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    iou_bg = bg_tp / (bg_tp + bg_fp + bg_fn)
                    iou_fg = fg_tp / (fg_tp + fg_fp + fg_fn)

                mIoU_bg = self._safe_nanmean_1d(iou_bg)
                mIoU_fg = self._safe_nanmean_1d(iou_fg)
                mIoU_class_list.append(np.array([mIoU_bg, mIoU_fg]))

            mIoU_class = np.vstack(mIoU_class_list)
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_all = self._stack_scan_entries(self.tp_lst[n_scan])
            fp_all = self._stack_scan_entries(self.fp_lst[n_scan])
            fn_all = self._stack_scan_entries(self.fn_lst[n_scan])

            if tp_all.size == 0:
                return np.array([np.nan, np.nan]), np.nan

            bg_tp = tp_all[:, 0]
            bg_fp = fp_all[:, 0]
            bg_fn = fn_all[:, 0]
            fg_tp = np.nansum(tp_all[:, 1:], axis=1)
            fg_fp = np.nansum(fp_all[:, 1:], axis=1)
            fg_fn = np.nansum(fn_all[:, 1:], axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                iou_bg = bg_tp / (bg_tp + bg_fp + bg_fn)
                iou_fg = fg_tp / (fg_tp + fg_fp + fg_fn)

            mIoU_bg = self._safe_nanmean_1d(iou_bg)
            mIoU_fg = self._safe_nanmean_1d(iou_fg)
            mIoU_class = np.array([mIoU_bg, mIoU_fg])
            mIoU = np.nanmean(mIoU_class)

            return mIoU_class, mIoU
