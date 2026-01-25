"""
Dataset for training with pseudolabels
TODO:
1. Merge with manual annotated dataset
2. superpixel_scale -> superpix_config, feed like a dict
"""
import glob
import numpy as np
import dataloaders.augutils as myaug
import torch
import random
import os
import copy
import platform
import json
import re
from dataloaders.common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList

class SuperpixelDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, transforms, scan_per_load, num_rep = 2, min_fg = '', nsup = 1, fix_length = None, tile_z_dim = 3, exclude_list = [], test_lbs = [], superpix_scale = 'SMALL', **kwargs):
        """
        Pseudolabel dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            nsup:               number of scans used as support. currently idle for superpixel dataset
            transforms:         data transform (augmentation) function
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            num_rep:            Number of augmentation applied for a same pseudolabel
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            test_lbs:           labels to be treated as unseen classes during training
            superpix_scale:     config of superpixels
        """

        super(SuperpixelDataset, self).__init__(base_dir) 

        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.pseu_label_name = DATASET_INFO[which_dataset]['PSEU_LABEL_NAME']
        self.real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

        self.transforms = transforms
        self.is_train = True if mode == 'train' else False 
        assert mode == 'train'
        self.fix_length = fix_length
        self.nclass = len(self.pseu_label_name)
        self.num_rep = num_rep
        self.tile_z_dim = tile_z_dim
        self.test_lbs = test_lbs

        # find scans in the data folder
        self.nsup = nsup
        self.base_dir = base_dir
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x)))

        # experiment configs
        self.exclude_lbs = exclude_list
        self.superpix_scale = superpix_scale
        if self.superpix_scale is not None and self.superpix_scale != '':
            self.use_gt = False
        else:
            self.use_gt = True

        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')
        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)
        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold
        self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])

        if self.is_train:
            if scan_per_load > 0: # if the dataset is too large, only reload a subset in each sub-epoch
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        else:
            raise Exception
            
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()

        print("###### Initial scans loaded: ######")
        print(self.pid_curr_load)

    def get_scanids(self, mode, idx_split):
        """
        Load scans by train-test split
        leaving one additional scan as the support scan. if the last fold, taking scan 0 as the additional one
        Args:
            idx_split: index for spliting cross-validation folds
        """
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a only portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing notiong")
            return -1

        del self.actual_dataset
        del self.info_by_scan

        self.pid_curr_load = np.random.choice( self.scan_ids, size = self.scan_per_load, replace = False )
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        self.all_rater_ids = set()  # Track all rater IDs across the dataset
        
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            if self.use_gt == False:
                _lb_fids_dict = {0: os.path.join(self.base_dir, f'superpix-{self.superpix_scale}_{curr_id}.nii.gz')}
            else:
                # Find all available label files for this scan (label_{curr_id}_{rater_id}.nii.gz)
                label_pattern = os.path.join(self.base_dir, f'label_{curr_id}_*.nii.gz')
                label_files = sorted(glob.glob(label_pattern))
                _lb_fids_dict = {}
                for lb_fid in label_files:
                    # Extract rater ID from filename: label_{curr_id}_{rater_id}.nii.gz
                    rater_id = int(re.findall(r'label_\d+_(\d+)\.nii\.gz', lb_fid)[0])
                    _lb_fids_dict[rater_id] = lb_fid
                    self.all_rater_ids.add(rater_id)
                
                if len(_lb_fids_dict) == 0:
                    print(f"Warning: No label files found for scan {curr_id}")

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fids"] = _lb_fids_dict  # Dict mapping rater_id -> file_path
            out_list[str(curr_id)] = curr_dict
        
        self.all_rater_ids = sorted(list(self.all_rater_ids))
        return out_list

    def read_dataset(self):
        """
        Read images into memory and store them in 2D
        Build tables for the position of an individual 2D slice in the entire dataset
        """
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
            img = img.transpose(1,2,0)
            self.info_by_scan[scan_id] = _info

            img = np.float32(img)
            img = self.norm_func(img)

            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]

            # Load all available labels for this scan, organized by rater_id
            lbs_dict = {}  # {rater_id: label_array}
            for rater_id, lb_fid in itm["lbs_fids"].items():
                if os.path.exists(lb_fid):
                    lb = read_nii_bysitk(lb_fid)
                    lb = lb.transpose(1,2,0)
                    lb = np.int32(lb)
                    lb = lb[:256, :256, :]
                    lbs_dict[rater_id] = lb
            
            # Handle case where no labels are available
            if len(lbs_dict) == 0:
                print(f"Warning: No valid labels found for scan {scan_id}, skipping")
                continue
            
            img = img[:256, :256, :]
            # Use first available label for shape validation
            lb = lbs_dict[list(lbs_dict.keys())[0]]

            # format of slices: [axial_H x axial_W x Z]
            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            # re-organize 3D images into 2D slices and record essential information for each slice
            out_list.append( {"img": img[..., 0: 1],
                           "lbs":{rater_id: lb_single[..., 0: 0 + 1] for rater_id, lb_single in lbs_dict.items()},
                           "sup_max_cls": lb[..., 0: 0 + 1].max(),
                           "is_start": True,
                           "is_end": False,
                           "nframe": img.shape[-1],
                           "scan_id": scan_id,
                           "z_id":0,
                           "available_rater_ids": list(lbs_dict.keys())})

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lbs":{rater_id: lb_single[..., ii: ii + 1] for rater_id, lb_single in lbs_dict.items()},
                           "is_start": False,
                           "is_end": False,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii,
                           "available_rater_ids": list(lbs_dict.keys())
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last slice of a 3D volume
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lbs":{rater_id: lb_single[..., ii: ii+ 1] for rater_id, lb_single in lbs_dict.items()},
                           "is_start": False,
                           "is_end": True,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii,
                           "available_rater_ids": list(lbs_dict.keys())
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

        return out_list

    def read_classfiles(self):
        """
        Load the scan-slice-class indexing file
        """
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def supcls_pick_binarize(self, super_map, sup_max_cls, exclude = [], bi_val = None):
        """
        pick up a certain super-pixel class or multiple classes, and binarize it into segmentation target
        Args:
            super_map:      super-pixel map
            bi_val:         if given, pick up a certain superpixel. Otherwise, draw a random one
            exclude:        list of test classes to be excluded
            sup_max_cls:    max index of superpixel for avoiding overshooting when selecting superpixel

        """
        if self.use_gt == False:
            exclude = [] # do not exclude any class when using superpixel pseudolabels
        if bi_val == None:
            # select bi_val in [1, sup_max_cls], excluding those in exclude list
            candidate_cls = [ii for ii in range(1, int(sup_max_cls) + 1) if ii not in exclude]
            if len(candidate_cls) == 0:
                return np.float32(super_map == 0) # all background
            bi_val = random.choice(candidate_cls)

        return np.float32(super_map == bi_val)


    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        sup_max_cls = curr_dict['sup_max_cls']
        if sup_max_cls < 1:
            return self.__getitem__(index + 1)

        image_t = curr_dict["img"]
        labels_raw = curr_dict["lbs"]  # Dict mapping rater_id -> label
        available_rater_ids = curr_dict["available_rater_ids"]

        for _ex_cls in self.exclude_lbs:
            if curr_dict["z_id"] in self.tp1_cls_map[self.real_label_name[_ex_cls]][curr_dict["scan_id"]]: # if using setting 1, this slice need to be excluded since it contains label which is supposed to be unseen
                return self.__getitem__(torch.randint(low = 0, high = self.__len__(), size = (1,)))

        # Binarize all labels
        labels_t = {rater_id: self.supcls_pick_binarize(label_raw, sup_max_cls, exclude=self.test_lbs) 
                    for rater_id, label_raw in labels_raw.items()}

        pair_buffer = []

        for ii in range(self.num_rep):
            # Apply transforms to image + each label separately with same random seed
            sorted_rater_ids = sorted(available_rater_ids)
            
            # Use a deterministic seed based on index and iteration (must be < 2**32 - 1 for numpy)
            aug_seed = (index * 1000 + ii) % (2**31 - 1)
            torch.manual_seed(aug_seed)
            np.random.seed(aug_seed)
            random.seed(aug_seed)
            
            # Apply transforms to first label to get the transformed image
            first_rater_id = sorted_rater_ids[0]
            comp_first = np.concatenate([curr_dict["img"], labels_t[first_rater_id]], axis=-1)
            img, lbs_first = self.transforms(comp_first, c_img=1, c_label=1, nclass=self.nclass, is_train=True, use_onehot=False)
            
            # Apply same transforms to other labels with same seed
            lbs_dict = {first_rater_id: torch.from_numpy(lbs_first.squeeze(-1))}
            
            for idx, rater_id in enumerate(sorted_rater_ids[1:], 1):
                comp_other = np.concatenate([curr_dict["img"], labels_t[rater_id]], axis=-1)
                _, lbs_other = self.transforms(comp_other, c_img=1, c_label=1, nclass=self.nclass, is_train=True, use_onehot=False)
                lbs_dict[rater_id] = torch.from_numpy(lbs_other.squeeze(-1))
            
            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

            if self.tile_z_dim:
                img = img.repeat([self.tile_z_dim, 1, 1])
                assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

            is_start = curr_dict["is_start"]
            is_end = curr_dict["is_end"]
            nframe = np.int32(curr_dict["nframe"])
            scan_id = curr_dict["scan_id"]
            z_id = curr_dict["z_id"]

            sample = {"image": img,
                    "labels": lbs_dict,  # Dict mapping rater_id -> label tensor
                    "available_rater_ids": available_rater_ids,
                    "is_start": is_start,
                    "is_end": is_end,
                    "nframe": nframe,
                    "scan_id": scan_id,
                    "z_id": z_id
                    }

            # Add auxiliary attributes
            if self.aux_attrib is not None:
                for key_prefix in self.aux_attrib:
                    # Process the data sample, create new attributes and save them in a dictionary
                    # Use the first available label for auxiliary attribute computation
                    sample_for_aux = sample.copy()
                    first_rater_id = available_rater_ids[0]
                    sample_for_aux["label"] = lbs_dict[first_rater_id]  # Provide first label for compatibility
                    aux_attrib_val = self.aux_attrib[key_prefix](sample_for_aux, **self.aux_attrib_args[key_prefix])
                    for key_suffix in aux_attrib_val:
                        # one function may create multiple attributes, so we need suffix to distinguish them
                        sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]
            pair_buffer.append(sample)

        support_images = []
        support_masks = {rater_id: [] for rater_id in self.all_rater_ids}  # Dict mapping rater_id -> list of masks
        support_class = []

        query_images = []
        query_labels_dict = {rater_id: [] for rater_id in self.all_rater_ids}  # Dict mapping rater_id -> list of labels
        query_class = []

        for idx, itm in enumerate(pair_buffer):
            if idx % 2 == 0:
                support_images.append(itm["image"])
                support_class.append(1)  # pseudolabel class
                # Create masks for each rater
                for rater_id in self.all_rater_ids:
                    if rater_id in itm["labels"]:
                        mask = self.getMaskMedImg(itm["labels"][rater_id], 1, [1])
                        support_masks[rater_id].append(mask)
                    else:
                        support_masks[rater_id].append(None)
            else:
                query_images.append(itm["image"])
                query_class.append(1)
                # Store all labels for each rater
                for rater_id in self.all_rater_ids:
                    if rater_id in itm["labels"]:
                        query_labels_dict[rater_id].append(itm["labels"][rater_id])
                    else:
                        query_labels_dict[rater_id].append(None)

        return {'class_ids': [support_class], # shape: [[1 x num_support]] where num_support=1 (1 support per num_rep=2 iterations)
            'support_images': [support_images], # shape: 1 way x 1 shot x [3 x H x W]
            'support_masks': support_masks,  # Dict mapping rater_id -> list of masks. Each list is [{'fg_mask': H x W, 'bg_mask': H x W}]
            'query_images': query_images, # n_queries x [3 x H x W]
            'query_labels': query_labels_dict,  # Dict mapping rater_id -> list of labels. Each list is n_queries x [H x W]
            'available_rater_ids': self.all_rater_ids,  # All rater IDs in the dataset. For example, [1, 2, 3]
            'sample_rater_ids': available_rater_ids,  # Rater IDs available for this specific sample. For example, [1, 3]
        }


    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length != None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
            class_ids:      all class id in this episode
        """
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}
