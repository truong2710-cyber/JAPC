"""
Manually labeled dataset
TODO: 
1. Merge with superpixel dataset
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
# from common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList

class ManualAnnoDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, transforms, scan_per_load, min_fg = '', fix_length = None, tile_z_dim = 3, nsup = 1, exclude_list = [], extern_normalize_func = None,**kwargs):
        """
        Manually labeled dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            transforms:         data transform (augmentation) function
            min_fg:             minimum number of positive pixels in a 2D slice, mainly for stablize training when trained on manually labeled dataset
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            nsup:               number of support scans
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            extern_normalize_function:  normalization function used for data pre-processing  
        """

        # allow overriding dataset directory per split using train_dir/test_dir kwargs
        self._use_explicit_dirs = True
        if isinstance(base_dir, dict):
            if mode == 'train' and 'train_dir' in base_dir:
                base_dir = base_dir['train_dir']
            elif mode == 'val' and 'test_dir' in base_dir:
                base_dir = base_dir['test_dir']
            else:
                base_dir = base_dir.get('data_dir', '')
                self._use_explicit_dirs = False

        super(ManualAnnoDataset, self).__init__(base_dir)
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.fix_length = fix_length
        self.all_label_names = self.label_name
        self.nclass = len(self.label_name)
        self.tile_z_dim = tile_z_dim
        self.base_dir = base_dir
        self.nsup = nsup
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x))) # make it circular for the ease of spliting folds

        self.exclude_lbs = exclude_list
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')

        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)

        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold

        if extern_normalize_func is not None: # helps to keep consistent between training and testing dataset.
            self.norm_func = extern_normalize_func
            print(f'###### Dataset: using external normalization statistics ######')
        else:
            self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])
            print(f'###### Dataset: using normalization statistics calculated from loaded data ######')

        if self.is_train:
            if scan_per_load > 0: # buffer needed
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
            self.potential_support_sid = []
        else:
            raise Exception
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()
        self.update_subclass_lookup()

    def get_scanids(self, mode, idx_split):
        # If explicit train/test directories were provided, do not perform cross-fold splitting
        if getattr(self, '_use_explicit_dirs', False):
            return list(self.img_pids)

        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        self.potential_support_sid = val_ids[-self.nsup:] # this is actual file scan id, not index
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a portion of the entire dataset, if the dataset is too large
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
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            # find all rater label files for this scan: label_{curr_id}_{rater_id}.nii.gz
            label_pattern = os.path.join(self.base_dir, f'label_{curr_id}_*.nii.gz')
            label_files = sorted(glob.glob(label_pattern))
            _lb_fids_dict = {}
            for lb_fid in label_files:
                m = re.findall(r'label_\d+_(\d+)\.nii\.gz', lb_fid)
                if len(m) == 0:
                    continue
                rater_id = int(m[0])
                _lb_fids_dict[rater_id] = lb_fid

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fids"] = _lb_fids_dict
            out_list[str(curr_id)] = curr_dict
        return out_list

    def read_dataset(self):
        """
        Build index pointers to individual slices
        Also keep a look-up table from scan_id, slice to index
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

            # Load all available labels for this scan (may be multiple raters)
            lbs_dict = {}
            for rater_id, lb_fid in itm.get("lbs_fids", {}).items():
                if os.path.exists(lb_fid):
                    lb = read_nii_bysitk(lb_fid)
                    lb = lb.transpose(1,2,0)
                    lb = np.int32(lb)
                    lb = lb[:256, :256, :]
                    lbs_dict[rater_id] = lb

            # if no labels found, skip this scan
            if len(lbs_dict) == 0:
                print(f"Warning: No valid labels found for scan {scan_id}, skipping")
                continue

            img = img[:256, :256, :]
            # use first available label for shape validation
            lb = lbs_dict[list(lbs_dict.keys())[0]]

            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            # write the beginning frame
            out_list.append( {"img": img[..., 0: 1],
                           "lbs":{rater_id: lb_single[..., 0: 0 + 1] for rater_id, lb_single in lbs_dict.items()},
                           "is_start": True,
                           "is_end": False,
                           "nframe": img.shape[-1],
                           "scan_id": scan_id,
                           "z_id":0})

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lbs":{rater_id: lb_single[..., ii: ii + 1] for rater_id, lb_single in lbs_dict.items()},
                           "is_start": False,
                           "is_end": False,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last frame, note the is_end flag
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lbs":{rater_id: lb_single[..., ii: ii+ 1] for rater_id, lb_single in lbs_dict.items()},
                           "is_start": False,
                           "is_end": True,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

        return out_list

    def read_classfiles(self):
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train:
            if len(self.exclude_lbs) > 0:
                for _ex_cls in self.exclude_lbs:
                    if curr_dict["z_id"] in self.tp1_cls_map[self.label_name[_ex_cls]][curr_dict["scan_id"]]: # this slice need to be excluded since it contains label which is supposed to be unseen
                        return self.__getitem__(index + torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))
            # When training, apply transforms to image+rater labels 
            labels_raw = curr_dict.get('lbs', {})
            available_rater_ids = sorted(list(labels_raw.keys()))

            img_out = None
            lbs_out_dict = {}
            
            # Concatenate image with all R compact labels and transform once
            label_list = [labels_raw[r] for r in available_rater_ids]
            # Convert any torch tensors to numpy
            label_list_np = [l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in label_list]
            labels_concat = np.concatenate(label_list_np, axis=-1)  # H x W x R
            comp = np.concatenate([curr_dict["img"], labels_concat], axis=-1)  # H x W x (1 + R)

            img_out, lbs_out = self.transforms(comp, c_img=1, c_label=1, nclass=self.nclass, use_onehot=False, r_label=len(available_rater_ids))

            img = img_out
            # lbs_out expected shape: H x W x R (compact labels per rater)
            for i, rater_id in enumerate(available_rater_ids):
                out_np = lbs_out[..., i]
                lbs_out_dict[rater_id] = torch.from_numpy(out_np)

            labels_raw = lbs_out_dict

        else:
            img = curr_dict['img']
            labels_raw = curr_dict.get('lbs', {})
            available_rater_ids = sorted(list(labels_raw.keys()))

        img = np.float32(img)

        # prepare labels dict as tensors (squeezed)
        labels_tensors = {}
        for rater_id in available_rater_ids:
            lab_np = labels_raw[rater_id]
            lab_np = np.float32(lab_np).squeeze(-1)
            labels_tensors[rater_id] = torch.from_numpy(lab_np)

        img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )

        if self.tile_z_dim:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

        is_start = curr_dict["is_start"]
        is_end = curr_dict["is_end"]
        nframe = np.int32(curr_dict["nframe"])
        scan_id = curr_dict["scan_id"]
        z_id    = curr_dict["z_id"]

        sample = {"image": img, # [C,H,W]
            "labels": labels_tensors, # {rater_id: label_tensor [H,W]}
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
                aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                for key_suffix in aux_attrib_val:
                    # one function may create multiple attributes, so we need suffix to distinguish them
                    sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length != None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def update_subclass_lookup(self):
        """
        Updating the class-slice indexing list
        Args:
            [internal] overall_slice_by_cls:
                {
                    class1: {pid1: [slice1, slice2, ....],
                                pid2: [slice1, slice2]},
                                ...}
                    class2:
                    ...
                }
        out[internal]:
                {
                    class1: [ idx1, idx2, ...  ],
                    class2: [ idx1, idx2, ...  ],
                    ...
                }

        """
        # delete previous ones if any
        assert self.overall_slice_by_cls is not None

        if not hasattr(self, 'idx_by_class'):
             self.idx_by_class = {}
        # filter the new one given the actual list
        for cls in self.label_name:
            if cls not in self.idx_by_class.keys():
                self.idx_by_class[cls] = []
            else:
                del self.idx_by_class[cls][:]
        for cls, dict_by_pid in self.overall_slice_by_cls.items():
            for pid, slice_list in dict_by_pid.items():
                if pid not in self.pid_curr_load:
                    continue
                self.idx_by_class[cls] += [ self.scan_z_idx[pid][_sli] for _sli in slice_list ]
        print("###### index-by-class table has been reloaded ######")

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask. Used when getting the support
        """
        # Dense Mask
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}

    def subsets(self, sub_args_lst=None):
        """
        Override base-class subset method
        Create subsets by scan_ids

        output: list [[<fid in each class>] <class1>, <class2>     ]
        """

        if sub_args_lst is not None:
            subsets = []
            ii = 0
            for cls_name, index_list in self.idx_by_class.items():
                subsets.append( Subset(dataset = self, indices = index_list, sub_attrib_args = sub_args_lst[ii])  )
                ii += 1
        else:
            subsets = [Subset(dataset=self, indices=index_list) for _, index_list in self.idx_by_class.items()]
        return subsets

    def get_support(self, curr_class: int, class_idx: list, scan_idx: list, npart: int):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                _zid = _zlist[int(pcts[_part] * len(_zlist))]
                _glb_idx = self.scan_z_idx[_scan_id][_zid]

                # almost copy-paste __getitem__ but no augmentation
                curr_dict = self.actual_dataset[_glb_idx]
                img = curr_dict['img']
                # load all available rater labels for this slice
                lbs_dict = curr_dict.get('lbs', {})

                img = np.float32(img)
                # prepare label tensors dict: rater_id -> tensor [H,W]
                labels_tensors = {}
                for rater_id, lb_np in lbs_dict.items():
                    lab = np.float32(lb_np).squeeze(-1)
                    labels_tensors[rater_id] = torch.from_numpy(lab)

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )

                if self.tile_z_dim:
                    img = img.repeat( [ self.tile_z_dim, 1, 1] )
                    assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                is_start    = curr_dict["is_start"]
                is_end      = curr_dict["is_end"]
                nframe      = np.int32(curr_dict["nframe"])
                scan_id     = curr_dict["scan_id"]
                z_id        = curr_dict["z_id"]

                sample = {"image": img,
                    "labels": labels_tensors,
                    "is_start": is_start,
                    "inst": None,
                    "scribble": None,
                    "is_end": is_end,
                    "nframe": nframe,
                    "scan_id": scan_id,
                    "z_id": z_id
                    }

                concat_buffer.append(sample)
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "labels": [itm["labels"] for itm in concat_buffer],  # list of dicts per shot

                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            # build masks per support shot; each shot may have multiple raters
            masks_for_part = []
            for labels_dict in itm["labels"]:
                # labels_dict: {rater_id: tensor}
                masks_for_shot = []
                for rater_id in sorted(labels_dict.keys()):
                    masks_for_shot.append(self.getMaskMedImg(labels_dict[rater_id], curr_class, class_idx))
                masks_for_part.append(masks_for_shot)
            support_mask.append(masks_for_part)

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

