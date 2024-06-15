import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json


class PRCC(object):
    """
    PRCC with human mask

    Reference:
    Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view
    
    Dataset statistics:
    # identities: 150 (train) + 71 (test)
    # images: 17896 (train) + 5002 (val) + 10800 (test)
    """
    dataset_dir = 'prcc'

    def __init__(self, root='data', train_with_mask=True, test_with_mask=False, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_with_mask = train_with_mask
        self.test_with_mask = test_with_mask
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.val_dir = osp.join(self.dataset_dir, 'rgb/val')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')

        self.train_mask_dir = osp.join(self.dataset_dir, 'human_parsing/train')
        self.val_mask_dir = osp.join(self.dataset_dir, 'human_parsing/val')
        self.test_mask_dir = osp.join(self.dataset_dir, 'human_parsing/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2cloth = \
            self._process_dir_train(self.train_dir)
        val, num_val_pids, num_val_imgs, num_val_clothes, _ = \
            self._process_dir_train(self.val_dir)

        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        print("=> PRCC loaded")
        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset      | # ids | # images | # clothes")
        print("  --------------------------------------------")
        print("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        print("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
        print("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        print("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        print("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        print("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        print("  --------------------------------------------")
        print("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        print("  --------------------------------------------")

        self.train = train
        self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2cloth = pid2cloth
        self.gallery_idx = gallery_idx

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        if self.train_with_mask:
            if not osp.exists(self.train_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.train_mask_dir))
        if self.test_with_mask:
            if not osp.exists(self.val_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.val_mask_dir))
            if not osp.exists(self.test_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.test_mask_dir))

    def _process_dir_train(self, dir_path):
        pdirs = glob.glob(osp.join(dir_path, '*'))

        pid_container = set()
        cloth_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                if cam in ['A', 'B']:
                    cloth_container.add(osp.basename(pdir))
                else:
                    cloth_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        cloth_container = sorted(cloth_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cloth2label = {clothid:label for label, clothid in enumerate(cloth_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(cloth_container)

        dataset = []
        pid2cloth = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothid = cloth2label[osp.basename(pdir)]
                else:
                    clothid = cloth2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                if self.train_with_mask:
                    mask = img_dir.replace('/rgb/', '/human_parsing/')[0:-4] + '.npy'
                    dataset.append((img_dir, label, camid, clothid, mask))
                else:
                    dataset.append((img_dir, label, camid, clothid))
                pid2cloth[label, clothid] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2cloth

    def _process_dir_test(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_cloth = []
        query_dataset_diff_cloth = []
        gallery_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    # pid = pid2label[pid]
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothid = pid2label[pid] * 2
                        if self.test_with_mask:
                            mask = img_dir.replace('/rgb/', '/human_parsing/')[0:-4] + '.npy'
                            gallery_dataset.append((img_dir, pid, camid, clothid, mask))
                        else:
                            gallery_dataset.append((img_dir, pid, camid, clothid))
                    elif cam == 'B':
                        clothid = pid2label[pid] * 2
                        if self.test_with_mask:
                            mask = img_dir.replace('/rgb/', '/human_parsing/')[0:-4] + '.npy'
                            query_dataset_same_cloth.append((img_dir, pid, camid, clothid, mask))
                        else:
                            query_dataset_same_cloth.append((img_dir, pid, camid, clothid))
                    else:
                        clothid = pid2label[pid] * 2 + 1
                        if self.test_with_mask:
                            mask = img_dir.replace('/rgb/', '/human_parsing/')[0:-4] + '.npy'
                            query_dataset_diff_cloth.append((img_dir, pid, camid, clothid, mask))
                        else:
                            query_dataset_diff_cloth.append((img_dir, pid, camid, clothid))

        pid2imgidx = {}
        for idx, (img_dir, pid, camid, clothid) in enumerate(gallery_dataset):
            if pid not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)
        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
                 
        num_imgs_query_same = len(query_dataset_same_cloth)
        num_imgs_query_diff = len(query_dataset_diff_cloth)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_cloth, query_dataset_diff_cloth, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx
