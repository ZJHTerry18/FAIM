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


class LTCC(object):
    """
    LTCC

    Reference:
    Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    
    Dataset statistics:
    # identities: 77 (train) + 75 (test)
    # images: 9576 (train) + 493 (query) + 7050 (gallery)
    """
    dataset_dir = 'ltcc'

    def __init__(self, root='data', train_with_mask=True, test_with_mask=False, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_with_mask = train_with_mask
        self.test_with_mask = test_with_mask
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.subset_txt = osp.join(self.dataset_dir, 'info/all_id_test.txt')

        self.train_mask_dir = osp.join(self.dataset_dir, 'human_parsing/train')
        self.query_mask_dir = osp.join(self.dataset_dir, 'human_parsing/query')
        self.gallery_mask_dir = osp.join(self.dataset_dir, 'human_parsing/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2cloth = self._process_dir_train()
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = self._process_dir_test()
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        print("=> LTCC loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        print("  ----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        print("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2cloth = pid2cloth

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if self.train_with_mask:
            if not osp.exists(self.train_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.train_mask_dir))
        if self.test_with_mask:
            if not osp.exists(self.query_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.query_mask_dir))
            if not osp.exists(self.gallery_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.gallery_mask_dir))

    def _process_dir_train(self):
        img_paths = glob.glob(osp.join(self.train_dir, '*.png'))
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = pattern = re.compile(r'(\w+)_c')

        pid_container = set()
        cloth_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            cloth_container.add(clothid)
        pid_container = sorted(pid_container)
        cloth_container = sorted(cloth_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cloth2label = {clothid:label for label, clothid in enumerate(cloth_container)}

        num_pids = len(pid_container)
        num_clothes = len(cloth_container)

        dataset = []
        pid2cloth = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothid = cloth2label[clothid]
            if self.train_with_mask:
                mask = osp.join(self.train_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                dataset.append((img_path, pid, camid, clothid, mask))
            else:
                dataset.append((img_path, pid, camid, clothid))
            pid2cloth[pid, clothid] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2cloth

    def _process_dir_test(self):
        query_img_paths = glob.glob(osp.join(self.query_dir, '*.png'))
        gallery_img_paths = glob.glob(osp.join(self.gallery_dir, '*.png'))
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = pattern = re.compile(r'(\w+)_c')

        with open(self.subset_txt, 'r') as f:
            subset_ids = f.readlines()
        subset_ids = list(map(int, subset_ids))

        pid_container = set()
        cloth_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            if pid in subset_ids:
                pid_container.add(pid)
                cloth_container.add(clothid)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            cloth_container.add(clothid)
        pid_container = sorted(pid_container)
        cloth_container = sorted(cloth_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cloth2label = {clothid:label for label, clothid in enumerate(cloth_container)}

        num_pids = len(pid_container)
        num_clothes = len(cloth_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            if pid in subset_ids:
                camid -= 1 # index starts from 0
                clothid = cloth2label[clothid]
                if self.test_with_mask:
                    mask = osp.join(self.query_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                    query_dataset.append((img_path, pid, camid, clothid, mask))
                else:
                    query_dataset.append((img_path, pid, camid, clothid))

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothid = cloth2label[clothid]
            if self.test_with_mask:
                mask = osp.join(self.gallery_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                gallery_dataset.append((img_path, pid, camid, clothid, mask))
            else:
                gallery_dataset.append((img_path, pid, camid, clothid))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes


class LTCC_SC(object):
    """
    LTCC

    Reference:
    Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    
    Dataset statistics:
    # identities: 77 (train) + 75 (test)
    # images: 9576 (train) + 493 (query) + 7050 (gallery)
    """
    dataset_dir = 'ltcc_sc'

    def __init__(self, root='data', train_with_mask=True, test_with_mask=False, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_with_mask = train_with_mask
        self.test_with_mask = test_with_mask
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.subset_txt = osp.join(self.dataset_dir, 'info/all_id_test.txt')

        self.train_mask_dir = osp.join(self.dataset_dir, 'human_parsing/train')
        self.query_mask_dir = osp.join(self.dataset_dir, 'human_parsing/query')
        self.gallery_mask_dir = osp.join(self.dataset_dir, 'human_parsing/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2cloth = self._process_dir_train()
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = self._process_dir_test()
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        print("=> LTCC loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        print("  ----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        print("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2cloth = pid2cloth

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if self.train_with_mask:
            if not osp.exists(self.train_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.train_mask_dir))
        if self.test_with_mask:
            if not osp.exists(self.query_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.query_mask_dir))
            if not osp.exists(self.gallery_mask_dir):
                raise RuntimeError("'{}' is not available".format(self.gallery_mask_dir))

    def _process_dir_train(self):
        img_paths = glob.glob(osp.join(self.train_dir, '*.png'))
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = pattern = re.compile(r'(\w+)_c')

        pid_container = set()
        cloth_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            cloth_container.add(clothid)
        pid_container = sorted(pid_container)
        cloth_container = sorted(cloth_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cloth2label = {clothid:label for label, clothid in enumerate(cloth_container)}

        num_pids = len(pid_container)
        num_clothes = len(cloth_container)

        dataset = []
        pid2cloth = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothid = cloth2label[clothid]
            if self.train_with_mask:
                mask = osp.join(self.train_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                dataset.append((img_path, pid, camid, clothid, mask))
            else:
                dataset.append((img_path, pid, camid, clothid))
            pid2cloth[pid, clothid] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2cloth

    def _process_dir_test(self):
        query_img_paths = glob.glob(osp.join(self.query_dir, '*.png'))
        gallery_img_paths = glob.glob(osp.join(self.gallery_dir, '*.png'))
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = pattern = re.compile(r'(\w+)_c')

        with open(self.subset_txt, 'r') as f:
            subset_ids = f.readlines()
        subset_ids = list(map(int, subset_ids))

        pid_container = set()
        cloth_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            if pid in subset_ids:
                pid_container.add(pid)
                cloth_container.add(clothid)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            cloth_container.add(clothid)
        pid_container = sorted(pid_container)
        cloth_container = sorted(cloth_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cloth2label = {clothid:label for label, clothid in enumerate(cloth_container)}

        num_pids = len(pid_container)
        num_clothes = len(cloth_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            if pid in subset_ids:
                camid -= 1 # index starts from 0
                clothid = cloth2label[clothid]
                if self.test_with_mask:
                    mask = osp.join(self.query_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                    query_dataset.append((img_path, pid, camid, clothid, mask))
                else:
                    query_dataset.append((img_path, pid, camid, clothid))

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothid = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothid = cloth2label[clothid]
            if self.test_with_mask:
                mask = osp.join(self.gallery_mask_dir, img_path.split('/')[-1][0:-4] + '.npy')
                gallery_dataset.append((img_path, pid, camid, clothid, mask))
            else:
                gallery_dataset.append((img_path, pid, camid, clothid))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes