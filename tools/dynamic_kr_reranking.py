# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Dec, 25 May 2019 20:29:09
# Faster version for kesci ReID challenge

# @author: luohao
# """

# """
# CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
# url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
# Matlab version: https://github.com/zhunzhong07/person-re-ranking
# """

# """
# API

# probFea: all feature vectors of the query set (torch tensor)
# probFea: all feature vectors of the gallery set (torch tensor)
# k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
# MemorySave: set to 'True' when using MemorySave mode
# Minibatch: avaliable when 'MemorySave' is 'True'
# """

# Save memory version

import numpy as np
import torch
import time
import gc
from tqdm import tqdm
import os

def sigmoid(x):
    x = x * 2 - 1
    return 1 / (1 + np.exp(-x))

def euclidean_distance(qf, gf):

    m = qf.shape[0]
    n = gf.shape[0]

    # dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
    #     torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    # dist_mat.addmm_(1,-2,qf,gf.t())

    # for L2-norm feature
    dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
    return dist_mat


def batch_euclidean_distance(qf, gf, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        dist_mat.append(temp_qd.t().cpu())
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    dist_mat = torch.cat(dist_mat, dim=0)
    return dist_mat


# 将topK排序放到GPU里运算，并且只返回k1+1个结果
# Compute TopK in GPU and return (k1+1) results
def batch_torch_topk(qf, gf, all_pids, all_clothids, all_aps, k1, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()

        # preclude the same person with same clothes
        preclude_portion = 1.0
        if all_clothids is not None:
            for qi in range(temp_qd.shape[0]):
                preclude_mask = torch.from_numpy((all_clothids == all_clothids[j * N + qi]))
                true_indices = torch.nonzero(preclude_mask)
                num_to_flip = int(len(true_indices) * (1.0 - preclude_portion))
                indices_to_flip = torch.randperm(len(true_indices))[:num_to_flip]
                preclude_mask[true_indices[indices_to_flip]] = False
                
                temp_qd[qi, preclude_mask] = 1e9
        
        # preclude persons with high id-reliability
        preclude_portion = 0.5
        if all_aps is not None:
            all_aps = [0.0 if x is None else x for x in all_aps]
            top_indices = sorted(range(len(all_aps)), key=lambda i: all_aps[i], reverse=True)[:int(preclude_portion * len(all_aps))]
            preclude_mask = [False] * len(all_aps)
            for i in top_indices:
                preclude_mask[i] = True
            preclude_mask = torch.as_tensor(preclude_mask)
            for qi in range(temp_qd.shape[0]):
                temp_qd[qi, preclude_mask] = 1e9

        initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    initial_rank = torch.cat(initial_rank, dim=0).cpu().numpy()
    return initial_rank

def batch_torch_topk_dist(distmat, k1, N=6000):
    initial_rank = []
    distmat = torch.from_numpy(distmat)
    n = distmat.shape[0]
    for j in range(n // N + 1):
        temp_d = distmat[j * N: j * N + N]
        initial_rank.append(torch.topk(temp_d, k=k1, dim=1, largest=False, sorted=True)[1])
    initial_rank = torch.cat(initial_rank, dim=0)
    # print('initial_rank', initial_rank.shape)
    return initial_rank

def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in range(m):
        temp_gf = feat[i].unsqueeze(0)
        # temp_qd = []
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (torch.max(temp_qd))
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd[R[i]]
        weight = torch.exp(-temp_qd)
        weight = (weight / torch.sum(weight)).cpu().numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V

def batch_v_dist(dist, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    dist = torch.from_numpy(dist)
    m = dist.shape[0]
    for i in range(m):
        temp_d = dist[i]
        temp_d = temp_d[R[i]]
        weight = torch.exp(-temp_d)
        weight = (weight / torch.sum(weight)).cpu().numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def dynamic_kr_re_ranking(dataset, probFea, galFea, probFea_id, galFea_id, probFea_cloth, galFea_cloth, 
                  q_pids, g_pids, q_clothids, g_clothids, id_rel, id_ap=None, k1=30, k2=6):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    dataset_all = dataset.query + dataset.gallery if hasattr(dataset, 'query') else dataset.query_diff + dataset.gallery
    pid_select = [25]
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    all_pids = np.concatenate([q_pids, g_pids])
    all_clothids = np.concatenate([q_clothids, g_clothids])
    ori_feat = torch.cat([probFea, galFea]).cuda()
    id_feat = torch.cat([probFea_id, galFea_id]).cuda()
    cloth_feat = torch.cat([probFea_cloth, galFea_cloth]).cuda()
    ori_rank = batch_torch_topk(ori_feat, ori_feat, all_pids, all_clothids=None, all_aps=None, k1=k1 + 1, N=6000)
    id_rank = batch_torch_topk(id_feat, id_feat, all_pids, all_clothids=None, all_aps=None, k1=k1 + 1, N=6000)
    cloth_rank = batch_torch_topk(cloth_feat, cloth_feat, all_pids, all_clothids=None, all_aps=None, k1=k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    # print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    # print('starting re_ranking')

    R_ori = []
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(ori_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(ori_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R_ori.append(k_reciprocal_expansion_index)
    V_ori = batch_v(ori_feat, R_ori, all_num)
    del R_ori
    gc.collect()

    R_id = []
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(id_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(id_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R_id.append(k_reciprocal_expansion_index)
    V_id = batch_v(id_feat, R_id, all_num)
    del R_id
    gc.collect()
    
    R_cloth = []
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(cloth_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(cloth_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R_cloth.append(k_reciprocal_expansion_index)
    V_cloth = batch_v(cloth_feat, R_cloth, all_num)
    del R_cloth
    gc.collect()

    # print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    ori_rank = ori_rank[:, :k2]
    id_rank = id_rank[:, :k2]
    cloth_rank = cloth_rank[:, :k2]

    ### 下面这个版本速度更快
    ### Faster version
    if k2 != 1:
        V_qeori = np.zeros_like(V_ori, dtype=np.float16)
        V_qeid = np.zeros_like(V_id, dtype=np.float16)
        V_qecloth = np.zeros_like(V_cloth, dtype=np.float16)
        for i in range(all_num):
            V_qeori[i, :] = np.mean(V_ori[ori_rank[i], :], axis=0)
            V_qeid[i, :] = np.mean(V_id[id_rank[i], :], axis=0)
            V_qecloth[i, :] = np.mean(V_cloth[cloth_rank[i], :], axis=0)
        V_ori = V_qeori
        V_id = V_qeid
        V_cloth = V_qecloth
    del ori_rank, id_rank, cloth_rank

    ### 下面这个版本更省内存(约40%)，但是更慢
    ### Low-memory version
    '''gc.collect()  # empty memory
    N = 2000
    for j in range(all_num // N + 1):

        if k2 != 1:
            V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
            V[:, j * N:j * N + N] = V_qe
            del V_qe
    del initial_rank'''

    gc.collect()  # empty memory
    # print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))

    ## original feature Jaccard dist
    invIndex_ori = []

    for i in range(all_num):
        invIndex_ori.append(np.where(V_ori[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist_ori = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V_ori[i, :] != 0)[0]
        indImages = [invIndex_ori[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_ori[i, indNonZero[j]],
                                                                               V_ori[indImages[j], indNonZero[j]])
        jaccard_dist_ori[i] = 1 - temp_min / (2. - temp_min)

    ## cloth->id Jaccard dist
    qnum = all_num #if lambda_cic > 0.0 else query_num
    jaccard_dist_ci = np.zeros((qnum, all_num), dtype=np.float32)
    dyn_weight_ci = np.zeros((qnum, all_num), dtype=np.float32)
    invIndex_ci = []

    for i in range(all_num):
        invIndex_ci.append(np.where(V_id[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))
    for i in range(qnum):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        num_intermediary = np.zeros(shape=[1, all_num])
        sum_dyn_weight = np.zeros(shape=[1, all_num])
        indNonZero = np.where(V_cloth[i, :] != 0)[0]
        indImages = [invIndex_ci[ind] for ind in indNonZero]

        intermediary_list = [set()] * all_num
        # calculate jaccard distance
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_cloth[i, indNonZero[j]],
                                                                            V_id[indImages[j], indNonZero[j]])
            num_intermediary[0, indImages[j]] += 1
            sim_clof = (torch.matmul(cloth_feat[i], cloth_feat[indNonZero[j]].reshape(-1, 1)).cpu().numpy() + 1) * 0.5
            sim_idf = (torch.matmul(id_feat[indNonZero[j]], id_feat[indImages[j]].t()).cpu().numpy() + 1) * 0.5
            rel_intermediary = id_rel[indNonZero[j]]
            rel_gallery = id_rel[indImages[j]]
            sum_dyn_weight[0, indImages[j]] = sum_dyn_weight[0, indImages[j]] + \
                ((sim_clof * sim_idf) ** 0.5) * rel_intermediary #((rel_intermediary * rel_gallery) ** 0.5)

            for k in indImages[j].tolist():
                intermediary_list[k].add(indNonZero[j])
        # ## For intermediary visualization
        # for k in range(query_num, all_num):
        #     if int(all_pids[i]) in pid_select and int(all_pids[k]) in pid_select and int(all_clothids[i]) != int(all_clothids[k]):
        #         if os.path.basename(dataset_all[i][0]) == '025_12_c10_008123.png':
        #             print('------------------------')
        #             print('query: %s' % (os.path.basename(dataset_all[i][0])))
        #             print('gallery: %s' % (os.path.basename(dataset_all[k][0])))
        #             print('intermediaries:')
        #             for m in intermediary_list[k]:
        #                 print(os.path.basename(dataset_all[m][0]))
        #             print('------------------------')

            ##
        indMask = (num_intermediary[0] > 0)
        dyn_weight_ci[i, indMask] = sum_dyn_weight[0, indMask] / num_intermediary[0, indMask]
        jaccard_dist_ci[i] = 1 - temp_min / (2. - temp_min)

    ## id->cloth Jaccard dist
    jaccard_dist_ic = np.zeros((query_num, all_num), dtype=np.float32)
    dyn_weight_ic = np.zeros((query_num, all_num), dtype=np.float32)
    invIndex_ic = []

    for i in range(all_num):
        invIndex_ic.append(np.where(V_cloth[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        num_intermediary = np.zeros(shape=[1, all_num])
        sum_dyn_weight = np.zeros(shape=[1, all_num])
        indNonZero = np.where(V_id[i, :] != 0)[0]
        indImages = [invIndex_ic[ind] for ind in indNonZero]

        intermediary_list = [set()] * all_num
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_id[i, indNonZero[j]],
                                                                            V_cloth[indImages[j], indNonZero[j]])
            num_intermediary[0, indImages[j]] += 1
            sim_idf = (torch.matmul(id_feat[i], id_feat[indNonZero[j]].reshape(-1, 1)).cpu().numpy() + 1) * 0.5
            sim_clof = (torch.matmul(cloth_feat[indNonZero[j]], cloth_feat[indImages[j]].t()).cpu().numpy() + 1) * 0.5
            rel_intermediary = id_rel[indNonZero[j]]
            rel_query = id_rel[i]
            sum_dyn_weight[0, indImages[j]] = sum_dyn_weight[0, indImages[j]] + \
                ((sim_clof * sim_idf) ** 0.5) * rel_intermediary #((rel_intermediary * rel_query) ** 0.5)
            
            for k in indImages[j].tolist():
                intermediary_list[k].add(indNonZero[j])
        # ## For intermediary visualization
        # for k in range(query_num, all_num):
        #     if int(all_pids[i]) in pid_select and int(all_pids[k]) in pid_select and int(all_clothids[i]) != int(all_clothids[k]):
        #         if os.path.basename(dataset_all[i][0]) == '025_12_c10_008123.png':
        #             print('------------------------')
        #             print('query: %s' % (os.path.basename(dataset_all[i][0])))
        #             print('gallery: %s' % (os.path.basename(dataset_all[k][0])))
        #             print('intermediaries:')
        #             for m in intermediary_list[k]:
        #                 print(os.path.basename(dataset_all[m][0]))
        #             print('------------------------')
        indMask = (num_intermediary[0] > 0)
        dyn_weight_ic[i, indMask] = sum_dyn_weight[0, indMask] / num_intermediary[0, indMask]
        jaccard_dist_ic[i] = 1 - temp_min / (2. - temp_min)
    
    # three-step matching: cloth->id->cloth
    jaccard_dist_cic = np.zeros((query_num, all_num), dtype=np.float32)
    dyn_weight_cic = np.zeros((query_num, all_num), dtype=np.float32)
    clo_id_rank = batch_torch_topk_dist(jaccard_dist_ci * dyn_weight_ci, k1, N=6000)
    R_cloid = []
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(clo_id_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(clo_id_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R_cloid.append(k_reciprocal_expansion_index)
    V_cloid = batch_v_dist(jaccard_dist_ci * dyn_weight_ci, R_cloid, all_num)
    del R_cloid
    gc.collect()

    clo_id_rank = clo_id_rank[:, :k2]
    if k2 != 1:
        V_qecloid = np.zeros_like(V_ori, dtype=np.float16)
        for i in range(all_num):
            V_qecloid[i, :] = np.mean(V_cloid[clo_id_rank[i], :], axis=0)
        V_cloid = V_qecloid
    del clo_id_rank

    # three-step Jaccard dist
    invIndex_cic = []
    for i in range(all_num):
        invIndex_cic.append(np.where(V_cloth[:, i] != 0)[0])
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        num_intermediary = np.zeros(shape=[1, all_num])
        sum_dyn_weight = np.zeros(shape=[1, all_num])
        indNonZero = np.where(V_cloid[i, :] != 0)[0]
        indImages = [invIndex_cic[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_cloid[i, indNonZero[j]],
                                                                            V_cloth[indImages[j], indNonZero[j]])
            num_intermediary[0, indImages[j]] += 1
            sim_idf = (torch.matmul(id_feat[i], id_feat[indNonZero[j]].reshape(-1, 1)).cpu().numpy() + 1) * 0.5
            sim_clof = (torch.matmul(cloth_feat[indNonZero[j]], cloth_feat[indImages[j]].t()).cpu().numpy() + 1) * 0.5
            rel_intermediary = id_rel[indNonZero[j]]
            rel_gallery = id_rel[indImages[j]]
            sum_dyn_weight[0, indImages[j]] = sum_dyn_weight[0, indImages[j]] + \
                ((sim_clof * sim_idf) ** 1.0) * rel_intermediary
        indMask = (num_intermediary[0] > 0)
        # dyn_weight_cic[i, indMask] = sum_dyn_weight[0, indMask] / num_intermediary[0, indMask]
        jaccard_dist_cic[i] = 1 - temp_min / (2. - temp_min)

    jaccard_dist_ci = jaccard_dist_ci[:query_num, :]
    dyn_weight_ci = dyn_weight_ci[:query_num, :]

    gc.collect()  # empty memory
    # dyn_weight_ci = sigmoid(dyn_weight_ci)
    # dyn_weight_ic = sigmoid(dyn_weight_ic)
    # dyn_weight_cic = sigmoid(dyn_weight_cic)
    jaccard_dist = dyn_weight_ci * jaccard_dist_ci + dyn_weight_ic * jaccard_dist_ic + \
            dyn_weight_cic * jaccard_dist_cic
    # jaccard_dist = 0.3 * jaccard_dist_ci + 0.6 * jaccard_dist_ic + 0.1 * jaccard_dist_cic
    original_dist = batch_euclidean_distance(ori_feat, ori_feat[:query_num, :]).numpy()
    dyn_weight_ori = 1.0 - (dyn_weight_ci + dyn_weight_ic + dyn_weight_cic) / 3 
    # print(dyn_weight_ci.mean(), dyn_weight_ic.mean(), dyn_weight_cic.mean(), dyn_weight_ori.mean())
    final_dist = (original_dist + jaccard_dist_ori) * dyn_weight_ori + jaccard_dist / 3 
    # print(jaccard_dist)
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    # print(final_dist)
    # print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    
    return final_dist

def kr_re_ranking_ori(probFea, galFea, k1, k2):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    ori_feat = torch.cat([probFea, galFea]).cuda()
    ori_rank = batch_torch_topk(ori_feat, ori_feat, k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    # print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    # print('starting re_ranking')

    R_ori = []
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(ori_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(ori_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R_ori.append(k_reciprocal_expansion_index)
    V_ori = batch_v(ori_feat, R_ori, all_num)
    del R_ori
    gc.collect()

    # print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    ori_rank = ori_rank[:, :k2]

    ### 下面这个版本速度更快
    ### Faster version
    if k2 != 1:
        V_qeori = np.zeros_like(V_ori, dtype=np.float16)
        for i in range(all_num):
            V_qeori[i, :] = np.mean(V_ori[ori_rank[i], :], axis=0)
        V_ori = V_qeori
    del ori_rank

    ### 下面这个版本更省内存(约40%)，但是更慢
    ### Low-memory version
    '''gc.collect()  # empty memory
    N = 2000
    for j in range(all_num // N + 1):

        if k2 != 1:
            V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
            V[:, j * N:j * N + N] = V_qe
            del V_qe
    del initial_rank'''

    gc.collect()  # empty memory
    # print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))

    ## original feature Jaccard dist
    invIndex_ori = []

    for i in range(all_num):
        invIndex_ori.append(np.where(V_ori[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist_ori = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V_ori[i, :] != 0)[0]
        indImages = [invIndex_ori[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_ori[i, indNonZero[j]],
                                                                               V_ori[indImages[j], indNonZero[j]])
        jaccard_dist_ori[i] = 1 - temp_min / (2. - temp_min)

    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(ori_feat, ori_feat[:query_num, :]).numpy()
    final_dist = original_dist + jaccard_dist_ori
    # print(jaccard_dist)
    del original_dist

    final_dist = final_dist[:query_num, query_num:]
    # print(final_dist)
    # print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    
    return final_dist

