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

import build_adjacency_matrix
import gnn_propagate


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
def batch_torch_topk(qf, gf, k1, N=6000):
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


def kr_re_ranking(probFea, galFea, probFea_id, galFea_id, probFea_cloth, galFea_cloth, k1, k2, lambda_j, lambda_ci, lambda_ic, lambda_cic):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    ori_feat = torch.cat([probFea, galFea]).cuda()
    id_feat = torch.cat([probFea_id, galFea_id]).cuda()
    cloth_feat = torch.cat([probFea_cloth, galFea_cloth]).cuda()
    ori_rank = batch_torch_topk(ori_feat, ori_feat, k1 + 1, N=6000)
    id_rank = batch_torch_topk(id_feat, id_feat, k1 + 1, N=6000)
    cloth_rank = batch_torch_topk(cloth_feat, cloth_feat, k1 + 1, N=6000)
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
    invIndex_ci = []

    for i in range(all_num):
        invIndex_ci.append(np.where(V_id[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    qnum = all_num if lambda_cic > 0.0 else query_num
    jaccard_dist_ci = np.zeros((qnum, all_num), dtype=np.float32)
    for i in range(qnum):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V_cloth[i, :] != 0)[0]
        indImages = [invIndex_ori[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_cloth[i, indNonZero[j]],
                                                                               V_id[indImages[j], indNonZero[j]])
        jaccard_dist_ci[i] = 1 - temp_min / (2. - temp_min)

    ## id->cloth Jaccard dist
    invIndex_ic = []

    for i in range(all_num):
        invIndex_ic.append(np.where(V_cloth[:, i] != 0)[0])
    # print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist_ic = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V_id[i, :] != 0)[0]
        indImages = [invIndex_ori[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_id[i, indNonZero[j]],
                                                                               V_cloth[indImages[j], indNonZero[j]])
        jaccard_dist_ic[i] = 1 - temp_min / (2. - temp_min)
    
    # three-step matching: cloth->id->cloth
    if lambda_cic > 0.0:
        clo_id_rank = batch_torch_topk_dist(jaccard_dist_ci, k1, N=6000)
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
        V_cloid = batch_v_dist(jaccard_dist_ci, R_cloid, all_num)
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
        
        jaccard_dist_cic = np.zeros((query_num, all_num), dtype=np.float32)
        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
            temp_max = np.zeros(shape=[1, all_num], dtype=np.float32)
            indNonZero = np.where(V_cloid[i, :] != 0)[0]
            indImages = [invIndex_cic[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V_cloid[i, indNonZero[j]],
                                                                                V_cloth[indImages[j], indNonZero[j]])
            jaccard_dist_cic[i] = 1 - temp_min / (2. - temp_min)

    jaccard_dist_ci = jaccard_dist_ci[:query_num, :]

    gc.collect()  # empty memory
    if lambda_cic > 0.0:
        jaccard_dist = lambda_ci * jaccard_dist_ci + lambda_ic * jaccard_dist_ic + \
            lambda_cic * jaccard_dist_cic
    else:
        jaccard_dist = lambda_ci * jaccard_dist_ci + lambda_ic * jaccard_dist_ic
    original_dist = batch_euclidean_distance(ori_feat, ori_feat[:query_num, :]).numpy()
    final_dist = original_dist + jaccard_dist_ori + jaccard_dist * lambda_j
    # print(jaccard_dist)
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    # print(final_dist)
    # print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    
    return final_dist


def gnn_reranking(qf, gf, qidf, gidf, qclothf, gclothf, k1, k2, lambda_gi, lambda_ci, lambda_ic, lambda_cic):
    query_num, gallery_num = qf.shape[0], gf.shape[0]

    uf = torch.cat((qf, gf), dim=0)
    o_score = torch.mm(uf, uf.t())
    # del X_u, X_q, X_g
    uidf = torch.cat((qidf, gidf), dim=0)
    id_score = torch.mm(uidf, uidf.t())
    uclothf = torch.cat((qclothf, gclothf), dim=0)
    cloth_score = torch.mm(uclothf, uclothf.t())

    # initial ranking list
    S_o, initial_rank_o = o_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    S_id, initial_rank_id = id_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    S_clo, initial_rank_clo = cloth_score.topktopk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A_o = build_adjacency_matrix.forward(initial_rank_o.float())   
    S_o = S_o * S_o
    A_id = build_adjacency_matrix.forward(initial_rank_id.float())   
    S_id = S_id * S_id
    A_clo = build_adjacency_matrix.forward(initial_rank_clo.float())   
    S_clo = S_clo * S_clo

    # stage 2
    if k2 != 1:      
        for i in range(2):
            # original feature
            A_o = A_o + A_o.T
            A_o = gnn_propagate.forward(A_o, initial_rank_o[:, :k2].contiguous().float(), S_o[:, :k2].contiguous().float())
            A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
            A_o = A_o.div(A_o_norm.expand_as(A_o))

            # id feature
            A_id = A_id + A_id.T
            A_id = gnn_propagate.forward(A_id, initial_rank_id[:, :k2].contiguous().float(), S_id[:, :k2].contiguous().float())
            A_id_norm = torch.norm(A_id, p=2, dim=1, keepdim=True)
            A_id = A_id.div(A_id_norm.expand_as(A_id))

            # clothes feature
            A_clo = A_clo + A_clo.T
            A_clo = gnn_propagate.forward(A_clo, initial_rank_clo[:, :k2].contiguous().float(), S_clo[:, :k2].contiguous().float())
            A_clo_norm = torch.norm(A_clo, p=2, dim=1, keepdim=True)
            A_clo = A_clo.div(A_clo_norm.expand_as(A_clo))                       
    
    original_distance = -o_score
    gnn_o_distance = -torch.mm(A_o[:query_num,], A_o[query_num:, ].t())
    gnn_ci_distance = -torch.mm(A_clo[:query_num, :], A_id[query_num:, ].t())
    gnn_ic_distance = -torch.mm(A_id[:query_num, :], A_clo[query_num:, ].t())
    gnn_cic_distance = -torch.mm(torch.mm(A_clo[:query_num, :], A_id.t()), A_clo[query_num:, ].t())

    gnn_distance = lambda_ci * gnn_ci_distance + lambda_ic * gnn_ic_distance + lambda_cic * gnn_cic_distance
    final_dist = original_distance + gnn_o_distance + lambda_gi * gnn_distance
    
    return final_dist