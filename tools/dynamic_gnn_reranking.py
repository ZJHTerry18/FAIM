import numpy as np
import torch
import time
import gc
from tqdm import tqdm

import build_adjacency_matrix
import gnn_propagate

def gnn_reranking_ori(qf, gf, k1, k2):
    query_num, gallery_num = qf.shape[0], gf.shape[0]
    all_num = query_num + gallery_num

    uf = torch.cat((qf, gf), dim=0)
    o_score = torch.mm(uf, uf.t())

    # initial ranking list
    S_o, initial_rank_o = o_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A_o = build_adjacency_matrix.forward(initial_rank_o.float(), all_num)   
    S_o = S_o * S_o

    # stage 2
    if k2 != 1:      
        for i in range(2):
            # original feature
            A_o = A_o + A_o.T
            A_o = gnn_propagate.forward(A_o, initial_rank_o[:, :k2].contiguous().float(), S_o[:, :k2].contiguous().float())
            A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
            A_o = A_o.div(A_o_norm.expand_as(A_o))                       
    
    original_distance = -o_score[:query_num, query_num:]
    gnn_o_distance = -torch.mm(A_o[:query_num,], A_o[query_num:, ].t())

    final_dist = original_distance + gnn_o_distance

    final_dist = final_dist.cpu().numpy()
    
    return final_dist

def gnn_reranking_ori_lowcost(qf, gf, k1, k2):
    query_num, gallery_num = qf.shape[0], gf.shape[0]
    all_num = query_num + gallery_num

    uf = torch.cat((qf, gf), dim=0).cpu()
    o_score = torch.mm(uf, uf.t())

    # initial ranking list
    _, initial_rank_o = o_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A_o = torch.zeros((all_num, all_num), dtype=torch.float16)
    A_o = build_adjacency_matrix.forward(initial_rank_o.float().cuda(), all_num).cpu()   
    # S_o = S_o * S_o

    # stage 2
    A_o = A_o + A_o.T
    A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
    A_o = A_o.div(A_o_norm.expand_as(A_o)).float()
    # if k2 != 1:      
    #     for i in range(2):
    #         # original feature
    #         A_o = A_o + A_o.T
    #         A_o = gnn_propagate.forward(A_o, initial_rank_o[:, :k2].contiguous().float(), S_o[:, :k2].contiguous().float())
    #         A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
    #         A_o = A_o.div(A_o_norm.expand_as(A_o))                       
    
    original_distance = -o_score[:query_num, query_num:].numpy()
    gnn_o_distance = -torch.mm(A_o[:query_num,], A_o[query_num:, ].t()).numpy()

    final_dist = original_distance + gnn_o_distance
    
    return final_dist

def dynamic_gnn_reranking(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothids, g_clothids, idrel, k1, k2):
    query_num, gallery_num = qf.shape[0], gf.shape[0]
    all_num = query_num + gallery_num

    all_pids = np.concatenate([q_pids, g_pids])
    all_clothids = np.concatenate([q_clothids, g_clothids])
    uf = torch.cat((qf, gf), dim=0)
    uidf = torch.cat((qidf, gidf), dim=0)
    uclothf = torch.cat((qclothf, gclothf), dim=0)
    o_score = torch.mm(uf, uf.t())
    id_score = torch.mm(uidf, uidf.t())
    cloth_score = torch.mm(uclothf, uclothf.t())

    # preclude same clothes samples
    for i in range(all_num):
        cloth_score[i, all_clothids == all_clothids[i]] = -1e5

    # initial ranking list
    S_o, initial_rank_o = o_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    S_id, initial_rank_id = id_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    S_clo, initial_rank_clo = cloth_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A_o = build_adjacency_matrix.forward(initial_rank_o.float(), all_num)   
    S_o = S_o * S_o
    A_id = build_adjacency_matrix.forward(initial_rank_id.float(), all_num)   
    S_id = S_id * S_id
    A_clo = build_adjacency_matrix.forward(initial_rank_clo.float(), all_num)   
    S_clo = S_clo * S_clo

    dyn_weight_ci = torch.zeros(all_num, all_num).cuda()
    invindex_ci = []
    for i in range(all_num):
        invindex_ci.append(torch.where(A_id[:, i] > 0)[0])
    for i in range(all_num):
        num_intermediary = torch.zeros(all_num).cuda()
        sum_dyn_weight = torch.zeros(all_num).cuda()
        ind_nonzero = torch.where(A_clo[i, :] > 0)[0]
        ind_images = [invindex_ci[ind] for ind in ind_nonzero]

        for j in range(len(ind_nonzero)):
            num_intermediary[ind_images[j]] += 1
            sim_clof = (torch.matmul(uclothf[i], uclothf[ind_nonzero[j]].reshape(-1, 1)) + 1) * 0.5
            sim_idf = (torch.matmul(uidf[ind_nonzero[j]], uidf[ind_images[j]].t()) + 1) * 0.5
            rel_intermediary = idrel[ind_nonzero[j]]
            sum_dyn_weight[ind_images[j]] = sum_dyn_weight[ind_images[j]] + ((sim_clof * sim_idf) ** 0.5) * rel_intermediary
        indmask = num_intermediary[0] > 0
        dyn_weight_ci[i, indmask] = sum_dyn_weight[indmask] / num_intermediary[indmask]

    dyn_weight_ic = torch.zeros(all_num, all_num).cuda()
    invindex_ic = []
    for i in range(all_num):
        invindex_ic.append(torch.where(A_clo[:, i] > 0)[0])
    for i in range(all_num):
        num_intermediary = torch.zeros(all_num).cuda()
        sum_dyn_weight = torch.zeros(all_num).cuda()
        ind_nonzero = torch.where(A_id[i, :] > 0)[0]
        ind_images = [invindex_ic[ind] for ind in ind_nonzero]

        for j in range(len(ind_nonzero)):
            num_intermediary[ind_images[j]] += 1
            sim_clof = (torch.matmul(uclothf[ind_nonzero[j]], uclothf[ind_images[j]].t()) + 1) * 0.5
            sim_idf = (torch.matmul(uidf[i], uidf[ind_nonzero[j]].reshape(-1, 1)) + 1) * 0.5
            rel_intermediary = idrel[ind_nonzero[j]]
            sum_dyn_weight[ind_images[j]] = sum_dyn_weight[ind_images[j]] + ((sim_clof * sim_idf) ** 0.5) * rel_intermediary
        indmask = num_intermediary[0] > 0
        dyn_weight_ic[i, indmask] = sum_dyn_weight[indmask] / num_intermediary[indmask]
    
    dyn_weight_cic = torch.zeros(all_num, all_num).cuda()

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
    
    original_distance = -o_score[:query_num, query_num:]
    gnn_o_distance = -torch.mm(A_o[:query_num,], A_o[query_num:, ].t())
    gnn_ci_distance = -torch.mm(A_clo[:query_num, :], A_id[query_num:, ].t())
    gnn_ic_distance = -torch.mm(A_id[:query_num, :], A_clo[query_num:, ].t())
    gnn_cic_distance = -torch.mm(torch.mm(A_clo[:query_num, :], A_id.t()), A_clo[query_num:, ].t())
    dyn_weight_ci = dyn_weight_ci[:query_num, query_num:]
    dyn_weight_ic = dyn_weight_ic[:query_num, query_num:]
    dyn_weight_cic = dyn_weight_cic[:query_num, query_num:]
    dyn_weight_ori = 1.0 - (dyn_weight_ci + dyn_weight_ic + dyn_weight_cic) / 3

    gnn_distance = dyn_weight_ci * gnn_ci_distance + dyn_weight_ic * gnn_ic_distance + dyn_weight_cic * gnn_cic_distance
    final_dist = (original_distance + gnn_o_distance) * dyn_weight_ori + gnn_distance / 3

    final_dist = final_dist.cpu().numpy()
    
    return final_dist

def dynamic_gnn_reranking_lowcost(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothids, g_clothids, idrel, k1, k2):
    query_num, gallery_num = qf.shape[0], gf.shape[0]
    all_num = query_num + gallery_num
    N = 15000
    # o_score = np.zeros((all_num, all_num), dtype=np.float16)
    # id_score = np.zeros((all_num, all_num), dtype=np.float16)
    # cloth_score = np.zeros((all_num, all_num), dtype=np.float16)
    start = time.time()
    uf = torch.cat((qf, gf), dim=0).cpu()
    uidf = torch.cat((qidf, gidf), dim=0).cpu()
    uclothf = torch.cat((qclothf, gclothf), dim=0).cpu()
    o_score = torch.matmul(uf, uf.t())
    id_score = torch.matmul(uidf, uidf.t())
    cloth_score = torch.matmul(uclothf, uclothf.t())
    # for j in range(all_num // N + 1):
    #     o_score[j * N:j * N + N, j * N:j * N + N] = np.matmul(uf[j * N:j * N + N, :], uf[j * N:j * N + N, :].T)
    #     id_score[j * N:j * N + N, j * N:j * N + N] = np.matmul(uidf[j * N:j * N + N, :], uidf[j * N:j * N + N, :].T)
    #     cloth_score[j * N:j * N + N, j * N:j * N + N] = np.matmul(uclothf[j * N:j * N + N, :], uclothf[j * N:j * N + N, :].T)
    # o_score = torch.from_numpy(o_score)
    # id_score = torch.from_numpy(id_score)
    # cloth_score = torch.from_numpy(cloth_score)

    # initial ranking list
    _, initial_rank_o = o_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    _, initial_rank_id = id_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    _, initial_rank_clo = cloth_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    print('initial ranking finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    start = time.time()
    
    # stage 1
    A_o = torch.zeros((all_num, all_num), dtype=torch.float16)
    A_id = torch.zeros((all_num, all_num), dtype=torch.float16)
    A_clo = torch.zeros((all_num, all_num), dtype=torch.float16)
    # S_o = S_o * S_o
    # S_id = S_id * S_id
    # S_clo = S_clo * S_clo
    for j in range(all_num // N + 1):
        A_o[j * N: min(j * N + N, all_num), :] = (build_adjacency_matrix.forward((initial_rank_o[j * N:j * N + N, :]).float().cuda(), all_num)).cpu()
        A_id[j * N: min(j * N + N, all_num), :] = (build_adjacency_matrix.forward((initial_rank_id[j * N:j * N + N, :] ).float().cuda(), all_num)).cpu()
        A_clo[j * N: min(j * N + N, all_num), :] = (build_adjacency_matrix.forward((initial_rank_clo[j * N:j * N + N, :]).float().cuda(), all_num)).cpu()
    print('adjacency matrix computing finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    start = time.time()

    dyn_weight_ci = torch.zeros(all_num, all_num)
    invindex_ci = []
    for i in range(all_num):
        invindex_ci.append(torch.where(A_id[:, i] > 0)[0])
    for i in range(all_num):
        num_intermediary = torch.zeros(all_num)
        sum_dyn_weight = torch.zeros(all_num)
        ind_nonzero = torch.where(A_clo[i, :] > 0)[0]
        ind_images = [invindex_ci[ind] for ind in ind_nonzero]

        for j in range(len(ind_nonzero)):
            num_intermediary[ind_images[j]] += 1
            sim_clof = (torch.matmul(uclothf[i], uclothf[ind_nonzero[j]].reshape(-1, 1)) + 1) * 0.5
            sim_idf = (torch.matmul(uidf[ind_nonzero[j]], uidf[ind_images[j]].t()) + 1) * 0.5
            rel_intermediary = idrel[ind_nonzero[j]]
            sum_dyn_weight[ind_images[j]] = sum_dyn_weight[ind_images[j]] + ((sim_clof * sim_idf) ** 0.5) * rel_intermediary
        indmask = num_intermediary[0] > 0
        dyn_weight_ci[i, indmask] = sum_dyn_weight[indmask] / num_intermediary[indmask]

    dyn_weight_ic = torch.zeros(all_num, all_num)
    invindex_ic = []
    for i in range(all_num):
        invindex_ic.append(torch.where(A_clo[:, i] > 0)[0])
    for i in range(all_num):
        num_intermediary = torch.zeros(all_num)
        sum_dyn_weight = torch.zeros(all_num)
        ind_nonzero = torch.where(A_id[i, :] > 0)[0]
        ind_images = [invindex_ic[ind] for ind in ind_nonzero]

        for j in range(len(ind_nonzero)):
            num_intermediary[ind_images[j]] += 1
            sim_clof = (torch.matmul(uclothf[ind_nonzero[j]], uclothf[ind_images[j]].t()) + 1) * 0.5
            sim_idf = (torch.matmul(uidf[i], uidf[ind_nonzero[j]].reshape(-1, 1)) + 1) * 0.5
            rel_intermediary = idrel[ind_nonzero[j]]
            sum_dyn_weight[ind_images[j]] = sum_dyn_weight[ind_images[j]] + ((sim_clof * sim_idf) ** 0.5) * rel_intermediary
        indmask = num_intermediary[0] > 0
        dyn_weight_ic[i, indmask] = sum_dyn_weight[indmask] / num_intermediary[indmask]
    
    dyn_weight_cic = torch.zeros(all_num, all_num)

    # stage 2
    A_o = A_o + A_o.T
    A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
    A_o = A_o.div(A_o_norm.expand_as(A_o)).float()
    A_id = A_id + A_id.T
    A_id_norm = torch.norm(A_id, p=2, dim=1, keepdim=True)
    A_id = A_id.div(A_id_norm.expand_as(A_id)).float()
    A_clo = A_clo + A_clo.T
    A_clo_norm = torch.norm(A_clo, p=2, dim=1, keepdim=True)
    A_clo = A_clo.div(A_clo_norm.expand_as(A_clo)).float()
    # if k2 != 1:      
    #     for i in range(2):
    #         # original feature
    #         A_o = gnn_propagate.forward(A_o, initial_rank_o[:, :k2].contiguous().half().cuda(), S_o[:, :k2].contiguous().half().cuda())
    #         A_o_norm = torch.norm(A_o, p=2, dim=1, keepdim=True)
    #         A_o = A_o.div(A_o_norm.expand_as(A_o))

    #         # id feature
    #         A_id = gnn_propagate.forward(A_id, initial_rank_id[:, :k2].contiguous().half().cuda(), S_id[:, :k2].contiguous().half().cuda())
    #         A_id_norm = torch.norm(A_id, p=2, dim=1, keepdim=True)
    #         A_id = A_id.div(A_id_norm.expand_as(A_id))

    #         # clothes feature
    #         A_clo = gnn_propagate.forward(A_clo, initial_rank_clo[:, :k2].contiguous().half().cuda(), S_clo[:, :k2].contiguous().half().cuda())
    #         A_clo_norm = torch.norm(A_clo, p=2, dim=1, keepdim=True)
    #         A_clo = A_clo.div(A_clo_norm.expand_as(A_clo))                       
    
    original_distance = -o_score[:query_num, query_num:].numpy()
    gnn_o_distance = -torch.matmul(A_o[:query_num, :], A_o[query_num:, :].T).numpy()
    print('gnn original distance computing finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    start = time.time()
    gnn_ci_distance = -torch.matmul(A_clo[:query_num, :], A_id[query_num:, ].T).numpy()
    print('gnn ci distance computing finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    start = time.time()
    gnn_ic_distance = -torch.matmul(A_id[:query_num, :], A_clo[query_num:, ].T).numpy()
    print('gnn ic distance computing finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    start = time.time()
    gnn_cic_distance = -torch.matmul(torch.matmul(A_clo[:query_num, :], A_id.T), A_clo[query_num:, ].T).numpy()
    dyn_weight_ci = dyn_weight_ci[:query_num, query_num:].numpy()
    dyn_weight_ic = dyn_weight_ic[:query_num, query_num:].numpy()
    dyn_weight_cic = dyn_weight_cic[:query_num, query_num:].numpy()
    dyn_weight_ori = 1.0 - (dyn_weight_ci + dyn_weight_ic + dyn_weight_cic) / 3  

    gnn_distance = dyn_weight_ci * gnn_ci_distance + dyn_weight_ic * gnn_ic_distance + dyn_weight_cic * gnn_cic_distance
    final_dist = (original_distance + gnn_o_distance) * dyn_weight_ori + gnn_distance / 3
    print('distance computing finished. Time elapsed: {:.1f}s'.format(time.time() - start))
    
    return final_dist