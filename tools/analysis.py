import numpy as np
import torch
import csv
from tqdm import tqdm

def knnsearch(qf, gfs, k=100):
    '''
        qf: d-dimension array
        gfs: N_g * d
    '''
    assert k <= gfs.shape[0]
    dist = 1.0 - np.matmul(gfs, qf)
    distk = np.sort(dist)[:k].tolist()
    topk = np.argsort(dist)[:k].tolist()
    return topk, distk

def rate(qf, gfs, q_pid, g_pids, q_clothid, g_clothids, dataset, K):
    qtopk, qdistk = knnsearch(qf, gfs, k=K)
    TP_sc = []
    TP_cc = []
    FP = []
    TPR_sc = 0.0
    TPR_cc = 0.0
    FPR = 0.0
    TP_sc_avgdist = 0.0
    TP_cc_avgdist = 0.0
    FP_avgdist = 0.0
    for i, gi in enumerate(qtopk):
        if q_pid == g_pids[gi] and q_clothid == g_clothids[gi]:
            TP_sc.append(gi)
            TPR_sc += 1.0 / K
            TP_sc_avgdist += qdistk[i]
        elif q_pid == g_pids[gi] and q_clothid != g_clothids[gi]:
            TP_cc.append(gi)
            TPR_cc += 1.0 / K
            TP_cc_avgdist += qdistk[i]
        else:
            FP.append(gi)
            FPR += 1.0 / K
            FP_avgdist += qdistk[i]
    
    return TP_sc, TP_cc, FP, TPR_sc, TPR_cc, FPR, TP_sc_avgdist, TP_cc_avgdist, FP_avgdist
    
def analysis(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothids, g_clothids, dataset, query_id):
    queryset = dataset.query
    galleryset = dataset.gallery
    q_num, g_num = qf.shape[0], gf.shape[0]
    K = 100

    query_id_list = np.unique(q_pids)
    csv_data_1 = []
    csv_data_2 = []
    for query_id in tqdm(query_id_list):
        avg_oTPR_list = []
        avg_ooTPR_list = []
        avg_ciTPR_list = []
        avg_cloidTPR_list = []
        avg_idcloTPR_list = []
        avg_cloidcloTPR_list = []
        avg_idTPR_cc_list = []
        avg_cloTPR_sc_list = []

        avg_oRR_list = []
        avg_ooRR_list = []
        avg_ciRR_list = []

        avgdist_cloTP_list = []
        avgdist_cloFP_list = []
        avgnum_ooTP_list = []
        avgnum_ooFP_list = []
        avgnum_cloidTP_list = []
        avgnum_cloidFP_list = []
        avgnum_idcloTP_list = []
        avgnum_idcloFP_list = []
        avgnum_cloidcloTP_list = []
        avgnum_cloidcloFP_list = []
        for i in range(q_num):
            if q_pids[i] == query_id:
                ## Part 1: verify the effect of intermediate 
                # original
                oo_TPRs = []
                o_TP_sc, o_TP_cc, qo_FP, qo_TPR_sc, qo_TPR_cc, qo_FPR, qo_TP_sc_avgdist, qo_TP_cc_avgdist, qo_FP_avgdist \
                = rate(qf[i], gf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                qo_TP_sc, qo_TP_cc, qo_FP, qo_TPR_sc, _, qo_FPR, qo_TP_sc_avgdist, qo_TP_cc_avgdist, qo_FP_avgdist \
                = rate(qf[i], gf, q_pids[i], g_pids, q_clothids[i], g_clothids, galleryset, K=K)
                oo_TPs = set()
                oo_FPs = set()
                for ii in (qo_TP_sc + qo_TP_cc):
                    oo_TP_sc, oo_TP_cc, oo_FP, oo_TPR_sc, oo_TPR_cc, oo_FPR, _, _, _ \
                    = rate(gf[ii], gf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    oo_TPRs.append(oo_TPR_cc)
                    for ci in oo_TP_cc:
                        oo_TPs.add(ci)
                for ii in (qo_FP):
                    _, _, oo_FP, _, _, _, _, _, _ \
                    = rate(gf[ii], gf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    for ci in oo_FP:
                        oo_FPs.add(ci)
                oo_avgTPs = 1 if len(qo_TP_sc + qo_TP_cc) == 0 else float(len(oo_TPs)) / len(qo_TP_sc + qo_TP_cc)
                oo_avgFPs = 1 if len(qo_FP) == 0 else float(len(oo_FPs)) / len(qo_FP)
                oo_avgTPR = 0.0 if len(oo_TPRs) == 0 else np.average(np.array(oo_TPRs))

                qid_TP_sc, qid_TP_cc, qid_FP, qid_TPR_sc, qid_TPR_cc, qid_FPR, qid_TP_sc_avgdist, qid_TP_cc_avgdist, qid_FP_avgdist \
                = rate(qidf[i], gidf, q_pids[i], g_pids, q_clothids[i], g_clothids, galleryset, K=K)
                qclo_TP_sc, qclo_TP_cc, qclo_FP, qclo_TPR_sc, qclo_TPR_cc, qclo_FPR, qclo_TP_sc_avgdist, qclo_TP_cc_avgdist, qclo_FP_avgdist \
                = rate(qclothf[i], gclothf, q_pids[i], g_pids, q_clothids[i], g_clothids, galleryset, K=K)
                # print('ori:', query_id, qo_TPR_cc)

                # cloth->id
                cloid_TPRs = []
                cloid_TPs = set()
                cloid_FPs = set()
                for ii in (qclo_TP_sc + qclo_TP_cc):
                    cloid_TP_sc, cloid_TP_cc, cloid_FP, cloid_TPR_sc, cloid_TPR_cc, cloid_FPR, cloid_TP_sc_avgdist, cloid_TP_cc_avgdist, cloid_FP_avgdist \
                    = rate(gidf[ii], gidf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    cloid_TPRs.append(cloid_TPR_cc)
                    for ci in cloid_TP_cc:
                        cloid_TPs.add(ci)
                for ii in (qclo_FP):
                    _, _, fcloid_FP, _, _, _, _, _, _ \
                    = rate(gidf[ii], gidf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    for ci in fcloid_FP:
                        cloid_FPs.add(ci)
                cloid_avgTPs = 1 if len(qclo_TP_sc + qclo_TP_cc) == 0 else float(len(cloid_TPs)) / len(qclo_TP_sc + qclo_TP_cc)
                cloid_avgFPs = 1 if len(qclo_FP) == 0 else float(len(cloid_FPs)) / len(qclo_FP)
                cloid_avgTPR = 0.0 if len(cloid_TPRs) == 0 else np.average(np.array(cloid_TPRs))
                # print('cloid:', query_id, cloid_avgTPR)

                # id->cloth
                idclo_TPRs = []
                idclo_TPs = set()
                idclo_FPs = set()
                for ii in (qid_TP_sc + qid_TP_cc):
                    idclo_TP_sc, idclo_TP_cc, idclo_FP, idclo_TPR_sc, idclo_TPR_cc, idclo_FPR, idclo_TP_sc_avgdist, idclo_TP_cc_avgdist, idclo_FP_avgdist \
                    = rate(gclothf[ii], gclothf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    idclo_TPRs.append(idclo_TPR_cc)
                    for ci in idclo_TP_cc:
                        idclo_TPs.add(ci)
                for ii in (qid_FP):
                    _, _, fidclo_FP, _, _, _, _, _, _ \
                    = rate(gclothf[ii], gclothf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                    for ci in fidclo_FP:
                        idclo_FPs.add(ci)
                idclo_avgTPs = 1 if len(qid_TP_sc + qid_TP_cc) == 0 else float(len(idclo_TPs)) / len(qid_TP_sc + qid_TP_cc)
                idclo_avgFPs = 1 if len(qid_FP) == 0 else float(len(idclo_FPs)) / len(qid_FP)
                idclo_avgTPR = 0.0 if len(idclo_TPRs) == 0 else np.average(np.array(idclo_TPRs))
                # print('idclo:', query_id, idclo_avgTPR)

                # cloth->id->cloth
                idclo_res = dict()
                # for the first two steps of cloth->id->cloth
                for ii in (qid_TP_sc + qid_TP_cc):
                    idclo_topk, _ = knnsearch(gclothf[ii], gclothf, k=K + 1)
                    idclo_topk = idclo_topk[1:]
                    for iii in idclo_topk:
                        idclo_res[iii] = 1 if iii not in idclo_res else idclo_res[iii] + 1
                idclo_all = sorted(idclo_res.items(), key=lambda x: x[1], reverse=True)[:K]
                idclo_all = [x[0] for x in idclo_all]
                idclo_TPnum = 0
                idclo_FPnum = 0
                cloidclo_TPRs = []
                cloidclo_TPs = set()
                cloidclo_FPs = set()
                for iii in idclo_all:
                    if q_pids[i] == g_pids[iii]: # TP
                        cloidclo_TP_sc, cloidclo_TP_cc, cloidclo_FP, cloidclo_TPR_sc, cloidclo_TPR_cc, cloidclo_FPR, _, _, _ \
                        = rate(gclothf[iii], gclothf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                        cloidclo_TPRs.append(cloidclo_TPR_cc)
                        for ci in cloidclo_TP_cc:
                            cloidclo_TPs.add(ci)
                        idclo_TPnum += 1          
                    else: # FP
                        _, _, fcloidclo_FP, _, _, _, _, _, _ \
                        = rate(gclothf[iii], gclothf[g_clothids != q_clothids[i]], q_pids[i], g_pids[g_clothids != q_clothids[i]], q_clothids[i], g_clothids[g_clothids != q_clothids[i]], galleryset, K=K)
                        for ci in fcloidclo_FP:
                            cloidclo_FPs.add(ci)
                        idclo_FPnum += 1
                cloidclo_avgTPs = 1 if idclo_TPnum == 0 else float(len(cloidclo_TPs)) / idclo_TPnum
                cloidclo_avgFPs = 1 if idclo_FPnum == 0 else float(len(cloidclo_FPs)) / idclo_FPnum
                cloidclo_avgTPR = 0.0 if len(cloidclo_TPRs) == 0 else np.average(np.array(cloidclo_TPRs))

                avg_oTPR_list.append(qo_TPR_cc)
                avg_ooTPR_list.append(oo_avgTPR)
                ciTP_num = len(cloid_TPs | idclo_TPs | cloidclo_TPs)
                ciFP_num = len(cloid_FPs | idclo_FPs | cloidclo_FPs)
                avg_ciTPR_list.append(float(len(cloid_TPs)) / len(cloid_TPs | cloid_FPs))
                avg_cloidTPR_list.append(cloid_avgTPR)
                avg_idcloTPR_list.append(idclo_avgTPR)
                avg_cloidcloTPR_list.append(cloidclo_avgTPR)
                avg_idTPR_cc_list.append(qid_TPR_cc)
                avg_cloTPR_sc_list.append(qclo_TPR_sc)

                ## Part 2: verify the negative impact of intermediate FPs
                if len(qclo_FP) != 0 and len(qclo_TP_cc + qclo_TP_sc) != 0:
                    qclo_TP_avgdist = (qclo_TP_cc_avgdist + qclo_TP_sc_avgdist) / (len(qclo_TP_cc + qclo_TP_sc))
                    qclo_FP_avgdist = (qclo_FP_avgdist) / (len(qclo_FP))
                    avgdist_cloTP_list.append(qclo_TP_avgdist)
                    avgdist_cloFP_list.append(qclo_FP_avgdist)
                avgnum_ooTP_list.append(oo_avgTPs)
                avgnum_ooFP_list.append(oo_avgFPs)
                avgnum_cloidTP_list.append(cloid_avgTPs)
                avgnum_cloidFP_list.append(cloid_avgFPs)
                avgnum_idcloTP_list.append(idclo_avgTPs)
                avgnum_idcloFP_list.append(idclo_avgFPs)
                avgnum_cloidcloTP_list.append(cloidclo_avgTPs)
                avgnum_cloidcloFP_list.append(cloidclo_avgFPs)

                positive_num = np.sum((g_pids == q_pids[i]) & (g_clothids != q_clothids[i]))
                avg_oRR_list.append(float(len(o_TP_cc)) / positive_num)
                avg_ooRR_list.append(float(len(oo_TPs)) / positive_num)
                avg_ciRR_list.append(float(len(idclo_TPs | cloid_TPs | cloidclo_TPs)) / positive_num)


        avg_oTPR = np.average(np.array(avg_oTPR_list))
        avg_ooTPR = np.average(np.array(avg_ooTPR_list))
        avg_cloidTPR = np.average(np.array(avg_cloidTPR_list))
        avg_idcloTPR = np.average(np.array(avg_idcloTPR_list))
        avg_cloidcloTPR = np.average(np.array(avg_cloidcloTPR_list))
        avg_ciTPR = np.max(np.array([avg_cloidTPR, avg_idcloTPR, avg_cloidcloTPR]))
        avg_idTPR_cc = np.average(np.array(avg_idTPR_cc_list))
        avg_cloTPR_sc = np.average(np.array(avg_cloTPR_sc_list))
        avg_oRR = np.average(np.array(avg_oRR_list))
        avg_ooRR = np.average(np.array(avg_ooRR_list))
        avg_ciRR = np.average(np.array(avg_ciRR_list))
        avgdist_cloTP = np.average(np.array(avgdist_cloTP_list))
        avgdist_cloFP = np.average(np.array(avgdist_cloFP_list))
        avgnum_ooTP = np.average(np.array(avgnum_ooTP_list))
        avgnum_ooFP = np.average(np.array(avgnum_ooFP_list))
        avgnum_cloidTP = np.average(np.array(avgnum_cloidTP_list))
        avgnum_cloidFP = np.average(np.array(avgnum_cloidFP_list))
        avgnum_idcloTP = np.average(np.array(avgnum_idcloTP_list))
        avgnum_idcloFP = np.average(np.array(avgnum_idcloFP_list))
        avgnum_cloidcloTP = np.average(np.array(avgnum_cloidcloTP_list))
        avgnum_cloidcloFP = np.average(np.array(avgnum_cloidcloFP_list))
        disp_result_1 = [query_id, len(avg_oTPR_list), avg_oRR, avg_ooRR, avg_ciRR, avg_oTPR, avg_ooTPR, avg_ciTPR, avg_idTPR_cc, avg_cloTPR_sc]
        disp_result_2 = [query_id, len(avg_oTPR_list), avgdist_cloTP, avgdist_cloFP, avgnum_ooTP, avgnum_ooFP, avgnum_cloidTP, avgnum_cloidFP, avgnum_idcloTP, avgnum_idcloFP, avgnum_cloidcloTP, avgnum_cloidcloFP]
        csv_data_1.append(disp_result_1)
        csv_data_2.append(disp_result_2)
        print(disp_result_1[:8])
        print(disp_result_2)
    
    header_1 = ['query id', 'num', 'ori RR', 'ori-ori RR', 'ci RR', 'ori TPR', 'ori-ori TPR', 'ci TPR', 'id_cc TPR', 'clo_sc TPR']
    with open('query_analysis_1.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_1)
        writer.writerows(csv_data_1)
    
    header_2 = ['query id', 'num', 'cloTP dist', 'cloFP dist', 'ori-ori average TP', 'ori-ori average FP','clo-id average TP', 'clo-id average FP', 'id-clo average TP', 'id-clo average FP', 'clo-id-clo average TP', 'clo-id-clo average FP']
    with open('query_analysis_2.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_2)
        writer.writerows(csv_data_2)

    print('finished.')