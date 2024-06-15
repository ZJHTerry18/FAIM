import os
import re
import numpy as np
import csv

root_path = '/public/zhaojiahe/results/ccreid/logs/ltcc/res50_patchsh_caufdn+clotri+maxavg_gap/s4'

baseline_folder = 'vis_allf_cc'
rr_folder = 'vis_allf_cc_rr'
rric_folder = 'vis_allf_cc_rric+ci'

baseline_res = sorted(os.listdir(os.path.join(root_path, baseline_folder)))
rr_res = sorted(os.listdir(os.path.join(root_path, rr_folder)))
rric_res = sorted(os.listdir(os.path.join(root_path, rric_folder)))

pattern = re.compile(r'(\d+)_q(\d+)_topk(\d+)_ap(\d+)_raw.jpg')

query_result_dict = {}

for bl, rr, rric in zip(baseline_res, rr_res, rric_res):
    if 'raw' in bl:
        bl_i, bl_qid, bl_topk, bl_ap = map(int, pattern.search(bl).groups())
        rr_i, rr_qid, rr_topk, rr_ap = map(int, pattern.search(rr).groups())
        rric_i, rric_qid, rric_topk, rric_ap = map(int, pattern.search(rric).groups())

        assert bl_i == rr_i and bl_i == rric_i
        assert bl_qid == rr_qid and bl_qid == rric_qid

        if bl_qid not in query_result_dict.keys():
            query_result_dict[bl_qid] = []
            query_result_dict[bl_qid].append([float(bl_topk == 1) * 100, float(bl_ap), float(rr_topk == 1) * 100, float(rr_ap), float(rric_topk == 1) * 100, float(rric_ap)])
        else:
            query_result_dict[bl_qid].append([float(bl_topk == 1) * 100, float(bl_ap), float(rr_topk == 1) * 100, float(rr_ap), float(rric_topk == 1) * 100, float(rric_ap)])

csv_data = []

for qid in query_result_dict.keys():
    qres = np.array(query_result_dict[qid])
    qnum = qres.shape[0]
    qavg = np.mean(qres, axis=0).tolist()
    qavg.insert(0, qnum)
    qavg.insert(0, qid)
    csv_data.append(qavg)

header = ['query id', 'sample num', 'baseline top1', 'baseline mAP', 'rr top1', 'rr mAP', 'rric top1', 'rric mAP']
with open(os.path.join(root_path, 'query_res_cc.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)
