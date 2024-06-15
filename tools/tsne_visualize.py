import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import cv2
import os.path as osp
import torch.nn.functional as F

def knnsearch(qf, gf, k=100):
    assert k <= gf.shape[0]
    dist = np.linalg.norm(gf - qf, ord=2, axis=1, keepdims=False)
    topk = np.argsort(dist)[:k].tolist()
    return topk

def plot(qdata, gdata, knndata, figname):
    pointsize = 3
    fontsize = 2
    plt.figure()
    qx, qy, q_seq, q_id, q_clothid, q_imgpath = qdata
    gx, gy, g_seq, g_id, g_clothid, g_imgpath = gdata
    knnx, knny, knn_seq, knn_id, knn_clothid, knn_imgpath = knndata
    g_seq = list(map(str, g_seq))
    knn_seq = list(map(str, knn_seq))
    knn_posid = [i for i, x in enumerate(knn_id) if x == q_id[0]]
    knn_negid = [i for i, x in enumerate(knn_id) if x != q_id[0]]
    knn_color = []
    for i in range(len(knn_clothid)):
        if knn_id[i] != q_id[0]:
            color = 'grey'
        else:
            color = 'g' if knn_clothid[i] == q_clothid[0] else 'r'
        knn_color.append(color)
    g_color = ['g' if i == q_clothid[0] else 'r' for i in g_clothid]

    # gallery
    for i in range(len(gx)):
        plt.scatter(gx[i], gy[i], s=pointsize, c=g_color[i], marker='o')
        plt.text(gx[i], gy[i], g_seq[i], fontsize=fontsize)

    # knn
    for i in knn_posid:
        plt.scatter(knnx[i], knny[i], s=pointsize, c=knn_color[i], marker='o')
        plt.text(knnx[i], knny[i], knn_seq[i], fontsize=fontsize)
    for i in knn_negid:
        plt.scatter(knnx[i], knny[i], s=pointsize, c=knn_color[i], marker='^')
        plt.text(knnx[i], knny[i], knn_seq[i], fontsize=fontsize)
    
    # query
    for i in range(len(qx)):
        plt.scatter(qx[i], qy[i], s=pointsize, c='b', marker='o')
        # plt.text(qx[i], qy[i], q_seq[i], fontsize=fontsize, color=(1.0, 0.0, 0.0))

    plt.savefig(figname, dpi=1000, bbox_inches='tight')

def tsne_neighbor(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothids, g_clothids, dataset, plot_id):
    '''
        qf, qidf, qclothf: numpy ndarray, N_q * d
        gf, gidf, gclothf: numpy ndarray, N_g * d
        q_pids, q_clothids: N_q * 1
        g_pids, g_clothids: N_g * 1
    '''
    queryset = dataset.query
    galleryset = dataset.gallery
    qnum, gnum = qf.shape[0], gf.shape[0]

    ## do t-SNE
    model_tsne = TSNE(n_components=2, init='pca')
    idf_t = model_tsne.fit_transform(np.concatenate((qidf, gidf), axis=0))
    qidf_t = idf_t[:qnum]
    gidf_t = idf_t[qnum:]
    clothf_t = model_tsne.fit_transform(np.concatenate((qclothf, gclothf), axis=0))
    qclothf_t = clothf_t[:qnum]
    gclothf_t = clothf_t[qnum:]

    ## select query sample for plotting
    qidx = []
    qidy = []
    qclothx = []
    qclothy = []
    qi_list = []
    for i, q_pid in enumerate(q_pids):
        if q_pid == plot_id:
            qi_list.append(i)
    for i, qi in enumerate(qi_list):
        print('%d: %s' % (i, queryset[qi][0]), end='\n')
    # qi = random.choice(qi_list)
    qi = qi_list[2]
    qidx.append(qidf_t[qi][0])
    qidy.append(qidf_t[qi][1])
    qclothx.append(qclothf_t[qi][0])
    qclothy.append(qclothf_t[qi][1])
    q_seq = [qi]
    q_id = [plot_id]
    q_clothid = [q_clothids[qi]]
    q_imgpath = [queryset[qi][0]]

    ## select gallery samples
    gidx = []
    gidy = []
    gclothx = []
    gclothy = []
    g_seq = []
    g_id = []
    g_clothid = []
    g_imgpath = []
    for i, g_pid in enumerate(g_pids):
        if g_pid == plot_id and g_clothids[i] != q_clothid[0]:
            gidx.append(gidf_t[i][0])
            gidy.append(gidf_t[i][1])
            gclothx.append(gclothf_t[i][0])
            gclothy.append(gclothf_t[i][1])
            g_id.append(g_pid)
            g_clothid.append(g_clothids[i])
            g_imgpath.append(galleryset[i][0])
            g_seq.append(i)

    ## select k-nearest neighbor in id space
    id_knn = knnsearch(qidf_t[qi], gidf_t, k=100)
    idknnx = []
    idknny = []
    idknn_seq = []
    idknn_id = []
    idknn_clothid = []
    idknn_imgpath = []
    for i in id_knn:
        idknnx.append(gidf_t[i][0])
        idknny.append(gidf_t[i][1])
        idknn_id.append(g_pids[i])
        idknn_clothid.append(g_clothids[i])
        idknn_imgpath.append(galleryset[i][0])
        idknn_seq.append(i)
    qid_data = (qidx, qidy, q_seq, q_id, q_clothid, q_imgpath)
    gid_data = (gidx, gidy, g_seq, g_id, g_clothid, g_imgpath)
    knnid_data = (idknnx, idknny, idknn_seq, idknn_id, idknn_clothid, idknn_imgpath)
    ## plot in id-feature space
    plot(qid_data, gid_data, knnid_data, figname='tsne-id.jpg')

    ## select k-nearest neighbor in clothes space
    clo_knn = knnsearch(qclothf_t[qi], gclothf_t, k=100)
    cloknnx = []
    cloknny = []
    cloknn_seq = []
    cloknn_id = []
    cloknn_clothid = []
    cloknn_imgpath = []
    for i in clo_knn:
        cloknnx.append(gclothf_t[i][0])
        cloknny.append(gclothf_t[i][1])
        cloknn_id.append(g_pids[i])
        cloknn_clothid.append(g_clothids[i])
        cloknn_imgpath.append(galleryset[i][0])
        cloknn_seq.append(i)
    qcloth_data = (qclothx, qclothy, q_seq, q_id, q_clothid, q_imgpath)
    gcloth_data = (gclothx, gclothy, g_seq, g_id, g_clothid, g_imgpath)
    knnclo_data = (cloknnx, cloknny, cloknn_seq, cloknn_id, cloknn_clothid, cloknn_imgpath)
    ## plot in clothes-feature space
    plot(qcloth_data, gcloth_data, knnclo_data, figname='tsne-cloth.jpg')

    ## show query image
    query_image = cv2.imread(q_imgpath[0])
    cv2.imwrite('./query.jpg', query_image)
    print('finished.')

def tsne(gidf, gclothf, g_pids, g_clothids, gidvar=None, dataset=None):
    g_pids = g_pids.astype(int)
    g_clothids = g_clothids.astype(int)
    # queryset = dataset.query
    galleryset = dataset.gallery

    tsne_model = TSNE(n_components=2)
    # f = tsne_model.fit_transform(gf)
    pid_set = list(set(g_pids))
    # pid_selected = random.sample(pid_set, 4)
    pid_selected = [16, 32, 105, 145]
    print(pid_selected)
    index_selected = np.concatenate([np.where(g_pids == x)[0] for x in pid_selected])
    
    gidf = gidf[index_selected]
    f = tsne_model.fit_transform(gidf)
    pids = g_pids[index_selected]
    g_clothids = g_clothids[index_selected]
    clothids = g_clothids % 5
    galleryset = [galleryset[i] for i in index_selected]
    gidvar = gidvar[index_selected]
    # texts_1 = [osp.basename(x[0])[:-4].split('_')[-1] for x in galleryset]
    # texts_2 = [str(round(1 - (gidvar[i]) ** 0.5, 2)) for i in range(gidvar.shape[0])]

    plt.figure(figsize=(15,8))
    colormap = ['r', 'g', 'b', 'purple', 'grey']
    markermap = ['o', 'v', '1', 'x', '*']
    colors = [colormap[pid_selected.index(pi)] for pi in pids]
    markers = [markermap[ci] for ci in clothids]
    for i in range(0, f.shape[0], 3):
        plt.scatter(f[i, 0], f[i, 1], c=colors[i], marker=markers[i], s=40)
    # for i in range(f.shape[0]):
    #     plt.text(f[i, 0], f[i, 1], str(g_clothids[i]), fontsize=10)
    plt.show()
    plt.savefig('tsne_id.jpg')

    gclothf = gclothf[index_selected]
    f = tsne_model.fit_transform(gclothf)
    pids = g_pids[index_selected]

    plt.figure(figsize=(15,8))
    for i in range(0, f.shape[0], 3):
        plt.scatter(f[i, 0], f[i, 1], c=colors[i], marker=markers[i], s=40)
    # for i in range(f.shape[0]):
    #     plt.text(f[i, 0], f[i, 1], str(g_clothids[i]), fontsize=10)
    plt.show()
    plt.savefig('tsne_cloth.jpg')