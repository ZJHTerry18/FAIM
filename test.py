import time
import cv2
import os
import gc
import datetime
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from data.dataset_loader import read_image
from tools.eval_metrics import evaluate, evaluate_with_clothes, evaluate_only_cloth_changing
from tools.gradcam import GradCAM
from tools.kr_reranking import kr_re_ranking, kr_re_ranking_ori
from tools.dynamic_kr_reranking import dynamic_kr_re_ranking
from tools.gnn_reranking import gnn_reranking, gnn_reranking_lowcost, gnn_reranking_ori, gnn_reranking_ori_lowcost
from tools.dynamic_gnn_reranking import dynamic_gnn_reranking, dynamic_gnn_reranking_lowcost
from tools.tsne_visualize import tsne
from tools.analysis import analysis


VID_DATASET = ['ccvid']


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(config, model, dataloader):
    features, id_features, cloth_features, id_vars, pids, camids, clothes_ids = [], [], [], [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        if config.MODEL.DECOUPLE:
            if config.MODEL.RELIABILITY:
                batch_features, batch_idf, batch_clothf, batch_idvar = model(imgs)
                batch_features_flip, batch_idf_flip, batch_clothf_flip, batch_idvar_flip = model(flip_imgs)
                # # incorporate features
                # batch_inc = torch.cat([batch_idf, batch_clothf], dim=1)
                # batch_inc_flip = torch.cat([batch_idf_flip, batch_clothf_flip], dim=1)
                # batch_idf = batch_inc
                # batch_clothf = batch_inc
                # batch_idf_flip = batch_inc_flip
                # batch_clothf_flip = batch_inc_flip
            else:
                batch_features, batch_idf, batch_clothf = model(imgs)
                batch_features_flip, batch_idf_flip, batch_clothf_flip = model(flip_imgs)

            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)
            batch_idf += batch_idf_flip
            batch_idf = F.normalize(batch_idf, p=2, dim=1)
            batch_clothf += batch_clothf_flip
            batch_clothf = F.normalize(batch_clothf, p=2, dim=1)
            features.append(batch_features.cpu())
            id_features.append(batch_idf.cpu())
            cloth_features.append(batch_clothf.cpu())
            if config.MODEL.RELIABILITY:
                batch_idvar = 0.5 * (batch_idvar + batch_idvar_flip)
                batch_idvar = batch_idvar.mean(dim=1)
                print(batch_idvar.max().item(), batch_idvar.min().item())
                id_vars.append(batch_idvar)
        else:
            batch_features = model(imgs)
            batch_features_flip = model(flip_imgs)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)
            features.append(batch_features.cpu())

        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)
    if config.MODEL.DECOUPLE:
        id_features = torch.cat(id_features, 0)
        cloth_features = torch.cat(cloth_features, 0)
        if config.MODEL.RELIABILITY:
            id_vars = torch.cat(id_vars, 0)
            features = (features, id_features, cloth_features, id_vars)
        else:
            features = (features, id_features, cloth_features)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        if (batch_idx + 1) % 200==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def visualize(config, model, queryloader, galleryloader, dataset, idvar, distmat, cmc_list, ap_list, index_list, topk):
    output_folder = os.path.join(config.OUTPUT, 'vis_cc_rr_ibfw')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dist_rank = np.argsort(distmat, axis=1)
    # cam_model = GradCAM(config, model, size=[config.DATA.WIDTH, config.DATA.HEIGHT], num_cls=config.MODEL.CA_DIM, layer_name=config.TEST.CAM_LAYER)
    vw = config.DATA.WIDTH // 2
    vh = config.DATA.HEIGHT // 2
    if hasattr(dataset, 'query_diff'):
        queryset = dataset.query_diff
    else:
        queryset = dataset.query
    
    subset = []
    for qi in range(len(queryset)):
        if not isinstance(cmc_list[qi], np.ndarray):
            continue
        cmc_topk = topk - sum(cmc_list[qi][:topk]) + 1
        ap = ap_list[qi]
        if cmc_topk > -1:
            subset.append(qi)
    print('%d selected identifications in %d queries' % (len(subset), len(queryset)))

    qidvar = idvar[:len(queryset)]
    gidvar = idvar[len(queryset):]
    for i, qi in tqdm(enumerate(subset)):
        if i % 1 == 0:
            cmc_topk = topk - sum(cmc_list[qi][:topk]) + 1
            ap = ap_list[qi]
            # do query CAM
            q_path, q_pid, q_camid, q_clothid = queryset[qi]
            q_imgname = os.path.basename(q_path)[:-4]
            q_rawimg = cv2.imread(q_path)
            q_inputimg = (queryloader.dataset[qi])[0]
            # q_camimg = cam_model.forward(q_rawimg)
            q_rawimg = cv2.resize(q_rawimg, (vw, vh))
            # q_camimg = cv2.resize(q_camimg, (vw, vh))
            # cv2.putText(q_rawimg, str(round(1 - (qidvar[qi].item()) ** 0.5,2)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
            # cv2.putText(q_rawimg, str(int(q_pid)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

            vis_rawimg = np.concatenate((q_rawimg, np.zeros((vh, 20, 3), dtype=np.uint8) + 255), axis=1)
            # vis_camimg = np.concatenate((q_camimg, np.zeros((vh, 20, 3), dtype=np.uint8)), axis=1)

            # do gallery CAM for top-k results
            for k in range(topk):
                # gi = dist_rank[qi][k]
                gi = index_list[qi][k]
                g_path, g_pid, g_camid, g_clothid = dataset.gallery[gi]
                g_imgname = os.path.basename(g_path)[:-4]
                g_rawimg = cv2.imread(g_path)
                g_rawimg = cv2.resize(g_rawimg, (vw, vh))
                g_inputimg = (galleryloader.dataset[gi])[0]
                # g_camimg = cam_model.forward(g_rawimg)
                # g_camimg = cv2.resize(g_camimg, (vw, vh))
                # cv2.putText(g_rawimg, str(round(1 - (gidvar[gi].item()) ** 0.5,2)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                # cv2.putText(g_rawimg, str(int(g_pid)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

                if g_pid == q_pid:
                    cv2.rectangle(g_rawimg, (0, 0), (vw - 1, vh - 1), (69, 128, 0), 2)
                else:
                    cv2.rectangle(g_rawimg, (0, 0), (vw - 1, vh - 1), (0, 0, 255), 2)
                vis_rawimg = np.concatenate((vis_rawimg, np.zeros((vh, 5, 3), dtype=np.uint8) + 255, g_rawimg), axis=1)
                # vis_camimg = np.concatenate((vis_camimg, np.zeros((vh, 5, 3), dtype=np.uint8), g_camimg), axis=1)
            
            # save image
            output_raw = '%s.jpg' % (q_imgname)
            # output_cam = '%d_q%d_topk%d_ap%d_cam.jpg' % (qi, q_pid, int(cmc_topk), int(ap * 100))
            cv2.imwrite(os.path.join(output_folder, output_raw), vis_rawimg)
            # cv2.imwrite(os.path.join(output_folder, output_cam), vis_camimg)


def test(config, epoch, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(config, model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(config, model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        #
        if config.MODEL.DECOUPLE:
            if config.MODEL.RELIABILITY:
                qf, qidf, qclothf, qidvar = qf
                gf, gidf, gclothf, gidvar = gf
                qf, qidf, qclothf, qidvar, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, qidf, qclothf, qidvar, q_pids, q_camids, q_clothes_ids], len(dataset.query))
                gf, gidf, gclothf, gidvar, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, gidf, gclothf, gidvar, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))    
            else:
                qf, qidf, qclothf = qf
                gf, gidf, gclothf = gf
                qf, qidf, qclothf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, qidf, qclothf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
                gf, gidf, gclothf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, gidf, gclothf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
        else:
            qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
            gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    ## Original feature
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat_q_all = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity for original feature
    for i in range(m):
        distmat_q_all[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat_q_all = distmat_q_all.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP for Original feature")
    cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_q_all, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET not in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
        logger.info("Computing CMC and mAP only for the same clothes setting")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate_with_clothes(distmat_q_all, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

        logger.info("Computing CMC and mAP only for clothes-changing")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate_with_clothes(distmat_q_all, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

    # if epoch == -1 and config.TEST.VISUALIZE:
    #     logger.info('Generating visualization and CAM for clothes changing results')
    #     if config.MODEL.RELIABILITY:
    #         id_var = torch.cat((qidvar, gidvar), dim=0)
    #         visualize(config, model, queryloader, galleryloader, dataset, id_var, distmat_q_all, cmc_list, ap_list, index_list, 10)
    
    cmc_r = cmc[0]
    if config.MODEL.DECOUPLE:
        ## ID feature
        # Compute distance matrix between query and gallery
        since = time.time()
        m, n = qidf.size(0), gidf.size(0)
        distmat_q_id = torch.zeros((m,n))
        qidf, gidf = qidf.cuda(), gidf.cuda()
        # Cosine similarity for original feature
        for i in range(m):
            distmat_q_id[i] = (- torch.mm(qidf[i:i+1], gidf.t())).cpu()
        distmat_q_id = distmat_q_id.numpy()
        time_elapsed = time.time() - since
        logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        since = time.time()
        logger.info("Computing CMC and mAP for ID feature")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_q_id, q_pids, g_pids, q_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
        time_elapsed = time.time() - since
        logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if config.DATA.DATASET not in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
            logger.info("Computing CMC and mAP only for the same clothes setting")
            cmc, mAP, cmc_list, ap_list, index_list = evaluate_with_clothes(distmat_q_id, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")

            logger.info("Computing CMC and mAP only for clothes-changing")
            cmc, mAP, cmc_list, id_ap_list, index_list = evaluate_with_clothes(distmat_q_id, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")
        del distmat_q_id
        gc.collect()

        ## Clothes feature
        # Compute distance matrix between query and gallery
        since = time.time()
        m, n = qclothf.size(0), gclothf.size(0)
        distmat_q_cloth = torch.zeros((m,n))
        qclothf, gclothf = qclothf.cuda(), gclothf.cuda()
        # Cosine similarity for original feature
        for i in range(m):
            distmat_q_cloth[i] = (- torch.mm(qclothf[i:i+1], gclothf.t())).cpu()
        distmat_q_cloth = distmat_q_cloth.numpy()
        time_elapsed = time.time() - since
        logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        since = time.time()
        logger.info("Computing CMC and mAP for Clothes feature")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_q_cloth, q_pids, g_pids, q_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
        time_elapsed = time.time() - since
        logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

        if config.DATA.DATASET not in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
            logger.info("Computing CMC and mAP only for the same clothes setting")
            cmc, mAP, cmc_list, ap_list, index_list = evaluate_with_clothes(distmat_q_cloth, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")

            logger.info("Computing CMC and mAP only for clothes-changing")
            cmc, mAP, cmc_list, ap_list, index_list = evaluate_with_clothes(distmat_q_cloth, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")
        # del distmat_q_cloth
        # gc.collect()

        # if config.EVAL_MODE:
        #     logger.info('Generating tsne visualization')
        #     tsne(gidf.cpu().numpy(), gclothf.cpu().numpy(), g_pids, g_clothes_ids, gidvar.cpu().numpy(), dataset)
        #     # analysis(qf.cpu().numpy(), gf.cpu().numpy(), qidf.cpu().numpy(), gidf.cpu().numpy(), qclothf.cpu().numpy(), gclothf.cpu().numpy(),
        #     #      q_pids, g_pids, q_clothes_ids, g_clothes_ids, dataset, query_id=config.TEST.TSNE)
    
    if config.MODEL.RELIABILITY:
        idrel = 1.0 - torch.cat((qidvar, gidvar), dim=0).cpu().numpy()
    # Decoupled feature re-ranking
    if config.EVAL_MODE and config.TEST.RERANKING > 0:
        # result_list = []
        # result_list.append(config.DATA.DATASET + ',' + config.TAG + ',' + str(config.TEST.RERANKING) + '\n')
        lambda_j = 1.0
        lambda_ci = 0.3
        lambda_ic = 0.6
        lambda_cic = 0.1
        # res = [lambda_j, lambda_ci, lambda_ic, lambda_cic]
        if config.TEST.RERANKING == 1:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    ## For ablation
                    # distmat_g_id = torch.zeros((n, n))
                    # for i in range(n):
                    #     distmat_g_id[i] = (- torch.mm(gidf[i:i+1], gidf.t())).cpu()
                    # _, _, _, id_ap, _ = evaluate_with_clothes(distmat_g_id, g_pids, g_pids, g_camids, g_camids, g_clothes_ids, g_clothes_ids, mode='CC')
                    # id_ap = id_ap_list + id_ap
                    # ##
                    rr_dist = dynamic_kr_re_ranking(dataset, qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids, idrel, k1=30, k2=6)
                    # rr_dist = kr_re_ranking(dataset, qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids, k1=100, k2=6, lambda_j=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)                       
                    # rr_dist = kr_re_ranking_ori(qf, gf, k1=30, k2=6)
                else:
                    rr_dist = kr_re_ranking(dataset, qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids, k1=30, k2=6, lambda_j=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
                # rr_dist = kr_re_ranking_ori(qclothf, gclothf, k1=30, k2=6)
            else:
                rr_dist = kr_re_ranking_ori(qf, gf, k1=30, k2=6)
        elif config.TEST.RERANKING == 2:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    # rr_dist = dynamic_gnn_reranking_lowcost(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids, idrel, 
                    #                                 k1=100, k2=6)
                    # rr_dist = gnn_reranking_lowcost(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids,
                    #                             k1=100, k2=6, lambda_gi=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
                    rr_dist = gnn_reranking_ori_lowcost(qf, gf, k1=30, k2=6)
                else:
                    rr_dist = gnn_reranking(qf, gf, qidf, gidf, qclothf, gclothf, q_pids, g_pids, q_clothes_ids, g_clothes_ids,
                                                k1=30, k2=6, lambda_gi=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
            else:
                rr_dist = gnn_reranking_ori_lowcost(qf, gf, k1=30, k2=6)
        logger.info("Computing CMC and mAP after reranking for feature:")
        rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate(rr_dist, q_pids, g_pids, q_camids, g_camids)
        logger.info("Results ----------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
        logger.info("------------------------------------------------")

        # if epoch == -1 and config.TEST.VISUALIZE:
        #     logger.info('Generating visualization and CAM for clothes changing results')
        #     visualize(config, model, queryloader, galleryloader, dataset, rr_dist, rr_cmc_list, rr_ap_list, rr_index_list, 10)

        if config.DATA.DATASET not in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
            logger.info("Computing CMC and mAP only for the same clothes setting after reranking:")
            # rr_dist = re_ranking(qf, gf, qidf, gidf, qclothf, gclothf, k1=20, k2=6, lambda_value=0.3)
            rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate_with_clothes(rr_dist, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
            logger.info("Results ----------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
            logger.info("------------------------------------------------")

            logger.info("Computing CMC and mAP only for clothes-changing after reranking:")
            # rr_dist = re_ranking(qf, gf, qidf, gidf, qclothf, gclothf, k1=20, k2=6, lambda_value=0.3)
            rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate_with_clothes(rr_dist, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
            logger.info("Results ----------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
            logger.info("------------------------------------------------")
            # res.append(rr_cmc[0])
            # res.append(rr_mAP)

        if epoch == -1 and config.TEST.VISUALIZE:
            logger.info('Generating visualization for clothes changing results')
            if config.MODEL.RELIABILITY:
                id_var = torch.cat((qidvar, gidvar), dim=0)
                visualize(config, model, queryloader, galleryloader, dataset, id_var, rr_dist, rr_cmc_list, rr_ap_list, rr_index_list, 10)


    # # Decoupled feature re-ranking
    # if config.EVAL_MODE and config.TEST.RERANKING > 0:
    #     # k-reciprocal re-ranking
    #     result_list = []
    #     result_list.append(config.DATA.DATASET + ',' + config.TAG + '\n')
    #     for i in np.arange(0.1, 0.4, 0.1):
    #         for i1 in np.arange(0.2, 0.7, 0.1):
    #             for i2 in np.arange(0.1, 1.1 - i1, 0.1):
    #                 for i3 in np.arange(0.0, min(1.1 - i1 - i2, 0.5), 0.1):
    #                     res = [i, i1, i2, i3]
    #                     rr_dist = kr_re_ranking(qf, gf, qidf, gidf, qclothf, gclothf, k1=20, k2=6, lambda_j=i, lambda_ci=i1, lambda_ic=i2, lambda_cic=i3)
    #                     logger.info("lambda values:{:.1f}, {:.1f}, {:.1f}, {:.1f}".format(i, i1, i2, i3))
    #                     logger.info("Computing CMC and mAP after reranking for original feature:")
    #                     rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate(rr_dist, q_pids, g_pids, q_camids, g_camids)
    #                     logger.info("Results ----------------------------------------")
    #                     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
    #                     logger.info("------------------------------------------------")
    #                     res.append(rr_cmc[0])
    #                     res.append(rr_mAP)

    #                     if config.DATA.DATASET not in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
    #                         logger.info("Computing CMC and mAP only for the same clothes setting after reranking:")
    #                         # rr_dist = kr_re_ranking(qf, gf, qidf, gidf, qclothf, gclothf, k1=20, k2=6, lambda_value=0.3)
    #                         rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate_with_clothes(rr_dist, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    #                         logger.info("Results ----------------------------------------")
    #                         logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
    #                         logger.info("------------------------------------------------")
    #                         res.append(rr_cmc[0])
    #                         res.append(rr_mAP)

    #                         logger.info("Computing CMC and mAP only for clothes-changing after reranking:")
    #                         # rr_dist = kr_re_ranking(qf, gf, qidf, gidf, qclothf, gclothf, k1=20, k2=6, lambda_value=0.3)
    #                         rr_cmc, rr_mAP, rr_cmc_list, rr_ap_list, rr_index_list = evaluate_with_clothes(rr_dist, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    #                         logger.info("Results ----------------------------------------")
    #                         logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_mAP))
    #                         logger.info("------------------------------------------------")
    #                         res.append(rr_cmc[0])
    #                         res.append(rr_mAP)
    #                         res = '{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1%}, {:.1%}, {:.1%}, {:.1%}, {:.1%}, {:.1%}\n'.format(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9])
    #                         result_list.append(res)
    #                     else:
    #                         res = '{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1%}, {:.1%}\n'.format(res[0], res[1], res[2], res[3], res[4], res[5])
    #                         result_list.append(res)
        
    #     with open('re-rank-param-result-seed%d.txt' % (config.SEED), 'w') as f:
    #         f.writelines(result_list)

    #     if epoch == -1 and config.TEST.VISUALIZE:
    #         logger.info('Generating visualization and CAM for clothes changing results')
    #         visualize(config, model, queryloader, galleryloader, dataset, rr_dist, rr_cmc_list, rr_ap_list, rr_index_list, 10)

    return cmc_r


def test_prcc(config, epoch, model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(config, model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(config, model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(config, model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    if config.MODEL.DECOUPLE:
        if config.MODEL.RELIABILITY:
            qsf, qsf_id, qsf_cloth, qsf_idvar = qsf
            qdf, qdf_id, qdf_cloth, qdf_idvar = qdf
            gf, gf_id, gf_cloth, gf_idvar = gf
            qsf, qsf_id, qsf_cloth, qsf_idvar, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qsf_id, qsf_cloth, qsf_idvar, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
            qdf, qdf_id, qdf_cloth, qdf_idvar, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qdf_id, qdf_cloth, qdf_idvar, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
            gf, gf_id, gf_cloth, gf_idvar, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, gf_id, gf_cloth, gf_idvar, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
        else:
            qsf, qsf_id, qsf_cloth = qsf
            qdf, qdf_id, qdf_cloth = qdf
            gf, gf_id, gf_cloth = gf
            qsf, qsf_id, qsf_cloth, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qsf_id, qsf_cloth, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
            qdf, qdf_id, qdf_cloth, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qdf_id, qdf_cloth, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
            gf, gf_id, gf_cloth, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, gf_id, gf_cloth, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    else:
        qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
        qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    ## Original Feature
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for Original feature")
    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    
    if epoch == -1 and config.TEST.VISUALIZE:
        logger.info('Generating visualization and CAM for clothes changing results')
        visualize(config, model, queryloader_diff, galleryloader, dataset, distmat_diff, cmc_list, ap_list, index_list, 10)
    
    cmc_r = cmc[0]

    if config.MODEL.DECOUPLE:
        ## ID Feature
        # Compute distance matrix between query and gallery
        m, n, k = qsf_id.size(0), qdf_id.size(0), gf_id.size(0)
        distmat_same = torch.zeros((m, k))
        distmat_diff = torch.zeros((n, k))
        qsf_id, qdf_id, gf_id = qsf_id.cuda(), qdf_id.cuda(), gf_id.cuda()
        # Cosine similarity
        for i in range(m):
            distmat_same[i] = (- torch.mm(qsf_id[i:i+1], gf_id.t())).cpu()
        for i in range(n):
            distmat_diff[i] = (- torch.mm(qdf_id[i:i+1], gf_id.t())).cpu()
        distmat_same = distmat_same.numpy()
        distmat_diff = distmat_diff.numpy()
        qs_pids, qs_camids, qs_clothes_ids = qs_pids, qs_camids, qs_clothes_ids
        qd_pids, qd_camids, qd_clothes_ids = qd_pids, qd_camids, qd_clothes_ids
        g_pids, g_camids, g_clothes_ids = g_pids, g_camids, g_clothes_ids

        logger.info("Computing CMC and mAP for ID feature")
        logger.info("Computing CMC and mAP for the same clothes setting")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

        logger.info("Computing CMC and mAP only for clothes changing")
        cmc, mAP, cmc_list, id_ap_list, index_list = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

        ## Clothes Feature
        # Compute distance matrix between query and gallery
        m, n, k = qsf_cloth.size(0), qdf_cloth.size(0), gf_cloth.size(0)
        distmat_same = torch.zeros((m, k))
        distmat_diff = torch.zeros((n, k))
        qsf_cloth, qdf_cloth, gf_cloth = qsf_cloth.cuda(), qdf_cloth.cuda(), gf_cloth.cuda()
        # Cosine similarity
        for i in range(m):
            distmat_same[i] = (- torch.mm(qsf_cloth[i:i+1], gf_cloth.t())).cpu()
        for i in range(n):
            distmat_diff[i] = (- torch.mm(qdf_cloth[i:i+1], gf_cloth.t())).cpu()
        distmat_same = distmat_same.numpy()
        distmat_diff = distmat_diff.numpy()
        qs_pids, qs_camids, qs_clothes_ids = qs_pids, qs_camids, qs_clothes_ids
        qd_pids, qd_camids, qd_clothes_ids = qd_pids, qd_camids, qd_clothes_ids
        g_pids, g_camids, g_clothes_ids = g_pids, g_camids, g_clothes_ids

        logger.info("Computing CMC and mAP for Clothes feature")
        logger.info("Computing CMC and mAP for the same clothes setting")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

        logger.info("Computing CMC and mAP only for clothes changing")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
    
    if config.MODEL.RELIABILITY:
        s_idrel = 1.0 - torch.cat((qsf_idvar, gf_idvar), dim=0).cpu().numpy()
        d_idrel = 1.0 - torch.cat((qdf_idvar, gf_idvar), dim=0).cpu().numpy()
    # Decoupled feature re-ranking
    if config.EVAL_MODE and config.TEST.RERANKING > 0:
        lambda_j = 1.0
        lambda_ci = 0.3
        lambda_ic = 0.6
        lambda_cic = 0.1
        if config.TEST.RERANKING == 1:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    rr_dist_same = dynamic_kr_re_ranking(dataset, qsf, gf, qsf_id, gf_id, qsf_cloth, gf_cloth, qs_pids, g_pids, qs_clothes_ids, g_clothes_ids, id_ap=None, id_rel=s_idrel, k1=100, k2=6)
                else:
                    rr_dist_same = kr_re_ranking(dataset, qsf, gf, qsf_id, gf_id, qsf_cloth, gf_cloth, qs_pids, g_pids, qs_clothes_ids, g_clothes_ids, id_ap=None, k1=100, k2=6, lambda_j=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
            else:
                rr_dist_same = kr_re_ranking_ori(qsf, gf, k1=100, k2=6)       
        elif config.TEST.RERANKING == 2:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    rr_dist_same = dynamic_gnn_reranking(qsf, gf, qsf_id, gf_id, qsf_cloth, gf_cloth, qs_pids, g_pids, qs_clothes_ids, g_clothes_ids, s_idrel, k1=100, k2=30)
                    # rr_dist_same = gnn_reranking(qsf, gf, qsf_id, gf_id, qsf_cloth, gf_cloth, k1=100, k2=6, lambda_gi=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
            else:
                rr_dist_same = gnn_reranking_ori(qsf, gf, k1=100, k2=6)
        logger.info("Computing CMC and mAP for Clothes feature re-ranking")
        logger.info("Computing CMC and mAP for the same clothes setting")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(rr_dist_same, qs_pids, g_pids, qs_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

        if config.TEST.RERANKING == 1:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    ## For ablation
                    distmat_g_id = torch.zeros((k, k))
                    for i in range(k):
                        distmat_g_id[i] = (- torch.mm(gf_id[i:i+1], gf_id.t())).cpu()
                    _, _, _, id_ap, _ = evaluate(distmat_g_id, g_pids, g_pids, g_camids, g_camids)
                    id_ap = id_ap_list + id_ap
                    rr_dist_diff = dynamic_kr_re_ranking(dataset, qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, id_ap=id_ap, id_rel=d_idrel, k1=100, k2=30)

                    ##
                    # rr_dist_diff = dynamic_kr_re_ranking(dataset, qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, id_ap=None, id_rel=d_idrel, k1=100, k2=30)
                    # rr_dist_diff = kr_re_ranking_ori(qdf, gf, k1=100, k2=6)
                else:
                    rr_dist_diff = kr_re_ranking(dataset, qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, id_ap=None, k1=100, k2=6, lambda_j=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
            else:
                rr_dist_diff = kr_re_ranking_ori(qdf, gf, k1=100, k2=6)
        elif config.TEST.RERANKING == 2:
            if config.MODEL.DECOUPLE:
                if config.MODEL.RELIABILITY:
                    # rr_dist_diff = dynamic_gnn_reranking(qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, d_idrel, k1=100, k2=30)
                    # rr_dist_diff = gnn_reranking(qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, k1=100, k2=6, lambda_gi=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
                    rr_dist_diff = gnn_reranking_ori(qdf, gf, k1=100, k2=6)
                else:
                    rr_dist_diff = gnn_reranking(qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, k1=100, k2=6, lambda_gi=lambda_j, lambda_ci=lambda_ci, lambda_ic=lambda_ic, lambda_cic=lambda_cic)
            else:
                rr_dist_diff = gnn_reranking_ori(qdf, gf, k1=100, k2=6)
        logger.info("{:.1f}, {:.1f}, {:.1f}".format(lambda_ci, lambda_ic, lambda_cic))        
        logger.info("Computing CMC and mAP only for clothes changing")
        cmc, mAP, cmc_list, ap_list, index_list = evaluate(rr_dist_diff, qd_pids, g_pids, qd_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
    
    # # # Decoupled feature re-ranking
    # if config.EVAL_MODE and config.TEST.RERANKING > 0:
    #     # k-reciprocal re-ranking
    #     result_list = []
    #     for i in np.arange(0.0, 1.1, 0.1):
    #         for i1 in np.arange(0.0, 1.1, 0.1):
    #             for i2 in np.arange(0.0, 1.1, 0.1):
    #                 res = [i, i1, i2]
    #                 rr_dist_same = kr_re_ranking(dataset, qsf, gf, qsf_id, gf_id, qsf_cloth, gf_cloth, qs_pids, g_pids, qs_clothes_ids, g_clothes_ids, id_ap=None, k1=100, k2=6, lambda_j=lambda_j, lambda_ci=i, lambda_ic=i1, lambda_cic=i2)
    #                 logger.info("Computing CMC and mAP for Clothes feature re-ranking")
    #                 logger.info("Computing CMC and mAP for the same clothes setting")
    #                 rr_cmc, rr_mAP, cmc_list, ap_list, index_list = evaluate(rr_dist_same, qs_pids, g_pids, qs_camids, g_camids)
    #                 logger.info("Results ---------------------------------------------------")
    #                 logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_cmc[19], rr_mAP))
    #                 logger.info("-----------------------------------------------------------")
    #                 res.append(rr_cmc[0])
    #                 res.append(rr_mAP)

    #                 logger.info("lambda values:{:.1f}, {:.1f}, {:.1f}".format(i, i1, i2))
    #                 rr_dist_diff = kr_re_ranking(dataset, qdf, gf, qdf_id, gf_id, qdf_cloth, gf_cloth, qd_pids, g_pids, qd_clothes_ids, g_clothes_ids, id_ap=None, k1=100, k2=6, lambda_j=lambda_j, lambda_ci=i, lambda_ic=i1, lambda_cic=i2)
    #                 logger.info("Computing CMC and mAP only for clothes changing")
    #                 rr_cmc, rr_mAP, cmc_list, ap_list, index_list = evaluate(rr_dist_diff, qd_pids, g_pids, qd_camids, g_camids)
    #                 logger.info("Results ---------------------------------------------------")
    #                 logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(rr_cmc[0], rr_cmc[4], rr_cmc[9], rr_cmc[19], rr_mAP))
    #                 logger.info("-----------------------------------------------------------")
    #                 res.append(rr_cmc[0])
    #                 res.append(rr_mAP)
    #                 res = '{:.1f}, {:.1f}, {:.1f}, {:.1%}, {:.1%}, {:.1%}, {:.1%}\n'.format(res[0], res[1], res[2], res[3], res[4], res[5], res[6])
    #                 result_list.append(res)
        
    #     with open('re-rank-prcc-param-result-seed%d.txt' % (config.SEED), 'w') as f:
    #         f.writelines(result_list)


    return cmc_r