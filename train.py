import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tools.utils import AverageMeter
import copy
import os
from tools.tsne_visualize import tsne


def train_cal(config, epoch, model, classifier, clothes_classifier, criterions, optimizer, optimizer_cc, scheduler, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_mixup_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, masks) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask, masks = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda(), masks.cuda()

        if config.AUG.PIXEL_SAMPLING:
            parsing = masks.unsqueeze(1)      # [64, 1, 384, 192]
            parsing = parsing.expand_as(imgs)
            imgs_copy = copy.deepcopy(imgs)

            # upper clothes sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index]                                  # [64, 3, 384, 192]
            parsing_shuffle = parsing[index]                               # [64, 3, 384, 192]
            imgs_copy[parsing == 2] = imgs_shuffle[parsing_shuffle == 2]

            # pant sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index] 
            parsing_shuffle = parsing[index]
            imgs_copy[parsing == 3] = imgs_shuffle[parsing_shuffle == 3]

            imgs = imgs_copy

        idx_shuff = list(range(pids.shape[0]))
        random.shuffle(idx_shuff)
        if config.MODEL.USE_OLD_FEATURE:
            pids = torch.cat((pids, pids), dim=0)
            # print(pids.shape, pids)
            clothes_ids = torch.cat((clothes_ids, clothes_ids[idx_shuff]), dim=0)
            # print(clothes_ids.shape, clothes_ids)
            pos_mask = torch.cat((pos_mask, pos_mask), dim=0) # positive mask marks samples that has same person id with each sample, no shuffle during cloth feature mixup

        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs, masks, train=True, idx_shuff=idx_shuff)
        outputs = classifier(features)
        pred_clothes = clothes_classifier(features.detach())
        # pred_clothes = clothes_classifier(features)  # use clothes classifier to optimize the whole network in my experiment
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        if config.LOSS.CLOTHES_CLA_LOSS == 'crossentropycloth':
            clothes_loss = criterions['clo'](pred_clothes, pids, clothes_ids)
        else:
            clothes_loss = criterions['clo'](pred_clothes, clothes_ids)
        
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features.detach())
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterions['cla'](outputs, pids)
        pair_loss = criterions['tri'](features, pids)
        adv_loss = criterions['cal'](new_pred_clothes, clothes_ids, pos_mask)
        mixup_loss = criterions['mixup'](outputs, pids)
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + config.LOSS.MIXUP_LOSS_WEIGHT * mixup_loss + \
                config.LOSS.CLO_LOSS_WEIGHT * clothes_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_mixup_loss.update(mixup_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'LR:{lr:.2e} '
                  'ClaLoss:{cla_loss.avg:.3f} '
                  'PairLoss:{pair_loss.avg:.3f} '
                  'CloLoss:{clo_loss.avg:.3f} '
                  'AdvLoss:{adv_loss.avg:.3f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, lr=scheduler.get_lr()[0],
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))

def train_cal_ca(config, epoch, model, classifier, id_classifier, clothes_classifier, criterions, optimizer, optimizer_cc, scheduler, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_id_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_mixup_loss = AverageMeter()
    batch_ortho_loss = AverageMeter()
    corrects = AverageMeter()
    id_corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    id_classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, masks) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        batch_size = pids.shape[0]
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask, masks = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda(), masks.cuda()

        if config.AUG.PIXEL_SAMPLING:
            parsing = masks.unsqueeze(1)      # [64, 1, 384, 192]
            parsing = parsing.expand_as(imgs)
            imgs_copy = copy.deepcopy(imgs)

            # upper clothes sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index]                                  # [64, 3, 384, 192]
            parsing_shuffle = parsing[index]                               # [64, 3, 384, 192]
            imgs_copy[parsing == 2] = imgs_shuffle[parsing_shuffle == 2]

            # pant sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index] 
            parsing_shuffle = parsing[index]
            imgs_copy[parsing == 3] = imgs_shuffle[parsing_shuffle == 3]

            imgs = torch.cat((imgs, imgs_copy), dim=0)
            pids = torch.cat((pids, pids), dim=0)
            clothes_ids = torch.cat((clothes_ids, -1 * torch.ones_like(clothes_ids, dtype=int)), dim=0)

        idx_shuff = list(range(pids.shape[0]))
        random.shuffle(idx_shuff)
        if config.MODEL.USE_OLD_FEATURE:
            pids = torch.cat((pids, pids), dim=0)
            # print(pids.shape, pids)
            clothes_ids = torch.cat((clothes_ids, -1 * torch.ones_like(clothes_ids, dtype=int)), dim=0)
            # print(clothes_ids.shape, clothes_ids)
            pos_mask = torch.cat((pos_mask, pos_mask), dim=0) # positive mask marks samples that has same person id with each sample, no shuffle during cloth feature mixup

        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        if config.MODEL.SIM:
            features, id_features, cloth_features, idf_pool, clothf_pool = model(imgs, masks, train=True, idx_shuff=idx_shuff)
        else:
            features, id_features, cloth_features = model(imgs, masks, train=True, idx_shuff=idx_shuff)
        outputs = classifier(features)
        pred_ids = id_classifier(id_features)
        # cloth_features = torch.cat((id_features, cloth_features), dim=1)
        if config.MODEL.USE_OLD_FEATURE or not config.MODEL.PATCH_SHUFFLE:
            pred_clothes = clothes_classifier(cloth_features)  # use clothes classifier to optimize the whole network in my experiment
        else:
            pred_clothes = clothes_classifier(cloth_features.detach())
        _, preds = torch.max(outputs.data, 1)
        _, id_preds = torch.max(pred_ids.data, 1)
        _, clothes_preds = torch.max(pred_clothes.data, 1)

        output_bs = outputs.shape[0]
        # Compute loss
        cla_loss = criterions['cla'](outputs, pids)
        id_cla_loss = criterions['cla'](pred_ids, pids)
        # id_adv_loss = criterion_adv(pred_ids[:batch_size], clothes_ids[:batch_size], pos_mask[:batch_size])
        # Update the clothes discriminator
        # only calculate clothes loss for old features
        clothes_cla_loss = criterions['clo'](pred_clothes[:batch_size], clothes_ids[:batch_size])
        pair_loss = criterions['tri'](features, pids)
        id_pair_loss = criterions['tri'](id_features, pids)
        clothes_pair_loss = criterions['clotri'](cloth_features, pids, clothes_ids)
        # adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        mixup_loss = criterions['mixup'](outputs, pids)
        feat_mats = torch.cat((id_features.unsqueeze(1), cloth_features.unsqueeze(1)), dim=1)
        ortho_loss = criterions['ortho'](feat_mats)

        id_loss = id_cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * id_pair_loss
        clothes_loss = clothes_cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * clothes_pair_loss
        if config.MODEL.SIM:
            sim_loss = criterions['sim'](id_features, idf_pool.detach()) + criterions['sim'](cloth_features, clothf_pool.detach())
            
            loss = config.LOSS.CLA_LOSS_WEIGHT * (cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss) + config.LOSS.MIXUP_LOSS_WEIGHT * mixup_loss + \
                config.LOSS.ID_LOSS_WEIGHT * id_loss  + config.LOSS.CLO_LOSS_WEIGHT * clothes_loss + config.LOSS.ORTHO_LOSS_WEIGHT * ortho_loss \
                + config.LOSS.SIM_LOSS_WEIGHT * sim_loss
        else:
            loss = config.LOSS.CLA_LOSS_WEIGHT * (cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss) + config.LOSS.MIXUP_LOSS_WEIGHT * mixup_loss + \
                config.LOSS.ID_LOSS_WEIGHT * id_loss  + config.LOSS.CLO_LOSS_WEIGHT * clothes_loss + config.LOSS.ORTHO_LOSS_WEIGHT * ortho_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        id_corrects.update(torch.sum(id_preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds[:batch_size] == clothes_ids[:batch_size].data).float()/batch_size, batch_size)
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_id_loss.update(id_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_mixup_loss.update(mixup_loss.item(), pids.size(0))
        batch_ortho_loss.update(ortho_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.3f} '
                  'PairLoss:{pair_loss.avg:.3f} '
                  'LR:{lr:.2e} '
                  'IDLoss:{id_loss.avg:.3f} '
                  'CloLoss:{clo_loss.avg:.3f} '
                  'Acc:{acc.avg:.2%} '
                  'IDAcc:{id_acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, lr=scheduler.get_lr()[0],
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
                   id_loss=batch_id_loss, clo_loss=batch_clo_loss, acc=corrects, id_acc=id_corrects, clo_acc=clothes_corrects))

def train_cal_ca_var(config, epoch, model, classifier, id_classifier, clothes_classifier, criterions, optimizer, optimizer_cc, scheduler, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_id_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_mixup_loss = AverageMeter()
    batch_ortho_loss = AverageMeter()
    corrects = AverageMeter()
    id_corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    id_classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, masks) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        batch_size = pids.shape[0]
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask, masks = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda(), masks.cuda()

        if config.AUG.PIXEL_SAMPLING:
            parsing = masks.unsqueeze(1)      # [64, 1, 384, 192]
            parsing = parsing.expand_as(imgs)
            imgs_copy = copy.deepcopy(imgs)

            # upper clothes sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index]                                  # [64, 3, 384, 192]
            parsing_shuffle = parsing[index]                               # [64, 3, 384, 192]
            imgs_copy[parsing == 2] = imgs_shuffle[parsing_shuffle == 2]

            # pant sampling
            index = np.random.permutation(imgs.size(0))
            imgs_shuffle = imgs[index] 
            parsing_shuffle = parsing[index]
            imgs_copy[parsing == 3] = imgs_shuffle[parsing_shuffle == 3]

            imgs = torch.cat((imgs, imgs_copy), dim=0)
            pids = torch.cat((pids, pids), dim=0)
            clothes_ids = torch.cat((clothes_ids, -1 * torch.ones_like(clothes_ids, dtype=int)), dim=0)

        idx_shuff = list(range(pids.shape[0]))
        random.shuffle(idx_shuff)
        if config.MODEL.USE_OLD_FEATURE:
            pids = torch.cat((pids, pids), dim=0)
            # print(pids.shape, pids)
            clothes_ids = torch.cat((clothes_ids, -1 * torch.ones_like(clothes_ids, dtype=int)), dim=0)
            # print(clothes_ids.shape, clothes_ids)
            pos_mask = torch.cat((pos_mask, pos_mask), dim=0) # positive mask marks samples that has same person id with each sample, no shuffle during cloth feature mixup

        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        if config.MODEL.AUG_TIMES > 0:
            features, id_features, id_aug_features, cloth_features, id_var = \
                model(imgs, pids, clothes_ids, masks, train=True, idx_shuff=idx_shuff, sample_k=config.MODEL.AUG_TIMES)
            pids_aug = pids.repeat_interleave(config.MODEL.AUG_TIMES, dim=0)
            clothes_ids_aug = clothes_ids.repeat_interleave(config.MODEL.AUG_TIMES, dim=0)
        else:
            features, id_features, cloth_features, id_var = model(imgs, masks, train=True, idx_shuff=idx_shuff)

        outputs = classifier(features)
        pred_ids = id_classifier(id_features)
        if config.MODEL.AUG_TIMES > 0:
            pred_ids_aug = id_classifier(id_aug_features)
            # pred_clothes_aug = clothes_classifier(cloth_aug_features)
        # cloth_features = torch.cat((id_features, cloth_features), dim=1)
        if config.MODEL.USE_OLD_FEATURE or not config.MODEL.PATCH_SHUFFLE:
            pred_clothes = clothes_classifier(cloth_features)  # use clothes classifier to optimize the whole network in my experiment
        else:
            pred_clothes = clothes_classifier(cloth_features.detach())
        _, preds = torch.max(outputs.data, 1)
        _, id_preds = torch.max(pred_ids.data, 1)
        _, clothes_preds = torch.max(pred_clothes.data, 1)

        output_bs = outputs.shape[0]
        # Compute loss
        cla_loss = criterions['cla'](outputs, pids)
        id_cla_loss = criterions['rel'](pred_ids, pids, id_var, pred_ids_aug, pids_aug)
        # id_adv_loss = criterion_adv(pred_ids[:batch_size], clothes_ids[:batch_size], pos_mask[:batch_size])
        # Update the clothes discriminator
        # only calculate clothes loss for old features
        clothes_cla_loss = criterions['clo'](pred_clothes[:batch_size], clothes_ids[:batch_size])
        pair_loss = criterions['tri'](features, pids)
        id_pair_loss = criterions['tri'](id_features, pids)
        clothes_pair_loss = criterions['clotri'](cloth_features, pids, clothes_ids)
        # adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        mixup_loss = criterions['mixup'](outputs, pids)
        feat_mats = torch.cat((id_features.unsqueeze(1), cloth_features.unsqueeze(1)), dim=1)
        ortho_loss = criterions['ortho'](feat_mats)

        id_loss = id_cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * id_pair_loss
        # clothes_loss = clothes_cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * clothes_pair_loss
        clothes_loss = clothes_cla_loss
        loss = config.LOSS.CLA_LOSS_WEIGHT * (cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss) + config.LOSS.MIXUP_LOSS_WEIGHT * mixup_loss + \
            config.LOSS.ID_LOSS_WEIGHT * id_loss  + config.LOSS.CLO_LOSS_WEIGHT * clothes_loss + config.LOSS.ORTHO_LOSS_WEIGHT * ortho_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tsne(F.normalize(id_features, p=2, dim=1).detach().cpu().numpy(), pids.cpu().numpy(), clothes_ids.cpu().numpy(), id_var.mean(dim=1).detach().cpu().numpy())
        
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        id_corrects.update(torch.sum(id_preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds[:batch_size] == clothes_ids[:batch_size].data).float()/batch_size, batch_size)
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_id_loss.update(id_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_mixup_loss.update(mixup_loss.item(), pids.size(0))
        batch_ortho_loss.update(ortho_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.3f} '
                  'PairLoss:{pair_loss.avg:.3f} '
                  'LR:{lr:.2e} '
                  'IDLoss:{id_loss.avg:.3f} '
                  'CloLoss:{clo_loss.avg:.3f} '
                  'Acc:{acc.avg:.2%} '
                  'IDAcc:{id_acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, lr=scheduler.get_lr()[0],
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
                   id_loss=batch_id_loss, clo_loss=batch_clo_loss, acc=corrects, id_acc=id_corrects, clo_acc=clothes_corrects))