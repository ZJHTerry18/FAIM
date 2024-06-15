from torch import nn
from losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from losses.triplet_loss import TripletLoss, ClothTripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss, SimLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss, ClothesBasedAdversarialLossWithMemoryBank, CrossEntropyWithClothes
from losses.mixup_loss import MixupLoss
from losses.orthogonal_loss import OrthogonalLoss
from losses.generalized_celoss import GeneralizedCELoss
from losses.reliability_loss import ReliabilityCELoss


def build_losses(config, num_train_clothes):
    criterions = dict()
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterions['cla'] = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterions['cla'] = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'generalcrossentropy':
        criterions['cla'] = GeneralizedCELoss()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterions['cla'] = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterions['cla'] = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterions['cla'] = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterions['tri'] = TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterions['tri'] = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterions['tri'] = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterions['tri'] = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))

    # Build clothes classification loss
    if config.LOSS.CLOTHES_CLA_LOSS == 'crossentropy':
        criterions['clo'] = nn.CrossEntropyLoss()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'crossentropylabelsmooth':
        criterions['clo'] = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'crossentropycloth':
        criterions['clo'] = CrossEntropyWithClothes()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'cosface':
        criterions['clo'] = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    else:
        raise KeyError("Invalid clothes classification loss: '{}'".format(config.LOSS.CLOTHES_CLA_LOSS))

    # Build clothes pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterions['clotri'] = ClothTripletLoss(margin=config.LOSS.PAIR_M)
    
    # Build feature reliability loss
    if config.MODEL.RELIABILITY:
        criterions['rel'] = ReliabilityCELoss(lamda_aug=config.LOSS.REL_AUG_WEIGHT, lamda_fu=config.LOSS.REL_FU_WEIGHT)

    # Build clothes-based adversarial loss
    if config.LOSS.CAL == 'cal':
        criterions['cal'] = ClothesBasedAdversarialLoss(scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    elif config.LOSS.CAL == 'calwithmemory':
        criterions['cal'] = ClothesBasedAdversarialLossWithMemoryBank(num_clothes=num_train_clothes, feat_dim=config.MODEL.FEATURE_DIM,
                             momentum=config.LOSS.MOMENTUM, scale=config.LOSS.CLA_S, epsilon=config.LOSS.EPSILON)
    else:
        raise KeyError("Invalid clothing classification loss: '{}'".format(config.LOSS.CAL))

    # Build mixup pairwise loss
    criterions['mixup'] = MixupLoss(sim='cos')

    # Build decoupled feature orthogonal loss
    criterions['ortho'] = OrthogonalLoss()

    # Build similarity costraint loss
    criterions['sim'] = SimLoss()

    return criterions