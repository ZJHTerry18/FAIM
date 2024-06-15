import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/public/zhaojiahe/datasets'
# Dataset for evaluation
_C.DATA.DATASET = 'ltcc'
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 384
# Width of input image
_C.DATA.WIDTH = 192
# Batch size for training
_C.DATA.TRAIN_BATCH = 32
# Batch size for testing
_C.DATA.TEST_BATCH = 128
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# Random clothes erase prob
_C.AUG.RCE_PROB = 0.0
# Random clothes erase prob
_C.AUG.RCEB_PROB = 0.0
# Random clothes erase prob
_C.AUG.RCEE_PROB = 0.0
# Using pixel sampling for training
_C.AUG.PIXEL_SAMPLING = False
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 4096
# cross-attention dim
_C.MODEL.CA_DIM = 2048
# Model path for resuming
_C.MODEL.RESUME = ''
# Global pooling after the backbone
_C.MODEL.POOLING = CN()
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLING.NAME = 'maxavg'
# Initialized power for GeM pooling
_C.MODEL.POOLING.P = 3
# Mixup layer number in models
_C.MODEL.ERASE_LAYER = 'none'
# Preserve original feature (before shuffling)
_C.MODEL.USE_OLD_FEATURE = True
# Perform patch shuffle
_C.MODEL.PATCH_SHUFFLE = True
# Use cross-attention feature decouple
_C.MODEL.DECOUPLE = True
# Number of decoupling blocks
_C.MODEL.NUM_DECOUPLE_BLOCKS = 2
# Use mask feature to guide feature decouple
_C.MODEL.SIM = False
# Use prompt tuning
_C.MODEL.USE_PROMPT = False
# Use Reliability Prediction
_C.MODEL.RELIABILITY = False
# Times of Semantic Augmentation
_C.MODEL.AUG_TIMES = 10
# Use clothes-change variance
_C.MODEL.USE_CLO_VAR = False
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropylabelsmooth'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'crossentropy'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Scale(weight) for classification loss
_C.LOSS.CLA_LOSS_WEIGHT = 1.0
# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 1.0
# Scale for pairwise loss
_C.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.3
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.
# Scale for mixup loss
_C.LOSS.MIXUP_LOSS_WEIGHT = 0.0
# Scale for id loss
_C.LOSS.ID_LOSS_WEIGHT = 0.5
# Scale for clothes loss
_C.LOSS.CLO_LOSS_WEIGHT = 0.5
# Scale for orthogonal loss
_C.LOSS.ORTHO_LOSS_WEIGHT = 0.0
# Scale for similarity constraint loss
_C.LOSS.SIM_LOSS_WEIGHT = 0.0
# Scale for augmented samples in reliability loss
_C.LOSS.REL_AUG_WEIGHT = 1.0
# Scale for feature uncertainty loss in reliability loss
_C.LOSS.REL_FU_WEIGHT = 0.3
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 200
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 200
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Scheduler
_C.TRAIN.LR_SCHEDULER.NAME = 'none'
# Warmup epochs
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCH = 5
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Using amp for training
_C.TRAIN.AMP = False
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# Do visualize
_C.TEST.VISUALIZE = False
# Do t-SNE analysis for a given ID, -1 for not doing
_C.TEST.TSNE = -1
# Layer to perform CAM
_C.TEST.CAM_LAYER = 'layer4'
# Re-ranking type
_C.TEST.RERANKING = 0 # 1 for k-reciprocal, 2 for GNN, 0 for no reranking
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed, overwritten by command line argument
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = '/public/zhaojiahe/results/ccreid/logs'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'res50'


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.height:
        config.DATA.HEIGHT = args.height
    if args.width:
        config.DATA.WIDTH = args.width
    if args.pooling:
        config.MODEL.POOLING.NAME = args.pooling
    if args.patch_shuffle:
        config.MODEL.PATCH_SHUFFLE = True
    if args.use_old:
        config.MODEL.USE_OLD_FEATURE = True
    if args.decouple:
        config.MODEL.DECOUPLE = True

    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output
    if args.seed:
        config.SEED = args.seed

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu
    if args.amp:
        config.TRAIN.AMP = True

    # for comparative study
    if args.id_weight:
        config.LOSS.ID_LOSS_WEIGHT = args.id_weight
    if args.clo_weight:
        config.LOSS.CLO_LOSS_WEIGHT = args.clo_weight

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_img_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
