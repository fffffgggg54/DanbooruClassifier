import torch
import torch.cuda.amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DistributedSampler
import torchvision.datasets as dset
from torchvision import models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import time
import glob
import gc
import os
import sys

import contextlib

#import pytorch_optimizer

import multiprocessing

import timm
#import transformers

import timm.layers.ml_decoder as ml_decoder
from timm.data.mixup import FastCollateMixup, Mixup
from timm.data.random_erasing import RandomErasing

import parallelJsonReader
import danbooruDataset
import handleMultiLabel as MLCSL
import models

import timm.optim

import bz2
import pickle
import _pickle as cPickle

from pickle import dump

import scipy.stats

try:
    import plotext
    do_plot = True
except:
    do_plot = False
import re


# ================================================
#           CONFIGURATION OPTIONS
# ================================================

#currGPU = '3090'
#currGPU = 'm40'
#currGPU = 'v100'
currGPU = 'sol_gh200'
#currGPU = 'sol_multi'
#currGPU = 'none'


# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()
FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/danbooru2021/"
if currGPU == 'v100':
    FLAGS['rootPath'] = "/media/fredo/SAMSUNG_500GB/danbooru2021/"
    FLAGS['cocoRoot'] = "/media/fredo/SAMSUNG_500GB/coco2014/"
elif currGPU == 'sol_gh200' or currGPU == 'sol_multi':
    FLAGS['rootPath'] = "/scratch/fyguan/danbooru/"
if(torch.backends.mps.is_built() == True): FLAGS['rootPath'] = "/Users/fredoguan/Datasets/danbooru2021/"
FLAGS['postMetaRoot'] = FLAGS['rootPath'] #+ "TenthMeta/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + "original/"

FLAGS['postListFile'] = FLAGS['postMetaRoot'] + "data_posts.json"
FLAGS['tagListFile'] = FLAGS['postMetaRoot'] + "data_tags.json"
FLAGS['postDFPickle'] = FLAGS['postMetaRoot'] + "postData.pkl"
FLAGS['tagDFPickle'] = FLAGS['postMetaRoot'] + "tagData.pkl"
FLAGS['postDFPickleFiltered'] = FLAGS['postMetaRoot'] + "postDataFiltered.pkl"
FLAGS['tagDFPickleFiltered'] = FLAGS['postMetaRoot'] + "tagDataFiltered.pkl"
FLAGS['postDFPickleFilteredTrimmed'] = FLAGS['postMetaRoot'] + "postDataFilteredTrimmed.pkl"
FLAGS['subsetPickle'] = FLAGS['postMetaRoot'] + "subsetIndices"

'''
if currGPU == '3090':



    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/gernet_l-ASL-BCE/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.backends.mps.is_built() == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    #FLAGS['tagCount'] = 5500
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.7
    FLAGS['progressiveAugRatio'] = 1.8
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config


    FLAGS['ngpu'] = torch.cuda.is_available()
    FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.backends.mps.is_built() == True) else "cpu")
    FLAGS['device2'] = FLAGS['device']
    if(torch.backends.mps.is_built() == True): FLAGS['device2'] = "cpu"
    #FLAGS['use_AMP'] = True if FLAGS['device'] == 'cuda:0' else False
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 30
    FLAGS['postDataServerWorkerCount'] = 3
    if(torch.backends.mps.is_built() == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 100
    FLAGS['batch_size'] = 512
    FLAGS['gradient_accumulation_iterations'] = 4

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0

    FLAGS['finetune'] = False

    FLAGS['channels_last'] = FLAGS['use_AMP']

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

'''

if currGPU == '3090':



    FLAGS['modelDir'] = FLAGS['rootPath'] + "models/regnetx_002-OV_1_of_5_seed42-classEmbedGatingHeadLight-HighRegTest_gte_L_en_v1_5dNoNorm1024-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    
    
    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['dataset'] = 'danbooru'
    #FLAGS['dataset'] = 'coco'
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 224
    FLAGS['actual_image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config

    FLAGS['use_ddp'] = False
    FLAGS['device'] = "cuda"
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 20
    FLAGS['postDataServerWorkerCount'] = 3
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 50
    FLAGS['batch_size'] = 256
    FLAGS['gradient_accumulation_iterations'] = 12

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5
    FLAGS['use_lr_scheduler'] = True

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0
    
    FLAGS['use_mlr_act'] = False
    FLAGS['use_matryoshka_head'] = False
    FLAGS['use_class_embed_head'] = True

    FLAGS['logit_offset'] = True
    FLAGS['logit_offset_multiplier'] = 1.0
    FLAGS['logit_offset_source'] = 'dist'
    FLAGS['opt_dist'] = False
    FLAGS['splc'] = False
    FLAGS['splc_start_epoch'] = 0
    FLAGS['norm_weighted_loss'] = False

    FLAGS['finetune'] = False    #actually a linear probe of a frozen model
    FLAGS['compile_model'] = False
    FLAGS['fast_norm'] = False
    FLAGS['channels_last'] = True

    # tag k-fold cv config
    FLAGS['use_tag_kfold'] = True
    FLAGS['n_folds'] = 5
    FLAGS['current_fold'] = 1

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['store_latents'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False
    
elif currGPU == 'm40':


    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/efficientformerv2_s0-ASL-BCE-T-5500/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.backends.mps.is_built() == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['tagCount'] = 5500
    FLAGS['image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config


    FLAGS['ngpu'] = torch.cuda.is_available()
    FLAGS['device'] = torch.device('cuda:1')

    FLAGS['use_AMP'] = False
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 16
    FLAGS['postDataServerWorkerCount'] = 2
    if(torch.backends.mps.is_built() == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 100
    FLAGS['batch_size'] = 128
    FLAGS['gradient_accumulation_iterations'] = 8

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 0

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0

    FLAGS['finetune'] = False
    
    FLAGS['compile_model'] = False
    FLAGS['channels_last'] = FLAGS['use_AMP']

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

elif currGPU == 'v100':


    FLAGS['modelDir'] = "/media/fredo/Storage1/danbooru_models/scratch/"
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/gc_efficientnetv2_rw_t-448-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convnext_tiny-448-ASL_BCE-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convnext_tiny-448-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convformer_s18-224-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/tresnet_m-224-ASL_BCE_T-5500/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_040h-ASL_GP0_GNADAPC_-224-1588-50epoch/'
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-NormPL_D095_L065-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-ASL_BCE_T-dist_raw-x+20e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/coco_models/vit_large_patch24_gap_448-ASL_adaptivePC-448-100epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-NormPL_D095_L060-ASL_BCE_NormWL_TPOnly-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-PLScratch-PowerGate-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_large_patch24_gap_448-NormPL_D095_L065_ModUpdate_HardMod-ASL_BCE-448-1588-100epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-DLRHead_MlpFC_MlpEstimator_ExplicitTrain-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage1/danbooru_models/regnetz_040_224-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_gap_448-MLRHead_ExplicitTrain-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_gap_448-ml_decoder_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC-ASL_BCE-448-1588-100epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage1/danbooru_models/resnet152-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage1/danbooru_models/vit_base_patch16_gap_448-ml_decoder_NoInProj_NoAttnOutProj_NoMLP_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC-ASL_BCE_T-dist_log_odds-448-1588-100epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage1/danbooru_models/davit_tiny-MatryoshkaHead_K6_full_embedding_half_weight-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage1/coco_models/davit_tiny-ASL_BCE_T-dist_log_odds-448-coco-300epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-ASL_BCE_T-F1-x+80e-1-224-1588-50epoch-RawEval/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-MLR_NW-ADA_WL_T-PU_F_metric-x+10e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-Hill-T-F1-x+00e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-ADA_WL_T-PU_F_Metric-x+10e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-ADA_WL_T-AUL-x+10e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040_ml_decoder_new-MLR_NW-ADA_WL_T-PU_F_Metric-x+20e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_gap_224_ml_decoder_new-ADA_WL_T-PU_F_Metric-x+10e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-FC_PyrH-DLR-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/resnet50-GLU_PyrH-ASL_BCE_T-PU_F_Metric-x+20e-1-448-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-GLU_PyrH-ASL_BCE_T-PU_F_Metric-x+20e-1-448-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/davit_tiny-ml_decoder_no_dupe_class_embed-ASL_BCE-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_gap_448-ASL_BCE_T-PU_F_Metric-x+20e-1-448-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040-GLU_PyrH-ASL_BCE_T-PU_F_Metric-x+20e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-ASL_BCE_T-PU_F_Metric-x+40e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-MetricOPT-P4_T-PU_F_Metric-x+10e-1-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/eva02_large_patch14_224.mim_m38m-FT-ADA_WL_T-P4-x+160e-1-224-1588-10epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_224-gap-ASL_BCE_T-F1-x+00e-1-224-5500-50epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/caformer_s18-gap-ASL_BCE_T-P4-x+80e-1-224-1588-300epoch/"
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_040h-ASL_GP1_GN5_CL005-224-1588-50epoch/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_b16-ASL_BCE_-_T-224-1588/'
    
    
    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['dataset'] = 'danbooru'
    #FLAGS['dataset'] = 'coco'
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 224
    FLAGS['actual_image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config

    FLAGS['use_ddp'] = True
    FLAGS['device'] = None 
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 10
    FLAGS['postDataServerWorkerCount'] = 3
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 50
    FLAGS['batch_size'] = 64
    FLAGS['gradient_accumulation_iterations'] = 6

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5
    FLAGS['use_lr_scheduler'] = True

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0
    
    FLAGS['use_mlr_act'] = False
    FLAGS['use_matryoshka_head'] = False
    FLAGS['use_class_embed_head'] = False

    FLAGS['logit_offset'] = True
    FLAGS['logit_offset_multiplier'] = 1.0
    FLAGS['logit_offset_source'] = 'dist'
    FLAGS['opt_dist'] = False
    FLAGS['splc'] = False
    FLAGS['splc_start_epoch'] = 0
    FLAGS['norm_weighted_loss'] = False

    FLAGS['finetune'] = False    #actually a linear probe of a frozen model
    FLAGS['compile_model'] = False
    FLAGS['fast_norm'] = True
    FLAGS['channels_last'] = True

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['store_latents'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

elif currGPU == 'sol_gh200':
    #FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/scratch/"
    #FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/davit_tiny-OV_1_of_5_seed42-ml_decoder_NoInProj_NoAttnOutProj_NoMLP_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/davit_tiny-OV_1_of_5_seed42-classEmbedGatingHead2048_HighQueryNoiseAug_PosNegRandQueryAug_gte_L_en_v1_5dNoNorm1024-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    #FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/convformer_s18-ml_decoder_NoMlp_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC-ASL_BCE_T-dist_log_odds-224-1588-50epoch/"
    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['dataset'] = 'danbooru'
    #FLAGS['dataset'] = 'coco'
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 224
    FLAGS['actual_image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config

    FLAGS['use_ddp'] = False
    FLAGS['device'] = "cuda"
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 20
    FLAGS['postDataServerWorkerCount'] = 3
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 50
    FLAGS['batch_size'] = 256
    FLAGS['gradient_accumulation_iterations'] = 12

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5
    FLAGS['use_lr_scheduler'] = True

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0
    
    FLAGS['use_mlr_act'] = False
    FLAGS['use_matryoshka_head'] = False
    FLAGS['use_class_embed_head'] = True

    FLAGS['logit_offset'] = True
    FLAGS['logit_offset_multiplier'] = 1.0
    FLAGS['logit_offset_source'] = 'dist'
    FLAGS['opt_dist'] = False
    FLAGS['splc'] = False
    FLAGS['splc_start_epoch'] = 0
    FLAGS['norm_weighted_loss'] = False

    FLAGS['finetune'] = False    #actually a linear probe of a frozen model
    FLAGS['compile_model'] = False
    FLAGS['fast_norm'] = False
    FLAGS['channels_last'] = True

    # tag k-fold cv config
    FLAGS['use_tag_kfold'] = True
    FLAGS['n_folds'] = 5
    FLAGS['current_fold'] = 1

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['store_latents'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

elif currGPU == 'sol_multi':
    #FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/scratch/"
    #FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/davit_tiny-OV_1_of_5_seed42-classEmbedGatingHead2048_gte_L_en_v1_5dNoNorm1024-ASL_BCE_T-dist_log_odds_W-InvClassProp-224-1588-50epoch/"
    FLAGS['modelDir'] = "/scratch/fyguan/danbooru_models/davit_tiny-OV_1_of_5_seed42-classEmbedGatingHead2048_HighDrop_PreNorm_QueryNoiseAug_RandQueryAug_gte_L_en_v1_5dNoNorm1024-ASL_BCE_T-dist_log_odds_W-InvClassProp-224-1588-50epoch/"

    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['dataset'] = 'danbooru'
    #FLAGS['dataset'] = 'coco'
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 224
    FLAGS['actual_image_size'] = 224
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config

    FLAGS['use_ddp'] = True
    FLAGS['device'] = None
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 20
    FLAGS['postDataServerWorkerCount'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 50
    FLAGS['batch_size'] = 768
    FLAGS['gradient_accumulation_iterations'] = 1

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5
    FLAGS['use_lr_scheduler'] = True

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0
    
    FLAGS['use_mlr_act'] = False
    FLAGS['use_matryoshka_head'] = False
    FLAGS['use_class_embed_head'] = True

    FLAGS['logit_offset'] = True
    FLAGS['logit_offset_multiplier'] = 1.0
    FLAGS['logit_offset_source'] = 'dist'
    FLAGS['opt_dist'] = False
    FLAGS['splc'] = False
    FLAGS['splc_start_epoch'] = 0
    FLAGS['norm_weighted_loss'] = True

    FLAGS['finetune'] = False    #actually a linear probe of a frozen model
    FLAGS['compile_model'] = False
    FLAGS['fast_norm'] = False
    FLAGS['channels_last'] = True

    # tag k-fold cv config
    FLAGS['use_tag_kfold'] = True
    FLAGS['n_folds'] = 5
    FLAGS['current_fold'] = 1

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['store_latents'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

elif currGPU == 'none':


    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/tf_efficientnetv2_s-ASL-BCE/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.backends.mps.is_built() == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config

    FLAGS['image_size'] = 384
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 3.0
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config


    FLAGS['ngpu'] = torch.cuda.is_available()
    FLAGS['device'] = torch.device("cpu")
    FLAGS['device2'] = FLAGS['device']
    if(torch.backends.mps.is_built() == True): FLAGS['device2'] = "cpu"
    #FLAGS['use_AMP'] = True if FLAGS['device'] == 'cuda:0' else False
    FLAGS['use_AMP'] = False
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 1
    FLAGS['postDataServerWorkerCount'] = 1
    if(torch.backends.mps.is_built() == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 101
    FLAGS['batch_size'] = 512
    FLAGS['gradient_accumulation_iterations'] = 2

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 99

    FLAGS['finetune'] = False

    FLAGS['channels_last'] = FLAGS['use_AMP']

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = True

classes = None
myDataset = None



timm.layers.fast_norm.set_fast_norm(enable=FLAGS['fast_norm'])

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


'''
serverProcessPool = []
workQueue = multiprocessing.Queue()
'''

def getSubsetByID(dataset, postData, lower, upper, div = 1000):
    '''
    is_head_proc = not FLAGS['use_ddp'] or dist.get_rank() == 0
    subsetName = FLAGS['subsetPickle'] + '-' + str(lower) + '-' + str(upper) + '.pkl'
    try:
        print("attempting to read pickled subset information at " + subsetName)
        indices = pd.read_pickle(subsetName)
    except:
        indices = postData[lower <= postData['id'] % 1000][upper > postData['id'] % 1000].index
        if is_head_proc:
            print("saving subset pickled subset information to " + subsetName)
            indices.to_pickle(subsetName)
    '''
    return torch.utils.data.Subset(dataset, postData[lower <= postData['id'] % 1000][upper > postData['id'] % 1000].index.tolist())

def getData():
    global classes
    if FLAGS['dataset'] == 'danbooru':
        startTime = time.time()
        

        #tagData = pd.read_pickle(FLAGS['tagDFPickle'])
        #postData = pd.read_pickle(FLAGS['postDFPickle'])
        
        
        
        '''
        try:
            print("attempting to read pickled post metadata file at " + FLAGS['tagDFPickle'])
            tagData = pd.read_pickle(FLAGS['tagDFPickle'])
        except:
            print("pickled post metadata file at " + FLAGS['tagDFPickle'] + " not found")
            tagData = parallelJsonReader.dataImporter(FLAGS['tagListFile'])   # read tags from json in parallel
            print("saving pickled post metadata to " + FLAGS['tagDFPickle'])
            tagData.to_pickle(FLAGS['tagDFPickle'])
        
        #postData = pd.concat(map(dataImporter, glob.iglob(postMetaDir + 'posts*')), ignore_index=True) # read all post metadata files in metadata dir
        try:
            print("attempting to read pickled post metadata file at " + FLAGS['postDFPickle'])
            postData = pd.read_pickle(FLAGS['postDFPickle'])
        except:
            print("pickled post metadata file at " + FLAGS['postDFPickle'] + " not found")
            postData = parallelJsonReader.dataImporter(FLAGS['postListFile'], keep = 1)    # read posts
            print("saving pickled post metadata to " + FLAGS['postDFPickle'])
            postData.to_pickle(FLAGS['postDFPickle'])
            
            
        
        print("got posts, time spent: " + str(time.time() - startTime))
        
        # filter data
        startTime = time.time()
        print("applying filters")
        # TODO this filter process is slow, need to speed it up, currently only single threaded
        tagData, postData = danbooruDataset.filterDanbooruData(tagData, postData)   # apply various filters to preprocess data
        
        tagData.to_pickle(FLAGS['tagDFPickleFiltered'])
        postData.to_pickle(FLAGS['postDFPickleFiltered'])
        '''
        if FLAGS['tagCount'] == 1588:
            tagData = pd.read_pickle(FLAGS['tagDFPickleFiltered'])
            #tagData.to_pickle(FLAGS['tagDFPickleFiltered'])
        elif FLAGS['tagCount'] == 5500:
            tagData = pd.read_csv(FLAGS['rootPath'] + 'selected_tags.csv')
        postData = pd.read_pickle(FLAGS['postDFPickleFilteredTrimmed'])
        #postData.to_pickle(FLAGS['postDFPickleFilteredTrimmed'])
        #print(postData.info())
        
        # get posts that are not banned
        #queryStartTime = time.time()
        #postData.query("is_banned == False", inplace = True)
        #blockedIDs = [5190773, 5142098, 5210705, 5344403, 5237708, 5344394, 5190771, 5237705, 5174387, 5344400, 5344397, 5174384]
        #for postID in blockedIDs: postData.query("id != @postID", inplace = True)
        #print("banned post query time: " + str(time.time()-queryStartTime))
        

        
        #postData = postData[['id', 'tag_string', 'file_ext', 'file_url']]
        #postData = postData.convert_dtypes()
        #print(postData.info())
        #postData.to_pickle(FLAGS['postDFPickleFilteredTrimmed'])
        
        

        print("finished preprocessing, time spent: " + str(time.time() - startTime))
        print(f"got {len(postData)} posts with {len(tagData)} tags") #got 3821384 posts with 423 tags
        
        '''
        for nthWorkerProcess in range(FLAGS['postDataServerWorkerCount']):
            currProcess = multiprocessing.Process(target=danbooruDataset.DFServerWorkerProcess,
                                                  args=(workQueue,
                                                        postData.copy(deep=True),
                                                        pd.Series(tagData.name, dtype=pd.StringDtype()),
                                                        FLAGS['imageRoot'],
                                                        FLAGS['cacheRoot'],),
                                                  daemon = True)
            currProcess.start()
            serverProcessPool.append(currProcess)
            
        '''    
        # TODO custom normalization values that fit the dataset better
        # TODO investigate ways to return full size images instead of crops
        # this should allow use of full sized images that vary in size, which can then be fed into a model that takes images of arbitrary resolution
        '''
        myDataset = danbooruDataset.DanbooruDataset(FLAGS['imageRoot'], postData, tagData.name, transforms.Compose([
            #transforms.Resize((224,224)),
            danbooruDataset.CutoutPIL(cutout_factor=0.5),
            transforms.RandAugment(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            cacheRoot = FLAGS['cacheRoot']
            )
        '''
        '''
        myDataset = danbooruDataset.DanbooruDatasetWithServer(FLAGS['imageRoot'],
                                                              workQueue,
                                                              len(postData),
                                                              tagData.name,
                                                              transforms.Compose([#transforms.Resize((224,224)),
                                                                                  danbooruDataset.CutoutPIL(cutout_factor=0.5),
                                                                                  transforms.RandAugment(),
                                                                                  transforms.ToTensor(),
                                                                                  #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                                                  ]),
                                                              cacheRoot = FLAGS['cacheRoot'])
        
        '''
        
        
        
        global myDataset
        if currGPU == 'v100' or currGPU == '3090':
            myDataset = danbooruDataset.DanbooruDatasetWithServer(
                postData,
                tagData,
                FLAGS['imageRoot'],
                FLAGS['cacheRoot'],
                FLAGS['image_size'],
                FLAGS['postDataServerWorkerCount'])
        
        else:
            myDataset = danbooruDataset.DanbooruDatasetWithServerAndReader(
                postData,
                tagData,
                danbooruDataset.TarReader(FLAGS['rootPath'] + '/' + 'danbooru2021_' + str(FLAGS['image_size']) + '.tar'),
                danbooruDataset.TarReader(FLAGS['rootPath'] + '/' + 'tags.tar'),
                FLAGS['image_size'],
                FLAGS['postDataServerWorkerCount']
            )
            
            
            
        '''
        myDataset = danbooruDataset.DanbooruDatasetWithServer(workQueue,
                                                             len(postData),
                                                             None)
        '''                                                     
        
        classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
        
        #classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
        #trimmedSet, _ = torch.utils.data.random_split(myDataset, [int(FLAGS['workingSetSize'] * len(myDataset)), len(myDataset) - int(FLAGS['workingSetSize'] * len(myDataset))], generator=torch.Generator().manual_seed(42)) # discard part of dataset if desired
        
        # TODO implement modulo-based subsets for splits to standardize train/test sets and potentially a future val set for thresholding or wtv
        
        #trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset
        
        trainSet = getSubsetByID(myDataset, postData, 0, 900)
        testSet = getSubsetByID(myDataset, postData, 900, 930)
    elif FLAGS['dataset'] == 'coco':
        trainSet = danbooruDataset.CocoDataset(
            FLAGS['cocoRoot']+'train2014/',
            FLAGS['cocoRoot']+'annotations/instances_train2014.json'
        )
        testSet = danbooruDataset.CocoDataset(
            FLAGS['cocoRoot']+'val2014/',
            FLAGS['cocoRoot']+'annotations/instances_val2014.json'
        )
        classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    
    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

from typing import Optional
from functools import partial

import torch
from torch import nn
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn
import timm.layers.ml_decoder as ml_decoder
from timm.layers import NormMlpClassifierHead, ClassifierHead
from timm.layers.classifier import _create_pool
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
#MLDecoder = partial(ml_decoder.MLDecoderLegacy, simple_group_fc = True)
MLDecoder = ml_decoder.MLDecoder

class MLDecoderHead(nn.Module):
    """MLDecoder wrapper with forward compatible with ClassifierHead"""

    def __init__(self, in_features, num_classes, pool_type='avg', use_conv=False, input_fmt='NCHW'):
        super(MLDecoderHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        self.global_pool, num_pooled_features = _create_pool(in_features, num_classes, pool_type, use_conv=use_conv, input_fmt=input_fmt)
        self.head = MLDecoder(in_features=in_features, num_classes=num_classes)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()


    def reset(self, num_classes, global_pool=None):
        if global_pool is not None:
            if global_pool != self.global_pool.pool_type:
                self.global_pool, _ = _create_pool(self.in_features, num_classes, global_pool, use_conv=self.use_conv)
            self.flatten = nn.Flatten(1) if self.use_conv and global_pool else nn.Identity()
        num_pooled_features = self.in_features * self.global_pool.feat_mult()
        self.head = MLDecoder(in_features=in_features, num_classes=num_classes)


    def forward(self, x, pre_logits: bool = False):
        # pool for compatibility with ClassifierHead
        if self.input_fmt == 'NHWC':
            x = x.permute(0, 3, 1, 2)
        if pre_logits:
            x = self.global_pool(x)
            return x.flatten(1)
        else:
            x = self.head(x)
            return self.flatten(x)

def add_ml_decoder_head(model):

    # ignore CoaT, crossvit
    # ignore distillation models: deit_distilled, efficientformer V2
    num_classes = model.num_classes
    num_features = model.num_features

    assert num_classes > 0, "MLDecoder requires a model to have num_classes > 0"

    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.Identity()
        del model.fc

        model.fc = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'fc_norm') or 'Cait' in model._get_name(): # ViT, BEiT, EVA
        model.global_pool = None # disable any pooling, model instantiation leaves 1 norm layer after features, [B, n + K x K, C]
        if hasattr(model, 'attn_pool'):
            model.attn_pool = None
        model.head_drop = nn.Identity()
        model.head = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif 'MetaFormer' in model._get_name():
        if hasattr(model.head, 'flatten'):  # ConvNext case
            model.head.flatten = nn.Identity()
        model.head.global_pool = nn.Identity()
        model.head.drop = nn.Identity()
        del model.head.fc
        model.head.fc = MLDecoder(num_classes=num_classes, in_features=num_features)

    # maybe  and isinstance(model.head, (NormMlpClassifierHead, ClassifierHead) ?
    elif hasattr(model, 'head'):    # ClassifierHead, nn.Sequential
        input_fmt = getattr(model.head, 'input_fmt', 'NCHW')
        model.head = MLDecoderHead(num_features, num_classes)
        if hasattr(model, 'global_pool'):
            if(isinstance(model.global_pool, nn.Module)):
                model.global_pool = nn.Identity()
            else:
                model.global_pool = None
        if hasattr(model, 'head_drop'):
            model.head_drop = nn.Identity()

    elif 'MobileNetV3' in model._get_name(): # mobilenetv3 - conflict with efficientnet

        model.flatten = nn.Identity()
        del model.classifier
        model.classifier = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.Identity()
        del model.classifier
        model.classifier = MLDecoder(num_classes=num_classes, in_features=num_features)
    elif hasattr(model, 'global_pool') and hasattr(model, 'last_linear'):  # InceptionV4
        model.global_pool = nn.Identity()
        del model.last_linear
        model.last_linear = MLDecoder(num_classes=num_classes, in_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classif'):  # InceptionResnetV2
        model.global_pool = nn.Identity()
        del model.classif
        model.classif = MLDecoder(num_classes=num_classes, in_features=num_features)

    else:
        raise Exception("Model code-writing is not aligned currently with ml-decoder")

    # FIXME does not work
    if hasattr(model, 'drop_rate'):  # Ml-Decoder has inner dropout
        model.drop_rate = 0
    return model

from timm.layers import create_norm_layer, GluMlp, Mlp, SelectAdaptivePool2d, get_act_layer, create_act_layer
class StarAct(nn.Module):
    """
    StarAct: s * act(x) ** 2 + b
    """

    def __init__(
            self,
            act_fn='relu',
            scale_value=1.0,
            bias_value=0.0,
            scale_learnable=True,
            bias_learnable=True,
            mode=None,
            inplace=False
    ):
        super().__init__()
        self.inplace = inplace
        self.act = create_act_layer(act_fn, inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.act(x) ** 2 + self.bias
class PyramidFeatureAggregationModel(nn.Module):
    def __init__(
        self,
        model,
        num_classes = 1000,
        head_type = "glu",
        head_act = "silu",
    ):
        super().__init__()
        self.model = model

        self.feature_dims = model.feature_info.channels()
        self.num_features = sum(self.feature_dims)

        self.norms = nn.ModuleList([create_norm_layer('layernorm2d', dim) for dim in self.feature_dims])
        self.pools = nn.ModuleList([SelectAdaptivePool2d(pool_type='fast_avg', flatten=True) for dim in self.feature_dims])
        self.num_classes = num_classes

        if(head_type == "fc"):
            self.head = nn.Linear(self.num_features, self.num_classes)
        elif(head_type == "mlp"):
            self.head = Mlp(
                in_features = self.num_features,
                hidden_features = int(4*self.num_features),
                out_features = self.num_classes,
                act_layer = get_act_layer(head_act),
                norm_layer = None,
            )
        elif(head_type == "glu"):
            self.head = GluMlp(
                in_features = self.num_features,
                hidden_features = int(2*2.5*self.num_features),
                out_features = self.num_classes,
                act_layer = get_act_layer(head_act),
                norm_layer = None,
            )
        elif(head_type == "dlr"):
            self.head = MLCSL.DualLogisticRegression(self.num_features, self.num_classes)
        
        
    def forward(self, x):
        x=self.model(x)
        #x = torch.column_stack([nn.functional.gelu(out).mean((-2, -1)) for out in x]) # NCHW only for now
        #return self.head(x)

        x = torch.column_stack([pool(norm(out)) for pool, norm, out in zip(self.pools, self.norms, x)])
        return self.head(x)

class FeaturePyramid2Token(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
        self.feature_dims = model.feature_info.channels()
        self.norms = nn.ModuleList([create_norm_layer('layernorm2d', dim) for dim in self.feature_dims])
        
        
        
def modelSetup(classes):
    
    

    '''
    myCvtConfig = transformers.CvtConfig(num_channels=3,
        patch_sizes=[7, 5, 3, 3],
        patch_stride=[4, 3, 2, 2],
        patch_padding=[2, 2, 1, 1],
        embed_dim=[64, 240, 384, 896],
        num_heads=[1, 3, 6, 14],
        depth=[1, 4, 8, 16],
        mlp_ratio=[4.0, 4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.0, 0.1],
        qkv_bias=[True, True, True, True],
        cls_token=[False, False, False, True],
        qkv_projection_method=['dw_bn', 'dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv=[3, 3, 3, 3],
        padding_kv=[1, 1, 1, 1],
        stride_kv=[2, 2, 2, 2],
        padding_q=[1, 1, 1, 1],
        stride_q=[1, 1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        num_labels=len(classes))
    '''
    
    # custom cvt
    
    #model = transformers.CvtForImageClassification(myCvtConfig)
    
    # pytorch builtin models
    
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    #model = models.resnet152()
    #model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    #model = models.resnet34()
    #model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    #model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    
    #model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    # regular timm models
    
    #model = timm.create_model('efficientformerv2_s0', pretrained=False, num_classes=len(classes), drop_path_rate=0.05)
    #model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True, num_classes=len(classes), drop_path_rate=0.6)
    #model = timm.create_model('gernet_s', pretrained=False, num_classes=len(classes), drop_path_rate = 0.0)
    #model = timm.create_model('edgenext_small', pretrained=False, num_classes=len(classes), drop_path_rate = 0.15)
    model = timm.create_model('davit_tiny', pretrained=False, num_classes=len(classes), drop_path_rate = 0.2)
    #model = timm.create_model('vit_medium_shallow_patch16_gap_224', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('vit_base_patch16_siglip_gap_224.v2_webli', pretrained=True, num_classes=len(classes), drop_path_rate = 0.3)
    #model = timm.create_model('regnetz_040', pretrained=False, num_classes=len(classes), drop_path_rate=0.15)
    #model = timm.create_model('vit_base_patch16_gap_224', pretrained=False, num_classes=len(classes), drop_path_rate=0.4)
    #model = timm.create_model('vit_base_patch16_gap_224', pretrained=False, num_classes=len(classes), drop_path_rate=0.4, patch_size=32, img_size=FLAGS['actual_image_size'])
    #model = timm.create_model('vit_huge_patch14_gap_224', pretrained=True, pretrained_cfg_overlay=dict(file="./jepa-latest.pth.tar"))
    #model = timm.create_model('ese_vovnet99b_iabn', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1, drop_rate=0.02)
    #model = timm.create_model('tresnet_m', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('efficientvit_b3', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('eva02_large_patch14_224.mim_m38m', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('vit_base_patch16_gap_224', pretrained=False, num_classes=len(classes), drop_path_rate=0.4, img_size=448)
    
    #model = timm.create_model('davit_tiny', pretrained=False, features_only=True, drop_path_rate=0.2)
    #model = PyramidFeatureAggregationModel(model, len(classes), head_type='dlr')
    '''
    model = timm.create_model(
        'vit_base_patch16_224', 
        img_size = FLAGS['actual_image_size'], 
        patch_size = 28, 
        global_pool='avg', 
        class_token = False, 
        qkv_bias=False, 
        init_values=1e-6, 
        fc_norm=False,
        pretrained=False, 
        num_classes=len(classes), 
        drop_path_rate = 0.4, 
        drop_rate=0.02
    )
    '''
    
    # gap model
    # vit_large_patch16_gap_448: p16 d1024 L24 nh16
    # vit_base_patch16_gap_448: p16 d768 L12 nh12
    # vit_medium_patch16_gap_224: p16 d512 L12 nh8
    '''
    model = timm.models.VisionTransformer(
        img_size = FLAGS['actual_image_size'], 
        patch_size = 16,
        num_classes = len(classes), 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        global_pool='avg', 
        class_token = False,
        no_embed_class=True, 
        qkv_bias=False, 
        init_values=1e-6, 
        fc_norm=False,
        drop_path_rate=0.3)
    '''
    '''
    model = models.TagEmbedCrossAttentionViT(
        torch.load(f'./DanbooruWikiEmbeddings{str(FLAGS['tagCount'])}_gte_large_en_v1.5_no_norm_d1024.pth', map_location='cpu', weights_only=True),
        img_size = FLAGS['actual_image_size'], 
        patch_size = 16,
        num_classes = 1, 
        embed_dim=384, 
        depth=8, 
        num_heads=6, 
        drop_path_rate=0.1,
    )
    '''
    r'''
    model.reset_classifier(0)
    state_dict = torch.load('/media/fredo/Storage3/danbooru_models/vit_base_patch16_gap_448-ml_decoder_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC-ASL_BCE-448-1588-100epoch/saved_model_epoch_99.pth', map_location=torch.device('cpu'), weights_only=True)
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r"^0\.", "", k)
        if "head" not in k:
            out_dict[k] = v
    model.load_state_dict(out_dict)
    model.reset_classifier(len(classes))
    '''
    # cvt
    
    #model = transformers.CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    #model.classifier = nn.Linear(model.config.embed_dim[-1], len(classes))

    # regular huggingface models

    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-384", num_labels=len(classes), ignore_mismatched_sizes=True)
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", num_labels=len(classes), ignore_mismatched_sizes=True)
    
    if FLAGS['finetune'] == True: 
        #model.reset_classifier(num_classes=len(classes))
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        if hasattr(model, "head_dist"):
            for param in model.head_dist.parameters():
                param.requires_grad = True
    
    # modified timm models with custom head with hidden layers
    '''
    model = timm.create_model('mixnet_s', pretrained=True, num_classes=-1) # -1 classes for identity head by default
    
    model = nn.Sequential(model,
                          nn.LazyLinear(len(classes)),
                          nn.ReLU(),
                          nn.Linear(len(classes), len(classes)))
    
    '''
    

    #model = ml_decoder.add_ml_decoder_head(model, num_groups = 1588)
    '''
    model = ml_decoder.add_ml_decoder_head(
        model, 
        num_groups = 1588, 
        class_embed = torch.load('./DanbooruWikiEmbeddings1588.pth', map_location='cpu', weights_only=True),
        class_embed_merge = 'concat',)
    '''
    '''
    # ml_decoder_no_dupe_class_embed_add_LearnableQueryEmbed_sharedFC
    model = ml_decoder.add_ml_decoder_head(
        model, 
        num_groups = 0, 
        class_embed = torch.load('./DanbooruWikiEmbeddings1588.pth', map_location='cpu', weights_only=True),
        class_embed_merge = 'add',
        learnable_embed = True,
        shared_fc = True,)
    '''
    '''
    # ml_decoder_NoInProj_NoAttnOutProj_NoMLP_no_dupe_OnlyClassEmbed_gte_L_en_v1_5dNoNorm1024_sharedFC
    model = ml_decoder.add_ml_decoder_head(
        model,
        num_groups = 0,
        class_embed = torch.load(f'./DanbooruWikiEmbeddings{str(FLAGS['tagCount'])}_gte_large_en_v1.5_no_norm_d1024.pth', map_location='cpu', weights_only=True),
        class_embed_merge = '',
        shared_fc = True,
        post_input_proj_act = True,
        use_input_proj = False,
        attn_out_proj = False,
        use_mlp = False,
    )
    '''
    '''
    model = ml_decoder.add_ml_decoder_head(
        model,
        num_groups = 0,
        class_embed = torch.load(f'./DanbooruWikiEmbeddings{str(FLAGS['tagCount'])}_gte_large_en_v1.5_no_norm_d1024.pth', map_location='cpu', weights_only=True),
        class_embed_merge = '',
        shared_fc = True,
        post_input_proj_act = True,
        use_input_proj = True,
        attn_out_proj = True,
        use_mlp = False,
    )
    '''
    num_features = model.num_features

    if FLAGS['use_mlr_act'] == True or FLAGS['use_matryoshka_head'] == True or FLAGS['use_class_embed_head'] == True:
        model.reset_classifier(0)
        
    model = nn.Sequential(model)
    

    #model.train()
    
    # threshold as a neural network module, everything about its use needs to be handled manually, nested optimization loops are a headache
    #threshold_penalty = MLCSL.thresholdPenalty(FLAGS['logit_offset_multiplier'], initial_threshold = 0.5, lr = 1e-5, threshold_min = 0.1, threshold_max = 0.9, num_classes = len(classes))
    
    # I mean it's really just an activation fn with trainable weights
    #mlr_act = MLCSL.ModifiedLogisticRegression(num_classes = len(classes), initial_weight = 1.0, initial_beta = 0.0, eps = 1e-8)
    
    
    

    if FLAGS['use_mlr_act'] == True:
        #mlr_act = MLCSL.ModifiedLogisticRegression_NoWeight(num_classes = len(classes), initial_beta = 1.0, eps = 1e-8)
        #model.append(mlr_act)
        '''
        mlr_head = MLCSL.ModifiedLogisticRegression_Head(num_features, num_classes = len(classes), bias=True, initial_beta = 1.0, eps = 1e-8)
        '''
        mlr_head = MLCSL.DualLogisticRegression_Head(
                num_features,
                num_classes=len(classes),
                fc_type='linear',
                estimator_type='linear',
                bias_fc=True,
                bias_estimator=True,
                eps=1e-8)
        
        model.append(mlr_head)
    elif FLAGS['use_matryoshka_head'] == True:
        model.append(MLCSL.MatryoshkaClassificationHead(num_features, len(classes), k=6))
    elif FLAGS['use_class_embed_head'] == True:
        model.append(MLCSL.ClassEmbedClassifierHead(
            num_features, 
            len(classes), 
            torch.load('./DanbooruWikiEmbeddings1588_gte_large_en_v1.5_no_norm_d1024.pth', map_location='cpu', weights_only=True),
            in_drop=0.0,
            embed_drop=0.1,
            head_drop=0.0,
            use_query_noise=True,
            query_noise_strength=1.0,
            use_random_query=True,
            num_random_query=2,
            pre_norm=False,
        ))
    #model = torch.compile(model, options={'max_autotune': True, 'epilogue_fusion': True})

    
    return model
    
def getDataLoader(dataset, batch_size, epoch, use_dist_sampler):
    distSampler = None
    if(use_dist_sampler):
        distSampler = DistributedSampler(dataset=dataset, shuffle=True, seed=17, drop_last=True)
        distSampler.set_epoch(epoch)
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler=distSampler, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=3, pin_memory = True, generator=torch.Generator().manual_seed(41))
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=2, pin_memory = True, generator=torch.Generator().manual_seed(41))

def trainCycle(image_datasets, model):
    #print("starting training")
    startTime = time.time()

    
    
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']

    if FLAGS['use_tag_kfold']:
        mask = torch.rand(len(classes), generator = torch.Generator().manual_seed(42)) * FLAGS['n_folds']
        mask = (mask >= (FLAGS['current_fold'] - 1)) & (mask < FLAGS['current_fold'])
        mask = mask.to(device)
        inv_mask = ~mask
        print(sum(mask))
        print((sum(inv_mask)))
    
    is_head_proc = not FLAGS['use_ddp'] or dist.get_rank() == 0
    
    memory_format = torch.channels_last if FLAGS['channels_last'] else torch.contiguous_format
    
    
    model = model.to(device, memory_format=memory_format)
    #mlr_act = MLCSL.ModifiedLogisticRegression_NoWeight(num_classes = len(classes), initial_beta = 0.0, eps = 1e-8)
    #mlr_act = mlr_act.to(device, memory_format = memory_format)

    if is_head_proc:
        print(model)
    
    if (FLAGS['resume_epoch'] > 0) and is_head_proc:
        state_dict = torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth', map_location=torch.device('cpu'), weights_only=True)
        #out_dict={}
        #for k, v in state_dict.items():
        #    k = k.replace('_orig_mod.', '')
        #    k = k.replace('module.', '')
        #    out_dict[k] = v
            
        model.load_state_dict(state_dict)
        #mlr_act_state_dict = torch.load(FLAGS['modelDir'] + 'mlr_act_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth', map_location=torch.device('cpu'))
        #mlr_act.load_state_dict(mlr_act_state_dict)
    
    
    

        
    

    print("initialized training, time spent: " + str(time.time() - startTime))
    

    
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=tagWeights.to(FLAGS['device']))
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    # Observe that all parameters are being optimized
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    #expLRScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # use partial label approaches from http://arxiv.org/abs/2110.10955v1
    #ema = MLCSL.ModelEma(model, 0.9997)  # 0.9997^641=0.82
    
    
    #criterion = MLCSL.Hill()
    #criterion = MLCSL.SPLC(gamma=2.0)
    #criterion = MLCSL.SPLCModified(gamma=2.0)
    #criterion = MLCSL.AdaptiveWeightedLoss(initial_weight = 0.0, lr = 1e-4, weight_limit = 1e5)
    criterion = MLCSL.AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0.0, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.AsymmetricLossOptimized(gamma_neg=5, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.AsymmetricLossAdaptive(gamma_neg=1, gamma_pos=1, clip=0.0, eps=1e-8, disable_torch_grad_focal_loss=False, adaptive = True, gap_target = 0.1, gamma_step = 0.003)
    #criterion = MLCSL.AsymmetricLossAdaptiveWorking(gamma_neg=1, gamma_pos=0, clip=0.0, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.003)
    #criterion = MLCSL.GapWeightLoss(initial_weight=0., gap_target=0.1, weight_step=1e-3)
    #criterion = MLCSL.PartialSelectiveLoss(device, prior_path=None, clip=0.05, gamma_pos=1, gamma_neg=6, gamma_unann=4, alpha_pos=1, alpha_neg=1, alpha_unann=1)
    #parameters = MLCSL.add_weight_decay(model, FLAGS['weight_decay'])
    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum=0.9)
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = pytorch_optimizer.Lamb(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = timm.optim.Adan(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])

    
    
    if (FLAGS['use_ddp'] == True):
        model = DDP(model, device_ids=[FLAGS['device']], gradient_as_bucket_view=True)
        #mlr_act = DDP(mlr_act, device_ids=[FLAGS['device']], gradient_as_bucket_view=True)
        
    if(FLAGS['compile_model'] == True):
        model = torch.compile(model)
    
    #mlr_act_opt = timm.optim.Adan(mlr_act.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    
    #mixup = Mixup(mixup_alpha = 0.2, cutmix_alpha = 0, num_classes = len(classes))
    
    boundaryCalculator = MLCSL.getDecisionBoundaryWorking(initial_threshold = 0.5, lr = 1e-5, threshold_min = 0.1, threshold_max = 0.9)
    '''
    boundaryCalculator = None
    for name, module in model.named_modules():
        if(type(module) == MLCSL.getDecisionBoundaryWorking):
            boundaryCalculator = module
            break
    '''
    
    #dist_tracker = MLCSL.DistributionTrackerEMA()

    if (FLAGS['resume_epoch'] > 0):
        boundaryCalculator.thresholdPerClass = torch.load(FLAGS['modelDir'] + 'thresholds.pth', weights_only=True).to(device)
    
    if (FLAGS['resume_epoch'] > 0):
        optimizer.load_state_dict(torch.load(FLAGS['modelDir'] + 'optimizer' + '.pth', map_location=torch.device("cpu"), weights_only=True))
        
    
    if (FLAGS['use_scaler'] == True): scaler = torch.amp.GradScaler('cuda')
    
    # end MLCSL code
    
    losses = []
    best = None
    tagNames = list(classes.values())

    world_size = dist.get_world_size() if FLAGS['use_ddp'] else 1
    
    # histogram bins
    bins=100

    firstLoop = False
    
    MeanStackedAccuracyStored = torch.Tensor([2,1,2,1])
    if(is_head_proc):
        print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    epoch = FLAGS['resume_epoch']
    
    offset = torch.special.logit(torch.Tensor([0.65])).to(device)
    
    while (epoch < FLAGS['num_epochs']):
        #prior = MLCSL.ComputePrior(classes, device)
        epochTime = time.time()
        
        dataloaders = {x: getDataLoader(image_datasets[x], FLAGS['batch_size'], epoch, FLAGS['use_ddp']) for x in image_datasets} # set up dataloaders

        if FLAGS['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
            scheduler.last_epoch = len(dataloaders['train'])*epoch

        
        if(is_head_proc):
            print("starting epoch: " + str(epoch))
        AP_regular = []
        AccuracyRunning = []
        AP_ema = []
        targets_running = None
        preds_running = None
        latent_features_running = None
        targets_running = []
        preds_running = []
        latent_features_running = []
        textOutput = None
        #lastPrior = None
        
        phases = ['train', 'val'] if FLAGS['val'] == False else ['val']
        currPhase = 0
        
        while currPhase < len(phases):
            phase = phases[currPhase]
            
            cm_tracker = MLCSL.MetricTracker()
            dist_tracker = MLCSL.DistributionTracker()

            if FLAGS['use_tag_kfold']:
                cm_tracker_holdout = MLCSL.MetricTracker()

            #try:
            if phase == 'train':
                model.train()  # Set model to training mode
                #mlr_act.train()
                #boundaryCalculator.train()
                #if (hasTPU == True): xm.master_print("training set")
                if(is_head_proc): print("training set")
                
                if FLAGS['progressiveImageSize'] == True:
                    dynamicResizeDim = int(FLAGS['image_size']*FLAGS['progressiveSizeStart'] + epoch * (FLAGS['image_size']-FLAGS['image_size']*FLAGS['progressiveSizeStart'])/FLAGS['num_epochs'])
                else:
                    dynamicResizeDim = FLAGS['actual_image_size']
                
                if(is_head_proc):
                    print(f'Using image size of {dynamicResizeDim}x{dynamicResizeDim}')
                
                newTransform = transforms.Compose([transforms.Resize((dynamicResizeDim, dynamicResizeDim)),
                                                          transforms.RandAugment(magnitude = epoch, num_magnitude_bins = int(FLAGS['num_epochs'] * FLAGS['progressiveAugRatio'])),
                                                          #transforms.RandAugment(),
                                                          transforms.RandomHorizontalFlip(),
                                                          #transforms.TrivialAugmentWide(),
                                                          #danbooruDataset.CutoutPIL(cutout_factor=0.2),
                                                          transforms.ToTensor(),
                                                          RandomErasing(probability=0.3, mode='pixel', device='cpu'),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])

            if phase == 'val':

                model.eval()   # Set model to evaluate mode
                #mlr_act.eval()
                #boundaryCalculator.eval()
                print("validation set")
                if(FLAGS['skip_test_set'] == True and (epoch != FLAGS['num_epochs'] - 1)):
                    print("skipping...")
                    break;
                
                newTransform = transforms.Compose([transforms.Resize((FLAGS['actual_image_size'], FLAGS['actual_image_size'])),
                                                          transforms.ToTensor(),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])
            
            #myDataset.transform = newTransform
            if hasattr(dataloaders[phase].dataset, 'dataset'):
                # modify original dataset if using a subset
                dataloaders[phase].dataset.dataset.transform = newTransform
            else:
                dataloaders[phase].dataset.transform = newTransform
            
            # For each batch in the dataloader
            '''
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/latestRun'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
                ) as prof:
            '''
            
            loaderIterable = enumerate(dataloaders[phase])
            if(firstLoop): print("got loader iterable")
            #if(is_head_proc): torch.cuda.memory._record_memory_history(enabled='all')
            for i, (images, tags) in loaderIterable:

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == 'train'):
                    # TODO switch between using autocast and not using it
                    
                    with torch.amp.autocast('cuda', enabled=FLAGS['use_AMP'], dtype = torch.float16 if currGPU == 'v100' else torch.bfloat16):

                        if FLAGS['use_mlr_act'] == True:   
                            preds = model(imageBatch)
                            outputs = torch.special.logit(preds)
                        elif FLAGS['use_matryoshka_head'] == True:
                            if FLAGS['use_ddp']:
                                latent_features = model.module[0](imageBatch)
                                outputs_all = model.module[1](latent_features)
                            else:
                                latent_features = model[0](imageBatch)
                                outputs_all = model[1](latent_features)
                            outputs_all = outputs_all.float()
                            outputs = outputs_all[0]
                            preds = torch.sigmoid(outputs)
                            matryoshka_loss_weights = torch.ones_like(outputs_all, requires_grad=False)
                            matryoshka_loss_weights[0] = outputs_all.shape[0] - 1
                        elif FLAGS['use_class_embed_head'] == True:
                            if FLAGS['use_ddp']:
                                latent_features = model.module[0](imageBatch)
                                outputs_all = model.module[1](latent_features)
                            else:
                                latent_features = model[0](imageBatch)
                                outputs_all = model[1](latent_features)
                            outputs_all = outputs_all.float()
                            # random query agumentation
                            if outputs_all.shape[1] > len(classes):
                                num_random_query = outputs_all.shape[1] - len(classes)
                                random_query_logits = outputs_all[:, -num_random_query:]
                                outputs_all = outputs_all[:, :-num_random_query].contiguous()
                                use_random_query = True
                            else:
                                use_random_query = False
                            outputs = outputs_all
                            preds = torch.sigmoid(outputs)
                        else:
                            if(phase == 'val'):
                                if FLAGS['use_ddp']:
                                    latent_features = model.module[0].forward_features(imageBatch)
                                    outputs = model.module[0].forward_head(latent_features)
                                    #latent_features = model.module[0](imageBatch)
                                    #outputs = model.module[1](latent_features)
                                else:
                                    latent_features = model[0].forward_features(imageBatch)
                                    outputs = model[0].forward_head(latent_features)
                            else:
                                outputs = model(imageBatch).float()
                            #outputs = model(imageBatch).logits
                            outputs_all = outputs.float()
                            preds = torch.sigmoid(outputs).float()

                        if(firstLoop): print("got fwd pass")

                        with torch.amp.autocast('cuda', enabled=False):
                        
                            # update boundary
                            boundaryCalculator(
                                preds.detach(),
                                tagBatch,
                                update=(phase == "train"),
                                step_opt=((i+1) % FLAGS['gradient_accumulation_iterations'] == 0)
                            )
                            if(firstLoop): print("got boundary update")
                            
                            # allreduce boundary during boundary optimizer updates
                            if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                torch.cuda.synchronize()
                                if FLAGS['use_ddp'] == True:
                                    with torch.no_grad():
                                        torch.distributed.all_reduce(boundaryCalculator.thresholdPerClass, op = torch.distributed.ReduceOp.AVG)
                            #offset = torch.Tensor([0.5]).to(device) if boundaryCalculator.needs_init else boundaryCalculator.thresholdPerClass.detach()
                            
                            # performance metric tracking
                            #predsModified=preds
                            #multiAccuracy = MLCSL.getAccuracy(predsModified.to(device2), tagBatch.to(device2))
                            if FLAGS['use_ddp']:
                                all_logits = [torch.ones_like(outputs) for _ in range(dist.get_world_size())]
                                torch.distributed.all_gather(all_logits, outputs)
                                if(FLAGS['opt_dist']): all_logits[dist.get_rank()] = outputs
                                all_logits = torch.cat(all_logits)
                            else:
                                all_logits = outputs

                            dist_tracker.set_device(all_logits.device)
                            with torch.no_grad():
                                if FLAGS['use_tag_kfold']:
                                    multiAccuracy = cm_tracker.update(preds.detach()[:, inv_mask], tagBatch.to(device)[:, inv_mask])
                                    multiAccuracyHoldout = cm_tracker_holdout.update(preds.detach()[:, mask], tagBatch.to(device)[:, mask])
                                else:
                                    #multiAccuracy = cm_tracker.update((preds.detach() > offset.detach()).float().to(device), tagBatch.to(device))
                                    multiAccuracy = cm_tracker.update(preds.detach(), tagBatch.to(device))
                                if(firstLoop): print("got CM update")
                                
                                
                                if not FLAGS['splc']:
                                    if FLAGS['use_ddp']:
                                        all_tags = torch.empty(dist.get_world_size() * tagBatch.shape[0], tagBatch.shape[1], device=outputs.device, dtype=tagBatch.dtype)
                                        torch.distributed.all_gather_into_tensor(all_tags, tagBatch)
                                    else:
                                        all_tags = tagBatch
                                    #dist_tracker(all_logits.to(torch.float64), all_tags.to(torch.long))

                        outputs = outputs.float()
                        '''
                        if phase == 'val':
                            #output_ema = torch.sigmoid(ema.module(imageBatch)).cpu()
                            output_regular = preds.cpu()
                        #loss = criterion(torch.mul(preds, tagBatch), tagBatch)
                        #loss = criterion(outputs, tagBatch)
                        '''
                        


                        tagsModified = tagBatch
                        # target modifications
                        if FLAGS['splc'] and epoch >= FLAGS['splc_start_epoch']:
                            with torch.no_grad():
                                
                                #targs = torch.where(preds > offset.detach(), torch.tensor(1).to(preds), labels) # hard SPLC
                                #tagsModified = ((1 - tagsModified) * MLCSL.stepAtThreshold(preds, offset) + tagsModified) # soft SPLC
                                tagsModified = MLCSL.adjust_labels(outputs.detach(), tagsModified, dist_tracker, clip_dist = 0.95, clip_logit = 0.65)
                                if FLAGS['use_ddp']:
                                    all_tags = torch.empty(dist.get_world_size() * tagBatch.shape[0], tagBatch.shape[1], device=outputs.device, dtype=tagsModified.dtype)

                                    torch.distributed.all_gather_into_tensor(all_tags, tagsModified)
                                    #torch.distrubuted.all_gather_into_tensor(all_tags, tagBatch)
                                else:
                                    all_tags = tagsModified
                        dist_tracker(all_logits.to(torch.float64), all_tags.to(torch.long))
                        if(firstLoop): print("got dist tracker update")
                        
                        # loss weighing
                        if FLAGS['norm_weighted_loss']:
                            loss_weight = MLCSL.generate_loss_weights(outputs.detach(), tagBatch, dist_tracker)
                        else: loss_weight = torch.ones(len(classes), device=device)
                        if(phase == 'train' and hasattr(criterion, 'update')):
                            criterion.update(outputs.detach(), tagsModified.to(device), update=(phase == "train"),
                                step_opt=((i+1) % FLAGS['gradient_accumulation_iterations'] == 0))
                                
                            if((FLAGS['use_ddp'] == True) and ((i+1) % FLAGS['gradient_accumulation_iterations'] == 0)):
                                torch.cuda.synchronize()
                                with torch.no_grad():
                                    torch.distributed.all_reduce(criterion.weight_per_class, op = torch.distributed.ReduceOp.AVG)
                                    
                        # logit offset
                        if FLAGS['logit_offset']:
                            #outputs = outputs - torch.special.logit(offset)
                            if FLAGS['logit_offset_source'] == 'dist':
                                with torch.no_grad():
                                    #offset = (dist_tracker.pos_mean.detach() + dist_tracker.neg_mean.detach()) / 2
                                    # use log odds
                                    offset = dist_tracker.log_odds.detach()
                            else:
                                offset = torch.special.logit(boundaryCalculator.thresholdPerClass.detach())
                            outputs_all = outputs_all + FLAGS['logit_offset_multiplier'] * offset
                            if(firstLoop): print("got logit offset")

                        
                        #loss = criterion(outputs.to(device2), tagBatch.to(device2), lastPrior)
                        if FLAGS['use_class_embed_head'] and use_random_query:
                            loss_weight = torch.cat([loss_weight[inv_mask], torch.tensor([1] * num_random_query, device=device)])
                            if num_random_query == 2:
                                augmented_targs = torch.cat([tagsModified.to(device)[:, inv_mask], torch.ones_like(tagsModified[:,0]).unsqueeze(1), torch.zeros_like(tagsModified[:,0]).unsqueeze(1)], dim=1)
                            else:
                                augmented_targs = torch.cat([tagsModified.to(device)[:, inv_mask], *([torch.ones_like(tagsModified[:,0]).unsqueeze(1)] * num_random_query)], dim=1)
                            loss = criterion(torch.cat([outputs_all.to(device)[:, inv_mask], random_query_logits], dim=1), augmented_targs, weight = loss_weight)
                        else:
                            loss = criterion(outputs_all.to(device)[:, inv_mask], tagsModified.to(device)[:, inv_mask], weight = loss_weight[inv_mask])
                        #loss = criterion(outputs_all.to(device), tagsModified.to(device), weight = matryoshka_loss_weights)
                        #loss = criterion(outputs.to(device), tagsModified.to(device), weight = loss_weight)
                        #loss += (((dist_tracker.pos_mean + dist_tracker.neg_mean) ** 2) ** 0.25).sum() #+ dist_tracker.pos_std.sum() + dist_tracker.neg_std.sum()
                        #loss -= ((dist_tracker.pos_mean - dist_tracker.neg_mean) / ((dist_tracker.pos_var + dist_tracker.neg_var) ** 0.5 + 1e-8)).sum()
                        #loss = criterion(outputs.to(device), tagsModified.to(device), ddp=FLAGS['use_ddp'])
                        #loss = criterion(outputs.to(device) - torch.special.logit(offset), tagBatch.to(device))
                        #loss = criterion(outputs.to(device2), tagBatch.to(device2), epoch)
                        #loss, textOutput = criterion(outputs.to(device), tagsModified.to(device), updateAdaptive = (phase == 'train'), printAdaptive = ((i % stepsPerPrintout == 0) and is_head_proc))
                        #loss, textOutput = criterion(outputs.to(device), tagBatch.to(device), updateAdaptive = (phase == 'train'))
                        #loss = criterion(outputs.cpu(), tags.cpu())
                        
                        #loss = (1 - multiAccuracy[:,4:]).pow(2).mul(torch.Tensor([2,1,2,1]).to(device2)).sum()
                        #loss = (1 - multiAccuracy[:,4:]).pow(2).sum()
                        #loss = (1 - multiAccuracy[:,6:7]).pow(2).sum()     # high precision with easy classes
                        #loss = (multiAccuracy[:,1] + multiAccuracy[:,2]).pow(2).sum()
                        #loss = criterion(multiAccuracy, referenceTable)
                        #loss = (multiAccuracy - referenceTable).pow(2).sum()
                        #loss = (-torch.log(multiAccuracy[0,4:])).sum()
                        #loss = (1 - multiAccuracy[:,4:]).pow(2).div(MeanStackedAccuracyStored.to(device2)).sum()
                        #loss = (1 - multiAccuracy[:,4:]).sum()
                        #loss = (1 - multiAccuracy[:,4:]).div(MeanStackedAccuracyStored.to(device2)).sum()
                        #loss = (1 - multiAccuracy[:,4:]).div(MeanStackedAccuracyStored.to(device2)).pow(2).sum()
                        #loss = (1 - multiAccuracy[:,8]).pow(2).sum()
                        #loss = -MLCSL.getSingleMetric(outputs.sigmoid(), tagsModified, MLCSL.P4).sum()
                        #loss = -MLCSL.AUL(outputs.sigmoid(), tagsModified).sum()
                        #loss = -MLCSL.AUROC(outputs.sigmoid(), tagsModified).sum()
                        #model.zero_grad()
                        
                        if(firstLoop): print("got loss")
                        
                        
                    # backward + optimize only if in training phase
                    if phase == 'train' and (loss.isnan() == False):
                        if (FLAGS['use_scaler'] == True):   # cuda gpu case
                            with model.no_sync() if FLAGS['use_ddp'] else contextlib.nullcontext():
                                scaler.scale(loss).backward()
                            if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                torch.cuda.synchronize()
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                                dist_tracker._zero_grad()
                                
                                #nn.utils.clip_grad_norm_(mlr_act.parameters(), max_norm=1.0, norm_type=2)
                                #scaler.step(mlr_act_opt)
                                #scaler.update()
                                #mlr_act_opt.zero_grad(set_to_none=True)
                                
                        else:                               # apple gpu/cpu case
                            with model.no_sync() if FLAGS['use_ddp'] else contextlib.nullcontext():
                                loss.backward()
                            if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                torch.cuda.synchronize()
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                                
                                #nn.utils.clip_grad_norm_(mlr_act.parameters(), max_norm=1.0, norm_type=2)
                                #lr_act_opt.step()
                                #mlr_act_opt.zero_grad(set_to_none=True)
                    
                    
                        #torch.cuda.synchronize()
                    
                        #ema.update(model)
                        #prior.update(outputs.to(device))
                    '''    
                    if(i==20):
                        if(is_head_proc):
                            s = torch.cuda.memory._snapshot()
                            with open(FLAGS['modelDir'] + "mem_snapshot.pickle", "wb") as f:
                                dump(s, f)
                            torch.cuda.memory._record_memory_history(enabled=None)
                        exit()
                    '''
                    if (phase == 'val'):
                        # for mAP calculation
                        # FIXME this is super slow and bottlenecked, figure out a faster way to do validation with correctly calculated metrics
                        if(FLAGS['use_ddp'] == True):
                            targets_all = None
                            preds_all = None
                            if FLAGS['store_latents']: latent_features_all = None
                            if(is_head_proc):
                                targets_all = [torch.zeros_like(tagBatch) for _ in range(dist.get_world_size())]
                                preds_all = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
                                if FLAGS['store_latents']: latent_features_all = [torch.zeros_like(latent_features) for _ in range(dist.get_world_size())]
                            torch.distributed.gather(tagBatch, gather_list = targets_all, async_op=True)
                            torch.distributed.gather(preds, gather_list = preds_all, async_op=True)
                            if FLAGS['store_latents']: torch.distributed.gather(latent_features, gather_list = latent_features_all, async_op=True)
                            if(is_head_proc):
                                targets_all = torch.cat(targets_all).detach().cpu()
                                preds_all = torch.cat(preds_all).detach().cpu()
                                if FLAGS['store_latents']: latent_features_all = torch.cat(latent_features_all).detach().cpu()
                        else:
                            targets_all = tags
                            preds_all = preds.detach().cpu()
                            if FLAGS['store_latents']: latent_features_all = latent_features.detach().cpu()
                        
                        if is_head_proc:
                            targets_running.append(targets_all.detach().clone())
                            preds_running.append(preds_all.detach().clone())
                            if FLAGS['store_latents']: latent_features_running.append(latent_features_all.detach().clone())
                            
                            #AP_ema.append(MLCSL.mAP(targets, preds_ema))
                            #AccuracyRunning.append(multiAccuracy)
                            targets_all = None
                            preds_all = None
                            if FLAGS['store_latents']: latent_features_all = None
            
                #print(device)
                if i % stepsPerPrintout == 0:
                    torch.cuda.synchronize()
                    if(firstLoop): print("got printout sync")

                    imagesPerSecond = (dataloaders[phase].batch_size*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()
                    
                    if(FLAGS['use_ddp'] == True):
                        #imagesPerSecond = torch.Tensor([imagesPerSecond]).to(device)
                        #torch.distributed.all_reduce(imagesPerSecond, op = torch.distributed.ReduceOp.SUM)
                        #imagesPerSecond = imagesPerSecond.cpu()
                        #imagesPerSecond = imagesPerSecond.item()
                        imagesPerSecond = imagesPerSecond * world_size

                    if(firstLoop): print("got imgs/sec sync")

                    #currPostTags = []
                    #batchTagAccuracy = list(zip(tagNames, perTagAccuracy.tolist()))
                    
                    # TODO find better way to generate this output that doesn't involve iterating, zip()?
                    #for tagIndex, tagVal in enumerate(torch.mul(preds, tagBatch)[0]):
                    #    if tagVal.item() != 0:
                    #        currPostTags.append((tagNames[tagIndex], tagVal.item()))
                    
                   
                    #print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tAccuracy: %.2f\tP4: %.2f\t%s' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy, multiAccuracy.mean(dim=0) * 100, textOutput))
                    
                    if is_head_proc:
                        if (phase == 'train'):
                            targets_batch = tags.numpy(force=True)
                            preds_regular_batch = preds.detach().numpy(force=True)
                            accuracy = MLCSL.mAP(targets_batch, preds_regular_batch)
                        else:
                            targets = torch.cat(targets_running).numpy(force=True)
                            preds_regular = torch.cat(preds_running).numpy(force=True)
                            #preds_ema = output_ema.cpu().detach().numpy()
                            accuracy = MLCSL.mAP(targets, preds_regular)
                        if(firstLoop): print("got mAP")
                        torch.set_printoptions(linewidth = 200, sci_mode = False)
                        print(f"[{epoch}/{FLAGS['num_epochs']}][{i}/{len(dataloaders[phase])}]\tLoss: {loss.detach():.4f}\tImages/Second: {imagesPerSecond:.4f}\tAccuracy: {accuracy:.2f}\t {[f'{num:.4f}' for num in list((multiAccuracy.detach() * 100))]}\t{textOutput}")
                        if FLAGS['use_tag_kfold']: print(f'kfold holdout perfomance metrics: {[f'{num:.4f}' for num in list((multiAccuracyHoldout.detach() * 100))]}')
                        torch.set_printoptions(profile='default')
                        if(firstLoop): print("info print call")
                        #print(dist_tracker.dump())
                        cohen_d_scores = MLCSL.cohen_d_effect_size(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std)
                        cohen_d_scores = cohen_d_scores.detach()
                        #t_stat = (dist_tracker.pos_mean-dist_tracker.neg_mean)/((dist_tracker.pos_var/(dist_tracker.pos_count + 1e-8) + dist_tracker.neg_var/(dist_tracker.neg_count + 1e-8)) ** 0.5 + 1e-8)
                        #print(f't score mean: {t_stat.mean()} std: {t_stat.std()}')
                        #t_p_values = scipy.stats.ttest_ind_from_stats(dist_tracker.pos_mean.cpu().numpy(), dist_tracker.pos_std.cpu().numpy(), dist_tracker.pos_count.cpu().numpy(), dist_tracker.neg_mean.cpu().numpy(), dist_tracker.neg_std.cpu().numpy(), dist_tracker.neg_count.cpu().numpy(), equal_var=False, alternative="greater").pvalue
                        if FLAGS['use_tag_kfold']:
                            print(f"cohen's d mean: {cohen_d_scores[inv_mask].mean()}, std: {cohen_d_scores[inv_mask].std()}, pos mean: {dist_tracker.pos_mean.detach()[inv_mask].mean()}, neg mean: {dist_tracker.neg_mean.detach()[inv_mask].mean()}")
                            print(f"holdout: cohen's d mean: {cohen_d_scores[mask].mean()}, std: {cohen_d_scores[mask].std()}, pos mean: {dist_tracker.pos_mean.detach()[mask].mean()}, neg mean: {dist_tracker.neg_mean.detach()[mask].mean()}")
                        else:
                            print(f"cohen's d mean: {cohen_d_scores.mean()}, std: {cohen_d_scores.std()}, pos mean: {dist_tracker.pos_mean.detach().mean()}, neg mean: {dist_tracker.neg_mean.detach().mean()}")
                        if(firstLoop): print("dist print call")
                        '''
                        plotext.hist(dist_tracker.neg_mean.detach().clamp(min=-15), bins, label='Neg means')
                        plotext.hist(dist_tracker.pos_mean.detach(), bins, label='Pos means')
                        plotext.title("Distributions of per-class means")
                        plotext.show()
                        plotext.clear_figure()
                        plotext.hist(((dist_tracker.pos_mean.detach() + dist_tracker.neg_mean.detach()) / 2).clamp(min=-10, max=10), bins, label='Mean of means')
                        plotext.title("Distributions of per-class mean of means")
                        plotext.show()
                        plotext.clear_figure()
                        '''
                    #print(id[0])
                    #print(currPostTags)
                    #print(sorted(batchTagAccuracy, key = lambda x: x[1], reverse=True))
                    
                    #torch.cuda.empty_cache()
                #losses.append(loss)
                '''
                if (phase == 'val'):
                    if best is None:
                        best = (float(loss), epoch, i, accuracy.item())
                    elif best[0] > float(loss):
                        best = (float(loss), epoch, i, accuracy.item())
                        print(f"NEW BEST: {best}!")
                '''
                if phase == 'train' and FLAGS['use_lr_scheduler']:
                    scheduler.step()
                    if(firstLoop): print("called scheduler.step()")
                
                #print(device)
                #if(FLAGS['ngpu'] > 0):
                    #torch.cuda.empty_cache()
                firstLoop = False
                sys.stdout.flush()
            
                    
            if FLAGS['use_ddp'] == True:
                torch.cuda.synchronize()
                torch.distributed.all_reduce(cm_tracker.running_confusion_matrix, op=torch.distributed.ReduceOp.AVG)
                if FLAGS['use_tag_kfold']: torch.distributed.all_reduce(cm_tracker_holdout.running_confusion_matrix, op=torch.distributed.ReduceOp.AVG)
                
                #torch.distributed.all_reduce(criterion.gamma_neg_per_class, op = torch.distributed.ReduceOp.AVG)
            if ((phase == 'val') and (FLAGS['skip_test_set'] == False or epoch == FLAGS['num_epochs'] - 1) and is_head_proc):
                if(epoch == FLAGS['num_epochs'] - 1):
                    print("saving eval data")
                    modelOutputs = {'labels':torch.cat(targets_running).cpu(), 'preds':torch.cat(preds_running).cpu()}
                    #print(modelOutputs)
                    cachePath = FLAGS['modelDir'] + "evalOutputs.pkl.bz2"
                    with bz2.BZ2File(cachePath, 'w') as cachedSample: cPickle.dump(modelOutputs, cachedSample)
                    #torch.set_printoptions(profile="full")
                    
                    #AvgAccuracy = torch.stack(AccuracyRunning)
                    #AvgAccuracy = AvgAccuracy.mean(dim=0)
                    AvgAccuracy = cm_tracker.get_full_metrics()
                    LabelledAccuracy = list(zip(AvgAccuracy.tolist(), tagNames, boundaryCalculator.thresholdPerClass.data))
                    LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][8], reverse=True)
                    
                    if(is_head_proc): print(*LabelledAccuracySorted, sep="\n")
                    #torch.set_printoptions(profile="default")
                MeanStackedAccuracy = cm_tracker.get_aggregate_metrics()
                MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
                if(is_head_proc):
                    print((MeanStackedAccuracy*100).tolist())
                    if(FLAGS['use_tag_kfold']): print((cm_tracker_holdout.get_aggregate_metrics()*100).tolist())
                
                if dist_tracker.neg_mean.sum().isnan() == False and do_plot:
                    if FLAGS['use_tag_kfold']:
                        plotext.hist(dist_tracker.neg_mean.detach()[inv_mask].clamp(min=-15), bins, label='Neg means')
                        plotext.hist(dist_tracker.pos_mean.detach()[inv_mask], bins, label='Pos means')
                        plotext.title("Distributions of per-class means (seen)")
                        plotext.show()
                        plotext.clear_figure()

                        plotext.hist(dist_tracker.neg_mean.detach()[mask].clamp(min=-15), bins, label='Neg means')
                        plotext.hist(dist_tracker.pos_mean.detach()[mask], bins, label='Pos means')
                        plotext.title("Distributions of per-class means (holdout)")
                        plotext.show()
                        plotext.clear_figure()
                    else:
                        plotext.hist(dist_tracker.neg_mean.detach().clamp(min=-15), bins, label='Neg means')
                        plotext.hist(dist_tracker.pos_mean.detach(), bins, label='Pos means')
                        plotext.title("Distributions of per-class means")
                        plotext.show()
                        plotext.clear_figure()

                    plotext.hist(dist_tracker.log_odds.detach().clamp(min=-15, max=15), bins, label='Log odds')
                    plotext.title("Distributions of per-class log odds ratio")
                    plotext.show()
                    plotext.clear_figure()

                    if dist_tracker.pos_mean.sum().isnan() == False:
                        plotext.hist(((dist_tracker.pos_mean.detach() + dist_tracker.neg_mean.detach()) / 2).clamp(min=-10, max=10), bins, label='Mean of means')
                        plotext.title("Distributions of per-class mean of means")
                        plotext.show()
                        plotext.clear_figure()

                        z_scores = MLCSL.z_score(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.pos_count, dist_tracker.neg_mean, dist_tracker.neg_std, dist_tracker.neg_count).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(z_scores[inv_mask], bins, label="z-score (seen)")
                            plotext.hist(z_scores[mask], bins, label="z-score (holdout)")
                        else:
                            plotext.hist(z_scores, bins, label="z-score")
                        plotext.title("Distributions of per-class z-score")
                        plotext.show()
                        plotext.clear_figure()

                        cohen_d_scores = MLCSL.cohen_d_effect_size(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(cohen_d_scores[inv_mask], bins, label="cohen's d (seen)")
                            plotext.hist(cohen_d_scores[mask], bins, label="cohen's d (holdout)")
                        else:
                            plotext.hist(cohen_d_scores, bins, label="cohen's d")
                        plotext.title("Distributions of per-class cohen's d")
                        plotext.show()
                        plotext.clear_figure()

                        kl_div = MLCSL.kl_divergence_univariate_normal(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(kl_div[inv_mask], bins, label="KL divergence (seen)")
                            plotext.hist(kl_div[mask], bins, label="KL divergence (holdout)")
                        else:
                            plotext.hist(kl_div, bins, label="KL divergence")
                        plotext.title("Distributions of per-class KL divergence")
                        plotext.show()
                        plotext.clear_figure()

                        hellinger_distance = MLCSL.hellinger_distance_univariate_normal(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(hellinger_distance[inv_mask], bins, label="Hellinger distance (seen)")
                            plotext.hist(hellinger_distance[mask], bins, label="Hellinger distance (holdout)")
                        else:
                            plotext.hist(hellinger_distance, bins, label="Hellinger distance")
                        plotext.title("Distributions of per-class Hellinger distance")
                        plotext.show()
                        plotext.clear_figure()

                        bhattacharyya_distance = MLCSL.bhattacharyya_distance_univariate_normal(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(bhattacharyya_distance[inv_mask], bins, label="Bhattacharyya distance (seen)")
                            plotext.hist(bhattacharyya_distance[mask], bins, label="Bhattacharyya distance (holdout)")
                        else:
                            plotext.hist(bhattacharyya_distance, bins, label="Bhattacharyya distance")
                        plotext.title("Distributions of per-class Bhattacharyya distance")
                        plotext.show()
                        plotext.clear_figure()

                        wasserstein_2_distance = MLCSL.wasserstein_2_distance_univariate_normal(dist_tracker.pos_mean, dist_tracker.pos_std, dist_tracker.neg_mean, dist_tracker.neg_std).detach()
                        if FLAGS['use_tag_kfold']:
                            plotext.hist(wasserstein_2_distance[inv_mask], bins, label="2-Wasserstein distance (seen)")
                            plotext.hist(wasserstein_2_distance[mask], bins, label="2-Wasserstein distance (holdout)")
                        else:
                            plotext.hist(wasserstein_2_distance, bins, label="2-Wasserstein distance")
                        plotext.title("Distributions of per-class 2-Wasserstein distance")
                        plotext.show()
                        plotext.clear_figure()

                        
                
                
                #prior.save_prior()
                #prior.get_top_freq_classes()
                #lastPrior = prior.avg_pred_train
                #if(is_head_proc): print(lastPrior[:30])
                
                #mAP_score_regular = MLCSL.mAP(torch.cat(targets_running).numpy(force=True), torch.cat(preds_running).numpy(force=True))
                #mAP_score_ema = np.mean(AP_ema)
                #if(is_head_proc): print("mAP score regular {:.2f}".format(mAP_score_regular))
                #top_mAP = max(mAP_score_regular, mAP_score_ema)
                if hasattr(criterion, 'tau_per_class'):
                    if(is_head_proc): print(criterion.tau_per_class)
                #print(boundaryCalculator.thresholdPerClass)
                #print(criterion.weight_per_class)
                
            currPhase += 1
            optimizer.zero_grad(set_to_none=True)
            if(boundaryCalculator.opt): boundaryCalculator.zero_grad(set_to_none=True)
            #mlr_act_opt.zero_grad(set_to_none=True)

            '''
            except Exception as e:
                print(e)
                
                batch_size = int(dataloaders[phase].batch_size / 2)
                print(f'setting batch size of {phase} dataloader to {batch_size}')
                
                dataloaders[phase] = getDataLoader(image_datasets[phase], batch_size)
                
                if phase == 'train':
                    FLAGS['gradient_accumulation_iterations'] = FLAGS['gradient_accumulation_iterations'] * 2
                    print(f"setting training gradient accumulation epochs to {FLAGS['gradient_accumulation_iterations']}")
            '''
                        
        
        
        
        # save everything
        if FLAGS['val'] == False and is_head_proc:
            print('saving checkpoint')
            modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
            state_dict = model.state_dict()
            
            out_dict={}
            for k, v in state_dict.items():
                k = k.replace('_orig_mod.', '')
                k = k.replace('module.', '')
                out_dict[k] = v
            
            torch.save(out_dict, modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
            #torch.save(mlr_act.state_dict(), modelDir + 'mlr_act_epoch_' + str(epoch) + '.pth')
            if(epoch > 0):
                os.remove(modelDir + 'saved_model_epoch_' + str(epoch - 1) + '.pth')
                #os.remove(modelDir + 'mlr_act_epoch_' + str(epoch - 1) + '.pth')
            torch.save(boundaryCalculator.thresholdPerClass, modelDir + 'thresholds.pth')
            torch.save(optimizer.state_dict(), modelDir + 'optimizer' + '.pth')
            pd.DataFrame(tagNames).to_pickle(modelDir + "tags.pkl")

            if FLAGS['store_latents']: torch.save(torch.cat(latent_features_running).cpu(), modelDir + 'eval_embeds.pth')
            
        time_elapsed = time.time() - epochTime
        if(is_head_proc): print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        epoch += 1
        '''
        if(is_head_proc):
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size(), obj.grad_fn, obj.grad)
                        
                except: pass
            print(torch.cuda.memory_summary(device = device))
        '''
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if(is_head_proc): print()
            
    #time_elapsed = time.time() - startTime
    #print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files
    if FLAGS['use_ddp']:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        FLAGS['device'] = rank % torch.cuda.device_count()
        torch.cuda.set_device(FLAGS['device'])
        torch.cuda.empty_cache()
        FLAGS['learning_rate'] *= dist.get_world_size()
    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)
    if FLAGS['use_ddp']:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()