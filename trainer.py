import torch
import torch.cuda.amp
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
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

import torch_optimizer

import multiprocessing

import timm
import transformers

import timm.layers.ml_decoder as ml_decoder
from timm.data.mixup import FastCollateMixup, Mixup
from timm.data.random_erasing import RandomErasing

import parallelJsonReader
import danbooruDataset
import handleMultiLabel as MLCSL

import timm.optim




# ================================================
#           CONFIGURATION OPTIONS
# ================================================

#currGPU = '3090'
#currGPU = 'm40'
currGPU = 'v100'
#currGPU = 'none'


# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()
FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/danbooru2021/"
#if currGPU == 'v100': FLAGS['rootPath'] = "/media/fredo/SAMSUNG_500GB/danbooru2021/"
if(torch.has_mps == True): FLAGS['rootPath'] = "/Users/fredoguan/Datasets/danbooru2021/"
FLAGS['postMetaRoot'] = FLAGS['rootPath'] #+ "TenthMeta/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + "original/"

FLAGS['postListFile'] = FLAGS['postMetaRoot'] + "data_posts.json"
FLAGS['tagListFile'] = FLAGS['postMetaRoot'] + "data_tags.json"
FLAGS['postDFPickle'] = FLAGS['postMetaRoot'] + "postData.pkl"
FLAGS['tagDFPickle'] = FLAGS['postMetaRoot'] + "tagData.pkl"
FLAGS['postDFPickleFiltered'] = FLAGS['postMetaRoot'] + "postDataFiltered.pkl"
FLAGS['tagDFPickleFiltered'] = FLAGS['postMetaRoot'] + "tagDataFiltered.pkl"
FLAGS['postDFPickleFilteredTrimmed'] = FLAGS['postMetaRoot'] + "postDataFilteredTrimmed.pkl"

'''
if currGPU == '3090':



    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/gernet_l-ASL-BCE/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
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
    FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
    FLAGS['device2'] = FLAGS['device']
    if(torch.has_mps == True): FLAGS['device2'] = "cpu"
    #FLAGS['use_AMP'] = True if FLAGS['device'] == 'cuda:0' else False
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 30
    FLAGS['postDataServerWorkerCount'] = 3
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
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



    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convnext_base-ASL_BCE_T-448-1588/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 448
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 1.6
    FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
    #FLAGS['cacheRoot'] = None

    FLAGS['workingSetSize'] = 1
    FLAGS['trainSetSize'] = 0.8

    # device config


    FLAGS['ngpu'] = torch.cuda.is_available()
    FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
    FLAGS['device2'] = FLAGS['device']
    if(torch.has_mps == True): FLAGS['device2'] = "cpu"
    #FLAGS['use_AMP'] = True if FLAGS['device'] == 'cuda:0' else False
    FLAGS['use_AMP'] = True
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 20
    FLAGS['postDataServerWorkerCount'] = 3
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 100
    FLAGS['batch_size'] = 32
    FLAGS['gradient_accumulation_iterations'] = 32

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 4

    FLAGS['finetune'] = False
    FLAGS['compile_model'] = True
    FLAGS['channels_last'] = FLAGS['use_AMP']

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False
    
elif currGPU == 'm40':


    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/efficientformerv2_s0-ASL-BCE-T-5500/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
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
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
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



    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetx_320-448-ASL_BCE_T-1588_v100/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config
    FLAGS['tagCount'] = 1588
    FLAGS['image_size'] = 448
    FLAGS['actual_image_size'] = 448
    FLAGS['progressiveImageSize'] = False
    FLAGS['progressiveSizeStart'] = 0.5
    FLAGS['progressiveAugRatio'] = 1.6
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
    FLAGS['postDataServerWorkerCount'] = 2
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 100
    FLAGS['batch_size'] = 16
    FLAGS['gradient_accumulation_iterations'] = 8

    FLAGS['base_learning_rate'] = 3e-3
    FLAGS['base_batch_size'] = 2048
    FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
    FLAGS['lr_warmup_epochs'] = 5

    FLAGS['weight_decay'] = 2e-2

    FLAGS['resume_epoch'] = 0

    FLAGS['finetune'] = False
    FLAGS['compile_model'] = True
    FLAGS['fast_norm'] = False
    FLAGS['channels_last'] = FLAGS['use_AMP']

    # debugging config

    FLAGS['verbose_debug'] = False
    FLAGS['skip_test_set'] = False
    FLAGS['stepsPerPrintout'] = 50
    FLAGS['val'] = False

elif currGPU == 'none':


    FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/tf_efficientnetv2_s-ASL-BCE/'


    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
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
    if(torch.has_mps == True): FLAGS['device2'] = "cpu"
    #FLAGS['use_AMP'] = True if FLAGS['device'] == 'cuda:0' else False
    FLAGS['use_AMP'] = False
    FLAGS['use_scaler'] = FLAGS['use_AMP']
    #if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

    # dataloader config

    FLAGS['num_workers'] = 1
    FLAGS['postDataServerWorkerCount'] = 1
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
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
torch.backends.cuda.matmul.allow_tf32 = True if FLAGS['device'] == 'cuda:0' else False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


'''
serverProcessPool = []
workQueue = multiprocessing.Queue()
'''
def getData():
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
    elif FLAGS['tagCount'] == 5500:
        tagData = pd.read_csv(FLAGS['rootPath'] + 'selected_tags.csv')
    postData = pd.read_pickle(FLAGS['postDFPickleFilteredTrimmed'])
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
    myDataset = danbooruDataset.DanbooruDatasetWithServer(
        postData,
        tagData,
        FLAGS['imageRoot'],
        FLAGS['cacheRoot'],
        FLAGS['image_size'],
        FLAGS['postDataServerWorkerCount'])
        
        
    '''
    myDataset = danbooruDataset.DanbooruDatasetWithServer(workQueue,
                                                         len(postData),
                                                         None)
    '''                                                     
    global classes
    classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
    
    #classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
    trimmedSet, _ = torch.utils.data.random_split(myDataset, [int(FLAGS['workingSetSize'] * len(myDataset)), len(myDataset) - int(FLAGS['workingSetSize'] * len(myDataset))], generator=torch.Generator().manual_seed(42)) # discard part of dataset if desired
    
    # TODO implement modulo-based subsets for splits to standardize train/test sets and potentially a future val set for thresholding or wtv
    
    trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset

    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

import timm.layers.ml_decoder as ml_decoder
MLDecoder = ml_decoder.MLDecoder

def add_ml_decoder_head(model):

    # TODO levit, ViT

    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.Identity()
        del model.fc
        num_classes = model.num_classes
        num_features = model.num_features
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    #this is kinda ugly, can make general case?
    elif 'RegNet' in model._get_name() or 'TResNet' in model._get_name():
        del model.head
        num_classes = model.num_classes
        num_features = model.num_features
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

    elif hasattr(model, 'head'):    # ClassifierHead and ConvNext
        if hasattr(model.head, 'flatten'):  # ConvNext case
            model.head.flatten = nn.Identity()
        model.head.global_pool = nn.Identity()
        del model.head.fc
        num_classes = model.num_classes
        num_features = model.num_features
        model.head.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    
    elif 'MobileNetV3' in model._get_name(): # mobilenetv3 - conflict with efficientnet
        
        model.flatten = nn.Identity()
        del model.classifier
        num_classes = model.num_classes
        num_features = model.num_features
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.Identity()
        del model.classifier
        num_classes = model.num_classes
        num_features = model.num_features
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features)

    

    
    

    else:
        print("Model code-writing is not aligned currently with ml-decoder")
        exit(-1)
    if hasattr(model, 'drop_rate'):  # Ml-Decoder has inner dropout
        model.drop_rate = 0
    return model



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
    model = timm.create_model('regnetx_320', pretrained=False, num_classes=len(classes), drop_path_rate=0.4)
    #model = timm.create_model('gernet_s', pretrained=False, num_classes=len(classes), drop_path_rate = 0.)
    #model = timm.create_model('edgenext_small', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('davit_base', pretrained=False, num_classes=len(classes), drop_path_rate = 0.4)
    #model = timm.create_model('davit_base', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('resnet50', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('gernet_l', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('davit_tiny', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    
    # ViT-better similar to https://arxiv.org/abs/2205.01580
    # really only using the avgpool for now, so basically S/32 with gap
    '''
    model = timm.models.VisionTransformer(
        img_size = FLAGS['image_size'], 
        patch_size = 32, 
        num_classes = len(classes), 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        global_pool='avg', 
        class_token = False, 
        fc_norm=True)
    '''
    # cvt
    
    #model = transformers.CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    #model.classifier = nn.Linear(model.config.embed_dim[-1], len(classes))

    # regular huggingface models

    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-384", num_labels=len(classes), ignore_mismatched_sizes=True)
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", num_labels=len(classes), ignore_mismatched_sizes=True)
    
    
    # modified timm models with custom head with hidden layers
    '''
    model = timm.create_model('mixnet_s', pretrained=True, num_classes=-1) # -1 classes for identity head by default
    
    model = nn.Sequential(model,
                          nn.LazyLinear(len(classes)),
                          nn.ReLU(),
                          nn.Linear(len(classes), len(classes)))
    
    '''
    
    #model = add_ml_decoder_head(model)
    

    #model.train()
    
    if FLAGS['finetune'] == True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        if hasattr(model, "head_dist"):
            for param in model.head_dist.parameters():
                param.requires_grad = True
        
    return model
    
def getDataLoader(dataset, batch_size):
    distSampler = DistributedSampler(dataset=dataset, seed=17, drop_last=True)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler=distSampler, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=10, pin_memory = True, generator=torch.Generator().manual_seed(41))

def trainCycle(image_datasets, model):
    #print("starting training")
    startTime = time.time()

    #timm.utils.jit.set_jit_fuser("te")
    
    dataloaders = {x: getDataLoader(image_datasets[x], FLAGS['batch_size']) for x in image_datasets} # set up dataloaders
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
        
    
    is_head_proc = not FLAGS['use_ddp'] or dist.get_rank() == 0
    
    memory_format = torch.channels_last if FLAGS['channels_last'] else torch.contiguous_format
    
    model = model.to(device, memory_format=memory_format)
    
    
        
    if (FLAGS['use_ddp'] == True):
        
        model = DDP(model, device_ids=[FLAGS['device']], gradient_as_bucket_view=True)
        
    if(FLAGS['compile_model'] == True):
        model = torch.compile(model)
        
    if (FLAGS['resume_epoch'] > 0):
        model.load_state_dict(torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth'))

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
    criterion = MLCSL.AsymmetricLossOptimized(gamma_neg=0, gamma_pos=0, clip=0.0, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.AsymmetricLossOptimized(gamma_neg=5, gamma_pos=5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.AsymmetricLossAdaptive(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.01)
    #criterion = MLCSL.AsymmetricLossAdaptiveWorking(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.2)
    #criterion = MLCSL.PartialSelectiveLoss(device, prior_path=None, clip=0.05, gamma_pos=1, gamma_neg=6, gamma_unann=4, alpha_pos=1, alpha_neg=1, alpha_unann=1)
    #parameters = MLCSL.add_weight_decay(model, FLAGS['weight_decay'])
    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = torch_optimizer.Lamb(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = timm.optim.Adan(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = len(dataloaders['train'])*FLAGS['resume_epoch']

    #mixup = Mixup(mixup_alpha = 0.2, cutmix_alpha = 0, num_classes = len(classes))
    
    boundaryCalculator = MLCSL.getDecisionBoundary(initial_threshold = 0.5, lr = 1e-5, threshold_min = 0.001, threshold_max = 0.999)

    if (FLAGS['resume_epoch'] > 0):
        boundaryCalculator.thresholdPerClass = torch.load(FLAGS['modelDir'] + 'thresholds.pth').to(device)
        optimizer.load_state_dict(torch.load(FLAGS['modelDir'] + 'optimizer' + '.pth'))
        
    
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()
    
    # end MLCSL code
    
    losses = []
    best = None
    tagNames = list(classes.values())
    
    
    
    MeanStackedAccuracyStored = torch.Tensor([2,1,2,1])
    if(is_head_proc):
        print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    epoch = FLAGS['resume_epoch']
    
    while (epoch < FLAGS['num_epochs']):
        #prior = MLCSL.ComputePrior(classes, device)
        epochTime = time.time()
        
        
        if(is_head_proc):
            print("starting epoch: " + str(epoch))
        AP_regular = []
        AccuracyRunning = []
        AP_ema = []
        textOutput = None
        #lastPrior = None
        
        phases = ['train', 'val'] if FLAGS['val'] == False else ['val']
        currPhase = 0
        
        while currPhase < len(phases):
            phase = phases[currPhase]
            
            cm_tracker = MLCSL.MetricTracker()

            #try:
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                if(is_head_proc): print("training set")
                
                if FLAGS['progressiveImageSize'] == True:
                    
                    
                    dynamicResizeDim = int(FLAGS['image_size']*FLAGS['progressiveSizeStart'] + epoch * (FLAGS['image_size']-FLAGS['image_size']*FLAGS['progressiveSizeStart'])/FLAGS['num_epochs'])
                else:
                    dynamicResizeDim = FLAGS['actual_image_size']
                
                if(is_head_proc):
                    print(f'Using image size of {dynamicResizeDim}x{dynamicResizeDim}')
                
                myDataset.transform = transforms.Compose([transforms.Resize(dynamicResizeDim),
                                                          transforms.RandAugment(magnitude = epoch, num_magnitude_bins = int(FLAGS['num_epochs'] * FLAGS['progressiveAugRatio'])),
                                                          #transforms.RandAugment(),
                                                          transforms.RandomHorizontalFlip(),
                                                          #transforms.TrivialAugmentWide(),
                                                          #danbooruDataset.CutoutPIL(cutout_factor=0.2),
                                                          
                                                          transforms.ToTensor(),
                                                          RandomErasing(probability=0.5, mode='pixel', device='cpu'),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])
                
                
            if phase == 'val':
                
                
                if FLAGS['val'] == False:
                    modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
                    if(is_head_proc): torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                    if(is_head_proc): torch.save(boundaryCalculator.thresholdPerClass, modelDir + 'thresholds.pth')
                    if(is_head_proc): torch.save(optimizer.state_dict(), modelDir + 'optimizer' + '.pth')
                    if(is_head_proc): pd.DataFrame(tagNames).to_pickle(modelDir + "tags.pkl")
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;
                
                myDataset.transform = transforms.Compose([#transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])
            
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
            for i, (images, tags) in loaderIterable:
                

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    # TODO switch between using autocast and not using it
                    
                    with torch.cuda.amp.autocast(enabled=FLAGS['use_AMP']):
                        
                        
                        outputs = model(imageBatch)
                        #outputs = model(imageBatch).logits
                        preds = torch.sigmoid(outputs)
                        boundary = boundaryCalculator(preds, tagBatch)

                        #predsModified=preds
                        #multiAccuracy = MLCSL.getAccuracy(predsModified.to(device2), tagBatch.to(device2))
                        with torch.no_grad():
                            multiAccuracy = cm_tracker.update((preds.detach() > boundary.detach()).float().to(device), tagBatch.to(device))
                            
                        
                        outputs = outputs.float()
                        '''
                        if phase == 'val':
                            #output_ema = torch.sigmoid(ema.module(imageBatch)).cpu()
                            output_regular = preds.cpu()
                        #loss = criterion(torch.mul(preds, tagBatch), tagBatch)
                        #loss = criterion(outputs, tagBatch)
                        '''

                        #loss = criterion(outputs.to(device2), tagBatch.to(device2), lastPrior)
                        #loss = criterion(outputs.to(device), tagBatch.to(device))
                        loss = criterion(outputs.to(device) - torch.special.logit(boundary.detach()).to(device), tagBatch.to(device))
                        #loss = criterion(outputs.to(device2), tagBatch.to(device2), epoch)
                        #loss, textOutput = criterion(outputs.to(device2), tagBatch.to(device2), updateAdaptive = (phase == 'train'), printAdaptive = (i % stepsPerPrintout == 0))
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
                        #model.zero_grad()
                        # backward + optimize only if in training phase
                        if phase == 'train' and (loss.isnan() == False):
                            if (FLAGS['use_scaler'] == True):   # cuda gpu case
                                with model.no_sync():
                                    scaler.scale(loss).backward()
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad(set_to_none=True)
                            else:                               # apple gpu/cpu case
                                with model.no_sync():
                                    loss.backward()
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                        
                        
                            torch.cuda.synchronize()
                        
                            #ema.update(model)
                            #prior.update(outputs.to(device))
                        
                        if (phase == 'val'):
                            # for mAP calculation
                            targets = tags.numpy(force=True)
                            preds_regular = preds.numpy(force=True)
                            #preds_ema = output_ema.cpu().detach().numpy()
                            accuracy = MLCSL.mAP(targets, preds_regular)
                            AP_regular.append(accuracy)
                            
                            #AP_ema.append(MLCSL.mAP(targets, preds_ema))
                            #AccuracyRunning.append(multiAccuracy)
                
                #print(device)
                if i % stepsPerPrintout == 0:
                    
                    if (phase == 'train'):
                        targets_batch = tags.numpy(force=True)
                        preds_regular_batch = preds.numpy(force=True)
                        accuracy = MLCSL.mAP(targets_batch, preds_regular_batch)
                        
                    

                    imagesPerSecond = (dataloaders[phase].batch_size*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()
                    
                    if(FLAGS['use_ddp'] == True):
                        imagesPerSecond = torch.Tensor([imagesPerSecond]).to(device)
                        torch.distributed.all_reduce(imagesPerSecond, op = torch.distributed.ReduceOp.SUM)
                        imagesPerSecond = imagesPerSecond.cpu()
                        imagesPerSecond = imagesPerSecond.item()
                        

                    #currPostTags = []
                    #batchTagAccuracy = list(zip(tagNames, perTagAccuracy.tolist()))
                    
                    # TODO find better way to generate this output that doesn't involve iterating, zip()?
                    #for tagIndex, tagVal in enumerate(torch.mul(preds, tagBatch)[0]):
                    #    if tagVal.item() != 0:
                    #        currPostTags.append((tagNames[tagIndex], tagVal.item()))
                    
                   
                    #print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tAccuracy: %.2f\tP4: %.2f\t%s' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy, multiAccuracy.mean(dim=0) * 100, textOutput))
                    torch.set_printoptions(linewidth = 200, sci_mode = False)
                    if(is_head_proc): print(f"[{epoch}/{FLAGS['num_epochs']}][{i}/{len(dataloaders[phase])}]\tLoss: {loss:.4f}\tImages/Second: {imagesPerSecond:.4f}\tAccuracy: {accuracy:.2f}\t {[f'{num:.4f}' for num in list((multiAccuracy * 100))]}\t{textOutput}")
                    torch.set_printoptions(profile='default')
                    #print(id[0])
                    #print(currPostTags)
                    #print(sorted(batchTagAccuracy, key = lambda x: x[1], reverse=True))
                    
                    #torch.cuda.empty_cache()
                #losses.append(loss)
                
                if (phase == 'val'):
                    if best is None:
                        best = (float(loss), epoch, i, accuracy.item())
                    elif best[0] > float(loss):
                        best = (float(loss), epoch, i, accuracy.item())
                        print(f"NEW BEST: {best}!")
                
                if phase == 'train':
                    scheduler.step()
                
                #print(device)
                #if(FLAGS['ngpu'] > 0):
                    #torch.cuda.empty_cache()
                    
                    
            if FLAGS['use_ddp'] == True:
                torch.distributed.all_reduce(boundaryCalculator.thresholdPerClass, op = torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(cm_tracker.running_confusion_matrix, op=torch.distributed.ReduceOp.AVG)
            if ((phase == 'val') and (FLAGS['skip_test_set'] == False)):
                #torch.set_printoptions(profile="full")
                
                #AvgAccuracy = torch.stack(AccuracyRunning)
                #AvgAccuracy = AvgAccuracy.mean(dim=0)
                AvgAccuracy = cm_tracker.get_full_metrics()
                LabelledAccuracy = list(zip(AvgAccuracy.tolist(), tagNames, boundaryCalculator.thresholdPerClass))
                LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][8], reverse=True)
                MeanStackedAccuracy = cm_tracker.get_aggregate_metrics()
                MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
                if(is_head_proc): print(*LabelledAccuracySorted, sep="\n")
                #torch.set_printoptions(profile="default")
                if(is_head_proc): print((MeanStackedAccuracy*100).tolist())
                
                
                prior.save_prior()
                prior.get_top_freq_classes()
                lastPrior = prior.avg_pred_train
                if(is_head_proc): print(lastPrior[:30])
                
                mAP_score_regular = np.mean(AP_regular)
                #mAP_score_ema = np.mean(AP_ema)
                if(is_head_proc): print("mAP score regular {:.2f}".format(mAP_score_regular))
                #top_mAP = max(mAP_score_regular, mAP_score_ema)
                if hasattr(criterion, 'tau_per_class'):
                    if(is_head_proc): print(criterion.tau_per_class)
                #print(boundaryCalculator.thresholdPerClass)
            currPhase += 1
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
                        
        
        
        
    
        
        time_elapsed = time.time() - epochTime
        if(is_head_proc): print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        epoch += 1
        

        gc.collect()

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
    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()
