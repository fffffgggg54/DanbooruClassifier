import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.parallel
import torch.profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torchvision import models, transforms
from torchvision.io import read_image
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import time
import glob
import gc
import os
import torchvision
from sklearn import preprocessing

import multiprocessing

import timm
import transformers
import datasets

import timm.layers.ml_decoder as ml_decoder
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossMultiLabel
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import FastCollateMixup, Mixup




def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

# ================================================
#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()

FLAGS['rootPath'] = "/mnt/disks/persist/imagenet/"
FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/'
FLAGS['tagsPath'] = './selected_tags.csv'


# dataloader config

FLAGS['num_workers'] = 5
FLAGS['imageSize'] = 448

FLAGS['interpolation'] = torchvision.transforms.InterpolationMode.BICUBIC
FLAGS['crop'] = 0.900
FLAGS['image_size_initial'] = int(FLAGS['imageSize'] // FLAGS['crop'])

# training config

FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 8
FLAGS['gradient_accumulation_iterations'] = 1

FLAGS['base_learning_rate'] = 3e-3
FLAGS['base_batch_size'] = 2048
FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
FLAGS['lr_warmup_epochs'] = 5

FLAGS['weight_decay'] = 2e-2

FLAGS['resume_epoch'] = 0

FLAGS['finetune'] = False

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['skip_test_set'] = False
FLAGS['stepsPerPrintout'] = 50

classes = None


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
        if hasattr(model.head, 'norm'):
            model.head.norm = nn.Identity()
        if hasattr(model, 'norm_pre'):
            model.norm_pre = nn.Identity()
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


trainTransforms = transforms.Compose([transforms.Resize((FLAGS['image_size'],FLAGS['image_size'])),
    transforms.RandAugment(),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    #timm.data.random_erasing.RandomErasing(probability=1, mode='pixel', device='cpu'),
    transforms.ToTensor(),
    #transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

valTransforms = transforms.Compose([
    transforms.Resize((FLAGS['image_size_initial'],FLAGS['image_size_initial']), interpolation = FLAGS['interpolation']),
    transforms.CenterCrop((int(FLAGS['image_size']),int(FLAGS['image_size']))),
    transforms.ToTensor(),
    #transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])





class transformsCallable():
    def __init__(self, tagList, transform=None):
        self.transform = transform
        
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit([tagList.to_list()])

    def __call__(self, examples):
        image = Image.open(BytesIO(examples['Image']['bytes']))
        image.load()    # check if file valid
        image = image.convert("RGBA")
            
        color = (255,255,255)

        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])
        image = background
        examples["Image"] = image
        if self.transform is not None:
            examples["Image"] = self.transform(examples["Image"])

        postTagList = set(examples['tag_string'].split()).intersection(set(tagList.to_list()))

        postTags = lb.transform([postTagList])
        examples['labels'] = torch.Tensor(postTags)

        return examples
        
        
def getData():
    startTime = time.time()
    tags = pd.read_csv(FLAGS['tagsPath'])
    tagList = tags['name']
    myDataset = datasets.load_dataset('fffffgggg54/danbooru2021', streaming=True)

    moduloVal = 10
    moduloBound = 9
    train_ds = myDataset['train'] \
        .with_format("torch") \
        .filter(lambda x: (x['__index_level_0__'] % 10) < moduloBound) \
        .map(transformsCallable(tagList, trainTransforms)) \
        .shuffle(buffer_size=1000, seed=42)

    val_ds = myDataset['train'] \
        .with_format("torch") \
        .filter(lambda x: (x['__index_level_0__'] % 10) >= moduloBound) \
        .map(transformsCallable(tagList valTransforms)) \
        .shuffle(buffer_size=1000, seed=42)
        
    global classes
    #classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    #classes = {classIndex : className for classIndex, className in enumerate(range(1000))}
    classes = {classIndex : className for classIndex, className in enumerate(tagList)}
    
    image_datasets = {'train': train_ds, 'val' : val_ds}   # put dataset into a list for easy handling
    return image_datasets

