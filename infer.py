# put -r "C:\Users\fredo\OneDrive - Lake Washington School District\Code\ML\danbooru2021\classification\" /home/fredo/Code/ML/danbooru2021/classification
# python "/home/fredo/Code/ML/danbooru2021/classification/infer.py"


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
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import time
import glob
import gc
from PIL import Image, ImageOps
import PIL
import random

import timm
import transformers

import requests
from io import BytesIO
import json

import parallelJsonReader
import danbooruDataset
import handleMultiLabel as MLCSL

#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables
# paths
# TODO use os join or whatever instead of string concat when joining paths

# root directory of danbooru dataset
#rootPath = "D:/Datasets/danbooru2021/"
#rootPath = "C:/Users/Fredo/Downloads/Datasets/danbooru2021/"
rootPath = "/media/fredo/KIOXIA/Datasets/danbooru2021/"
#rootPath = "/media/fredo/Datasets/danbooru2021/"
if(torch.has_mps == True): rootPath = "/Users/fredoguan/Datasets/danbooru2021/"
#cacheRoot = "G:/DanbooruCache/"
#postMetaDir = rootPath + "metadata/"
postMetaDir = rootPath
imageRoot = rootPath + "original/"
# file names
tagListFile = "data_tags.json"
postListFile = "data_posts.json"
postDFPickle = postMetaDir + "postData.pkl"
tagDFPickle = postMetaDir + "tagData.pkl"
postDFPickleFiltered = postMetaDir + "postDataFiltered.pkl"
tagDFPickleFiltered = postMetaDir + "tagDataFiltered.pkl"


# Number of GPUs available. Use 0 for CPU mode.
ngpu = torch.cuda.is_available()
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "mps" if (torch.has_mps == True) else "cpu")
device2 = device
if(torch.has_mps == True): device2 = "cpu"
lr = 0.0005


def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files
    startTime = time.time()
    '''
    tagData = pd.read_pickle(rootPath + "tagData.pkl")
    postData = pd.read_pickle(rootPath + "postData.pkl")
    
    
    
    
    try:
        print("attempting to read pickled post metadata file at " + tagDFPickle)
        tagData = pd.read_pickle(tagDFPickle)
    except:
        print("pickled post metadata file at " + tagDFPickle + " not found")
        tagData = parallelJsonReader.dataImporter(postMetaDir + tagListFile)   # read tags from json in parallel
        print("saving pickled post metadata to " + tagDFPickle)
        tagData.to_pickle(tagDFPickle)
    
    #postData = pd.concat(map(dataImporter, glob.iglob(postMetaDir + 'posts*')), ignore_index=True) # read all post metadata files in metadata dir
    try:
        print("attempting to read pickled post metadata file at " + postDFPickle)
        postData = pd.read_pickle(postDFPickle)
    except:
        print("pickled post metadata file at " + postDFPickle + " not found")
        postData = parallelJsonReader.dataImporter(postMetaDir + postListFile, keep = 0.1)    # read posts
        print("saving pickled post metadata to " + postDFPickle)
        postData.to_pickle(postDFPickle)
        
        
    
    print("got posts, time spent: " + str(time.time() - startTime))
    
    # filter data
    startTime = time.time()
    print("applying filters")
    # TODO this filter process is slow, need to speed it up, currently only single threaded
    tagData, postData = danbooruDataset.filterDanbooruData(tagData, postData)   # apply various filters to preprocess data
    
    #tagData.to_pickle(tagDFPickleFiltered)
    #postData.to_pickle(postDFPickleFiltered)
    
    #tagData = pd.read_pickle(tagDFPickleFiltered)
    #postData = pd.read_pickle(postDFPickleFiltered)
    #print(postData.info())
    
    print("finished preprocessing, time spent: " + str(time.time() - startTime))
    print(f"got {len(postData)} posts with {len(tagData)} tags") #got 3821384 posts with 423 tags
    
    startTime = time.time()
    
    
    
    classes = {classIndex : className for classIndex, className in enumerate(set(tagData.name))}
    
    
    '''
    startTime = time.time()
    tagNames = pd.read_pickle("/home/fredo/tags.pkl")
    tagNames = tagNames.squeeze('columns').tolist()
    #print(tagNames)
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    #model = models.resnet152()
    #model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.resnet34()
    #model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    #model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    #model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    #model = TResnetM({'num_classes':len(tagNames)})
    #model.load_state_dict(torch.load("/home/fredo/Code/ML/danbooru2021/tresnet_m.pth"), strict=False)
    #model = MLDecoderHead.add_ml_decoder_head(model, num_of_groups=int(len(tagNames)/48))
    
    #model.apply(weights_init)
    
    
    
    modelConfCust1 = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 4,
        'PATCH_SIZE': [7, 5, 3, 3],
        'PATCH_STRIDE': [4, 3, 2, 2],
        'PATCH_PADDING': [2, 2, 1, 1],
        'DIM_EMBED': [64, 240, 384, 896],
        'NUM_HEADS': [1, 3, 6, 14],
        'DEPTH': [1, 4, 8, 16],
        'MLP_RATIO': [4.0, 4.0, 4.0, 4.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0, 0.0],
        'DROP_RATE': [0.0, 0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.0, 0.1],
        'QKV_BIAS': [True, True, True, True],
        'CLS_TOKEN': [False, False, False, True],
        'POS_EMBED': [False, False, False, False],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3, 3],
        'PADDING_KV': [1, 1, 1, 1],
        'STRIDE_KV': [2, 2, 2, 2],
        'PADDING_Q': [1, 1, 1, 1],
        'STRIDE_Q': [1, 1, 1, 1]
        }
    
    
    
    modelConf21 = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [64, 192, 384],
        'NUM_HEADS': [1, 3, 6],
        'DEPTH': [1, 4, 16],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'QKV_BIAS': [True, True, True],
        'CLS_TOKEN': [False, False, True],
        'POS_EMBED': [False, False, False],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [2, 2, 2],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1]
        }
        
        
        
    modelConf13 = {
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [64, 192, 384],
        'NUM_HEADS': [1, 3, 6],
        'DEPTH': [1, 2, 10],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'QKV_BIAS': [True, True, True],
        'CLS_TOKEN': [False, False, True],
        'POS_EMBED': [False, False, False],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [2, 2, 2],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1]
        }
         
    myDevice = 'cpu'
    #model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(tagNames))
    #model = cvt.get_cls_model(len(tagNames), config=modelConfCust1)
    #model.load_state_dict(torch.load("models/saved_model_epoch_4.pth", map_location=myDevice))
    model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-256", num_labels=len(tagNames), ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(rootPath + "models/levit_256_1588_Hill/saved_model_epoch_14.pth", map_location=myDevice))
    model.eval()   # Set model to evaluate mode
    model = model.to(myDevice)
    
    
    postID = int(input("Danbooru Post ID:"))
    postURL = "https://danbooru.donmai.us/posts/" + str(postID) + ".json"

    postData = json.loads(requests.get(postURL).content)
    imageURL = postData['file_url']
    
    print("Getting image from " + imageURL)
    response = requests.get(imageURL)
    image = Image.open(BytesIO(response.content))
    
    
    #path = "testImage.jpg"
    #image = Image.open(path)    #check if file exists
    image.load()    # check if file valid
    image = image.convert("RGBA")
        
    color = (255,255,255)
    
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])
    image = background
    
    '''
    size = image.size
        
            
    targetMul = 56
    xPad = targetMul - (size[0] % targetMul)
    yPad = targetMul - (size[1] % targetMul)
    xPadL = int(xPad * random.random())
    xPadR = xPad - xPadL
    yPadT = int(yPad * random.random())
    yPadB = yPad - yPadT
    targetPad = (xPadL, yPadT, xPadR, yPadB)
    
    image = transforms.Pad(targetPad).forward(image)
    '''
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = transform(image)
    
    
    
    outputs = model(image.unsqueeze(0)).logits.sigmoid()
    
    
    currPostTags = []
    #print(outputs.tolist())
    currPostTags = list(zip(tagNames, outputs.tolist()[0]))
    currPostTags.sort(key=lambda y: y[1])
    
    print(currPostTags)
    



if __name__ == '__main__':
    main()