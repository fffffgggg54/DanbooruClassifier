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

import timm.layers.ml_decoder as ml_decoder
MLDecoder = ml_decoder.MLDecoder

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

headers = {"User-Agent": "fffffgggg54 inference test",}

def getPost(postID):
    postURL = "https://danbooru.donmai.us/posts/" + str(postID) + ".json"

    postData = json.loads(requests.get(postURL, headers=headers).content)
    imageURL = postData['file_url']
    
    print("Getting image from " + imageURL)
    response = requests.get(imageURL, headers=headers)
    return Image.open(BytesIO(response.content)), postData

def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files
    
    modelPath = rootPath + "models/regnetz_040h-ASL_BCE_T-224-1588/"
    tagPicklePath = modelPath + "tags.pkl"
    tagNames = pd.read_pickle(tagPicklePath)
    tagNames = tagNames.squeeze('columns').tolist()
    #print(tagNames)
    
    thresholds = None
    thresholdsPath = modelPath + "thresholds.pth"
    
    try:
        thresholds = torch.load(thresholdsPath)
        haveThresholds = True
    except:
        haveThresholds = False
    
    myDevice = 'cpu'
    #model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(tagNames))
    model = timm.create_model('regnetz_040_h', num_classes=len(tagNames))
    #model = add_ml_decoder_head(model)
    #model = cvt.get_cls_model(len(tagNames), config=modelConfCust1)
    #model.load_state_dict(torch.load("models/saved_model_epoch_4.pth", map_location=myDevice))
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-256", num_labels=len(tagNames), ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(modelPath + "saved_model_epoch_91.pth", map_location=myDevice))
    model.eval()   # Set model to evaluate mode
    model = torch.jit.script(model)
    model = torch.jit.optimize_for_inference(model)
    model = model.to(myDevice)
    
    while(True):
        try:
            postID = int(input("Danbooru Post ID:"))
        except:
            print("invalid post ID, exiting...")
            exit()
        
        image, postData = getPost(postID)
        
        startTime = time.time()
        #path = "testImage.jpg"
        #image = Image.open(path)    #check if file exists
        image.load()    # check if file valid
        image = image.convert("RGBA")
            
        color = (255,255,255)
        
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])
        image = background
        

        transform = transforms.Compose([
            transforms.Resize((288,288)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image = transform(image)
        processingTime = time.time() - startTime
        
        startTime = time.time()
        #outputs = model(image.unsqueeze(0)).logits.sigmoid()
        outputs = model(image.unsqueeze(0)).sigmoid()
        predTime = time.time() - startTime
        
        trueTags = set(postData['tag_string'].split(" "))
        
        print(f"preprocessing time: {processingTime} infer time: {predTime}")
        
        currPostTags = []
        #print(outputs.tolist())
        currPostTags = list(zip(tagNames, outputs.tolist()[0]))
        currPostTags.sort(key=lambda y: y[1])
        
        print(*currPostTags, sep="\n")
        
        if haveThresholds:
            tagsThresholded = [(*x, thresholds[i]) for i, x in enumerate(currPostTags) if x[1] > thresholds[i]]
            print("\nTags filtered using threshold:\n")
            print(*tagsThresholded, sep="\n")
            #predTags = {tag[0] for tag in tagsThresholded}
            
            #missingTags = trueTags.difference(predTags)
            #newTags = predTags.difference(trueTags)
            missingTags = [x for x tagsThresholded if trueTags not in set(x[0])]
            newTags = [x for x tagsThresholded if x[0] not in trueTags]
            print(f"missing tags: {missingTags}")
            print(f"newly detected tags: {newTags}")
            
        else:
            print("not using thresholds")



if __name__ == '__main__':
    main()