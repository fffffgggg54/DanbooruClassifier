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
import os

import torch_optimizer

import multiprocessing

import timm
import transformers

import timm.models.layers.ml_decoder as ml_decoder

import parallelJsonReader
import danbooruDataset
import handleMultiLabel as MLCSL




# ================================================
#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()

FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/danbooru2021/"
#FLAGS['rootPath'] = "/media/fredo/Datasets/danbooru2021/"
if(torch.has_mps == True): FLAGS['rootPath'] = "/Users/fredoguan/Datasets/danbooru2021/"
FLAGS['postMetaRoot'] = FLAGS['rootPath'] #+ "TenthMeta/"
FLAGS['imageRoot'] = "/media/fredo/EXOS_16TB/danbooru2021/original/"
FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
FLAGS['postListFile'] = FLAGS['postMetaRoot'] + "data_posts.json"
FLAGS['tagListFile'] = FLAGS['postMetaRoot'] + "data_tags.json"
FLAGS['postDFPickle'] = FLAGS['postMetaRoot'] + "postData.pkl"
FLAGS['tagDFPickle'] = FLAGS['postMetaRoot'] + "tagData.pkl"
FLAGS['postDFPickleFiltered'] = FLAGS['postMetaRoot'] + "postDataFiltered.pkl"
FLAGS['tagDFPickleFiltered'] = FLAGS['postMetaRoot'] + "tagDataFiltered.pkl"


# post importer config

FLAGS['chunkSize'] = 1000
FLAGS['importerProcessCount'] = 10
if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
FLAGS['stopReadingAt'] = 5000

# dataset config

FLAGS['image_size'] = 384

FLAGS['workingSetSize'] = 1
FLAGS['trainSetSize'] = 0.8


# dataloader config

FLAGS['num_workers'] = 1
FLAGS['postDataServerWorkerCount'] = 1

FLAGS['num_epochs'] = 1
FLAGS['batch_size'] = 512

FLAGS['stepsPerPrintout'] = 250

classes = None
myDataset = None


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
    tagData = pd.read_pickle(FLAGS['tagDFPickleFiltered'])
    postData = pd.read_pickle(FLAGS['postDFPickleFiltered'])
    #print(postData.info())
    
    # get posts that are not banned
    queryStartTime = time.time()
    postData.query("is_banned == False", inplace = True)
    blockedIDs = [5190773, 5142098, 5210705, 5344403, 5237708, 5344394, 5190771, 5237705, 5174387, 5344400, 5344397, 5174384]
    for postID in blockedIDs: postData.query("id != @postID", inplace = True)
    print("banned post query time: " + str(time.time()-queryStartTime))
    

    
    postData = postData[['id', 'tag_string', 'file_ext', 'file_url']]
    #postData = postData[['id', 'tag_string']]
    postData = postData.convert_dtypes()
    print(postData.info())
    

    print("finished preprocessing, time spent: " + str(time.time() - startTime))
    print(f"got {len(postData)} posts with {len(tagData)} tags") #got 3821384 posts with 423 tags
    

    
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
    trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset

    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets


def buildCache(image_datasets):
    #print("starting training")
    startTime = time.time()


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = False, prefetch_factor=3, drop_last=False, generator=torch.Generator().manual_seed(40)) for x in image_datasets} # set up dataloaders
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}



    print("initialized cycle, time spent: " + str(time.time() - startTime))
    



    tagNames = list(classes.values())
    pd.DataFrame(tagNames).to_pickle("tags.pkl")
    

    print("starting cycle")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    
    for epoch in range(FLAGS['num_epochs']):
       
        
        print("starting epoch: " + str(epoch))

        for phase in ['train', 'val']:
            if phase == 'train':
                
                print("training set")
                
                myDataset.transform = transforms.Compose([#transforms.Resize((224,224)),
                                                          #transforms.RandAugment(),
                                                          #transforms.TrivialAugmentWide(),
                                                          #danbooruDataset.CutoutPIL(cutout_factor=0.2),
                                                          transforms.ToTensor(),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])

            if phase == 'val':

                print("validation set")
                myDataset.transform = transforms.Compose([#transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])
            

            loaderIterable = enumerate(dataloaders[phase])
            for i, (images, tags) in loaderIterable:
                

                imageBatch = images
                tagBatch = tags
                
              
                if i % stepsPerPrintout == 0:

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    print('[%d/%d][%d/%d]\tImages/Second: %.4f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), imagesPerSecond))

        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        print()


def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files

    image_datasets = getData()
    buildCache(image_datasets)


if __name__ == '__main__':
    main()