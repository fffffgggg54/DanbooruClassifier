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


import timm

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
FLAGS['imageRoot'] = FLAGS['rootPath'] + "original/"
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

FLAGS['workingSetSize'] = 1
FLAGS['trainSetSize'] = 0.8

# device config


FLAGS['ngpu'] = torch.cuda.is_available()
FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
FLAGS['device2'] = FLAGS['device']
if(torch.has_mps == True): FLAGS['device2'] = "cpu"
FLAGS['use_scaler'] = False
#if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

# dataloader config

FLAGS['batch_size'] = 384
FLAGS['num_workers'] = 7
if(torch.has_mps == True): FLAGS['num_workers'] = 2
if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

# training config

FLAGS['num_epochs'] = 20
FLAGS['learning_rate'] = 3e-2
FLAGS['weight_decay'] = 1e-2

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['stepsPerPrintout'] = 50

classes = None


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
    # TODO custom normalization values that fit the dataset better
    # TODO investigate ways to return full size images instead of crops
    # this should allow use of full sized images that vary in size, which can then be fed into a model that takes images of arbitrary precision
    myDataset = danbooruDataset.DanbooruDataset(FLAGS['imageRoot'], postData, tagData.name, transforms.Compose([
        #transforms.Resize((224,224)),
        danbooruDataset.CutoutPIL(cutout_factor=0.5),
        transforms.RandAugment(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        cacheRoot = FLAGS['cacheRoot']
        )
    global classes
    classes = myDataset.classes
    
    #classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
    trimmedSet, _ = torch.utils.data.random_split(myDataset, [int(FLAGS['workingSetSize'] * len(myDataset)), len(myDataset) - int(FLAGS['workingSetSize'] * len(myDataset))], generator=torch.Generator().manual_seed(42)) # discard part of dataset if desired
    trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset
    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

def modelSetup(classes):

    
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    #model = models.resnet152()
    #model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    #model = models.resnet34()
    #model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    
    
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    #model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=len(classes))


    return model

def trainCycle(image_datasets, model):
    #print("starting training")
    startTime = time.time()

    
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=5, pin_memory = True, drop_last=True, generator=torch.Generator().manual_seed(42)) for x in image_datasets} # set up dataloaders
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
    device2 = FLAGS['device2']
        
    
    
    model = model.to(device)

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
    #prior = MLCSL.ComputePrior(classes, device2)
    
    
    criterion = MLCSL.AsymmetricLossOptimized(gamma_neg=8, gamma_pos=4, clip=0.1, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.PartialSelectiveLoss(device, prior_path=None, clip=0, gamma_pos=2, gamma_neg=10, gamma_unann=10, alpha_pos=1, alpha_neg=1, alpha_unann=1)
    parameters = MLCSL.add_weight_decay(model, FLAGS['weight_decay'])
    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=0)
    optimizer = optim.SGD(params=parameters, lr=FLAGS['learning_rate'], weight_decay=0)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=0.2)
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()
    
    # end MLCSL code
    
    losses = []
    best = None
    tagNames = list(classes.values())
    pd.DataFrame(tagNames).to_pickle("tags.pkl")
    

    print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['num_epochs']):
        epochTime = time.time()
        
        
        print("starting epoch: " + str(epoch))
        AP_regular = []
        AccuracyRunning = []
        AP_ema = []
        #lastPrior = None
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                print("training set")
            else:
                model.eval()   # Set model to evaluate mode
                

                modelDir = danbooruDataset.create_dir(FLAGS['rootPath'] + 'models/')
                torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                print("validation set")
            
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
            for i, (images, tags, id) in loaderIterable:
                

                imageBatch = images.to(device, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                model.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # TODO switch between using autocast and not using it
                    
                    #with torch.cuda.amp.autocast():
                    outputs = model(imageBatch)
                    multiAccuracy = MLCSL.getAccuracy(outputs.to(device2), tagBatch.to(device2))
                    preds = torch.sigmoid(outputs)
                    outputs = outputs.float()
                    
                    if phase == 'val':
                        #output_ema = torch.sigmoid(ema.module(imageBatch)).cpu()
                        output_regular = preds.cpu()
                    #loss = criterion(torch.mul(preds, tagBatch), tagBatch)
                    #loss = criterion(outputs, tagBatch)
                    

                    #loss = criterion(outputs.to(device2), tagBatch.to(device2), lastPrior)
                    #loss = criterion(outputs.to(device2), tagBatch.to(device2))
                    #loss = criterion(outputs.cpu(), tags.cpu())
                    #loss = (1 - multiAccuracy[:,4:]).pow(2).mul(torch.Tensor([1,1,3,1]).to(device2)).sum()
                    loss = (1 - multiAccuracy[:,4:]).pow(2).sum()
                    #loss = (multiAccuracy[:,1] + multiAccuracy[:,2]).pow(2).sum()
                    #model.zero_grad()
                    # backward + optimize only if in training phase
                    # TODO this is slow, profile and optimize
                    if phase == 'train':
                        if (FLAGS['use_scaler'] == True):   # cuda gpu case
                            scaler.scale(loss).backward()   #lotta time spent here
                            scaler.step(optimizer)
                            scaler.update()
                        else:                               # apple gpu/cpu case
                            loss.backward()
                            optimizer.step()

                        #ema.update(model)
                        #prior.update(outputs.to(device2))
                    
                    if (phase == 'val'):
                        
                        # for mAP calculation
                        targets = tags.cpu().detach().numpy()
                        preds_regular = output_regular.cpu().detach().numpy()
                        #preds_ema = output_ema.cpu().detach().numpy()
                        accuracy = MLCSL.mAP(targets, preds_regular)
                        AP_regular.append(accuracy)
                        
                        #AP_ema.append(MLCSL.mAP(targets, preds_ema))
                        AccuracyRunning.append(multiAccuracy)
                #print(device)
                if i % stepsPerPrintout == 0:
                    
                    if (phase == 'train'):
                        
                        targets_batch = tags.cpu().detach().numpy()
                        preds_regular_batch = preds.cpu().detach().numpy()
                        accuracy = MLCSL.mAP(targets_batch, preds_regular_batch)
                        
                    

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    #currPostTags = []
                    #batchTagAccuracy = list(zip(tagNames, perTagAccuracy.tolist()))
                    
                    # TODO find better way to generate this output that doesn't involve iterating, zip()?
                    #for tagIndex, tagVal in enumerate(torch.mul(preds, tagBatch)[0]):
                    #    if tagVal.item() != 0:
                    #        currPostTags.append((tagNames[tagIndex], tagVal.item()))
                    
                   
                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tAccuracy: %.2f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy))
                    #print(id[0])
                    #print(currPostTags)
                    #print(sorted(batchTagAccuracy, key = lambda x: x[1], reverse=True))
                    
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
                    
        
        #torch.set_printoptions(profile="full")
        
        AvgAccuracy = torch.stack(AccuracyRunning)
        AvgAccuracy = AvgAccuracy.mean(dim=0)
        LabelledAccuracy = list(zip(AvgAccuracy.tolist(), tagNames))
        LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][6], reverse=True)
        
        print(*LabelledAccuracySorted, sep="\n")
        #torch.set_printoptions(profile="default")
        
        
        #prior.save_prior()
        #prior.get_top_freq_classes()
        #lastPrior = prior.avg_pred_train
        #print(lastPrior[:30])
        
        mAP_score_regular = np.mean(AP_regular)
        #mAP_score_ema = np.mean(AP_ema)
        print("mAP score regular {:.2f}".format(mAP_score_regular))
        #top_mAP = max(mAP_score_regular, mAP_score_ema)
        
        
        
                        
        
        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        print()
    #time_elapsed = time.time() - startTime
    #print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files


    
    image_datasets = getData()
    
    model = modelSetup(classes)
    

    trainCycle(image_datasets, model)

    




if __name__ == '__main__':
    main()