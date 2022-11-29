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
FLAGS['imageRoot'] = FLAGS['rootPath'] + "original/"
FLAGS['cacheRoot'] = FLAGS['rootPath'] + "cache/"
FLAGS['postListFile'] = FLAGS['postMetaRoot'] + "data_posts.json"
FLAGS['tagListFile'] = FLAGS['postMetaRoot'] + "data_tags.json"
FLAGS['postDFPickle'] = FLAGS['postMetaRoot'] + "postData.pkl"
FLAGS['tagDFPickle'] = FLAGS['postMetaRoot'] + "tagData.pkl"
FLAGS['postDFPickleFiltered'] = FLAGS['postMetaRoot'] + "postDataFiltered.pkl"
FLAGS['tagDFPickleFiltered'] = FLAGS['postMetaRoot'] + "tagDataFiltered.pkl"

FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/mixnet_s-linHead_NCH-1588-Hill/'


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

FLAGS['num_workers'] = 18
FLAGS['postDataServerWorkerCount'] = 2
if(torch.has_mps == True): FLAGS['num_workers'] = 2
if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

# training config

FLAGS['num_epochs'] = 60
FLAGS['batch_size'] = 128
FLAGS['gradient_accumulation_iterations'] = 1

FLAGS['learning_rate'] = 3e-4
FLAGS['lr_warmup_epochs'] = 5

FLAGS['weight_decay'] = 1e-2

FLAGS['resume_epoch'] = 0

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['stepsPerPrintout'] = 250

classes = None
myDataset = None


serverProcessPool = []
workQueue = multiprocessing.Queue()

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
        
        
    # TODO custom normalization values that fit the dataset better
    # TODO investigate ways to return full size images instead of crops
    # this should allow use of full sized images that vary in size, which can then be fed into a model that takes images of arbitrary precision
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
    myDataset= danbooruDataset.DanbooruDatasetWithServer(workQueue,
                                                         len(postData),
                                                         None)
    global classes
    classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
    
    #classes = {classIndex : className for classIndex, className in enumerate(tagData.name)}
    trimmedSet, _ = torch.utils.data.random_split(myDataset, [int(FLAGS['workingSetSize'] * len(myDataset)), len(myDataset) - int(FLAGS['workingSetSize'] * len(myDataset))], generator=torch.Generator().manual_seed(42)) # discard part of dataset if desired
    trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset

    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

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
    
    #model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('ghostnet_050', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('regnetx_002', pretrained=True, num_classes=len(classes))
    
    #model = ml_decoder.add_ml_decoder_head(model)
    
    # cvt
    
    #model = transformers.CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    #model.classifier = nn.Linear(model.config.embed_dim[-1], len(classes))

    # regular huggingface models

    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-128S", num_labels=len(classes), ignore_mismatched_sizes=True)
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", num_labels=len(classes), ignore_mismatched_sizes=True)
    
    
    # modified timm models with custom head with hidden layers
    
    model = timm.create_model('mixnet_s', pretrained=True, num_classes=-1) # -1 classes for identity head by default
    
    model = nn.Sequential(model,
                          nn.LazyLinear(len(classes)),
                          nn.ReLU(),
                          nn.Linear(len(classes), len(classes)))
    
    
    
    if (FLAGS['resume_epoch'] > 0):
        model.load_state_dict(torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth'))
    #model.train()

    return model

def trainCycle(image_datasets, model):
    #print("starting training")
    startTime = time.time()

    
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = False, prefetch_factor=3, pin_memory = True, drop_last=True, generator=torch.Generator().manual_seed(42)) for x in image_datasets} # set up dataloaders
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
    
    
    criterion = MLCSL.Hill()
    #criterion = MLCSL.AsymmetricLossOptimized(gamma_neg=5, gamma_pos=5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False)
    #criterion = MLCSL.AsymmetricLossAdaptive(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.01)
    #criterion = MLCSL.AsymmetricLossAdaptiveWorking(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.2)
    #criterion = MLCSL.PartialSelectiveLoss(device, prior_path=None, clip=0.05, gamma_pos=1, gamma_neg=6, gamma_unann=4, alpha_pos=1, alpha_neg=1, alpha_unann=1)
    #parameters = MLCSL.add_weight_decay(model, FLAGS['weight_decay'])
    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = len(dataloaders['train'])*FLAGS['resume_epoch']
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()
    
    # end MLCSL code
    
    losses = []
    best = None
    tagNames = list(classes.values())
    pd.DataFrame(tagNames).to_pickle("tags.pkl")
    
    
    MeanStackedAccuracyStored = torch.Tensor([2,1,2,1])
    
    print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['resume_epoch'], FLAGS['num_epochs']):
        prior = MLCSL.ComputePrior(classes, device2)
        epochTime = time.time()
        
        
        print("starting epoch: " + str(epoch))
        AP_regular = []
        AccuracyRunning = []
        AP_ema = []
        textOutput = None
        #lastPrior = None
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                print("training set")
                
                myDataset.transform = transforms.Compose([#transforms.Resize((224,224)),
                                                          danbooruDataset.CutoutPIL(cutout_factor=0.5),
                                                          transforms.RandAugment(),
                                                          transforms.ToTensor(),
                                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                          ])
                
                
            else:
                
                

                modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
                torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                
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
            for i, (images, tags, id) in loaderIterable:
                

                imageBatch = images.to(device, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    # TODO switch between using autocast and not using it
                    
                    #with torch.cuda.amp.autocast():
                    outputs = model(imageBatch)
                    #outputs = model(imageBatch).logits
                    multiAccuracy = MLCSL.getAccuracy(outputs.to(device2), tagBatch.to(device2))
                    referenceTable = MLCSL.getAccuracy(tagBatch.to(device2), tagBatch.to(device2))
                    preds = torch.sigmoid(outputs)
                    outputs = outputs.float()
                    
                    if phase == 'val':
                        #output_ema = torch.sigmoid(ema.module(imageBatch)).cpu()
                        output_regular = preds.cpu()
                    #loss = criterion(torch.mul(preds, tagBatch), tagBatch)
                    #loss = criterion(outputs, tagBatch)
                    

                    #loss = criterion(outputs.to(device2), tagBatch.to(device2), lastPrior)
                    loss = criterion(outputs.to(device2), tagBatch.to(device2))
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
                    # TODO this is slow, profile and optimize
                    if phase == 'train' and (loss.isnan() == False):
                        if (FLAGS['use_scaler'] == True):   # cuda gpu case
                            scaler.scale(loss).backward()   #lotta time spent here
                            if(i % FLAGS['gradient_accumulation_iterations'] == 0):
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:                               # apple gpu/cpu case
                            loss.backward()
                            if(i % FLAGS['gradient_accumulation_iterations'] == 0):
                                optimizer.step()
                                optimizer.zero_grad()

                        #ema.update(model)
                        prior.update(outputs.to(device2))
                    
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
                    
                   
                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\tAccuracy: %.2f\tP4: %.2f\t%s' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy, multiAccuracy[:,8].mean() * 100, textOutput))
                    #print(id[0])
                    #print(currPostTags)
                    #print(sorted(batchTagAccuracy, key = lambda x: x[1], reverse=True))
                    
                    torch.cuda.empty_cache()
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
        
        #torch.set_printoptions(profile="full")
        
        AvgAccuracy = torch.stack(AccuracyRunning)
        AvgAccuracy = AvgAccuracy.mean(dim=0)
        LabelledAccuracy = list(zip(AvgAccuracy.tolist(), tagNames))
        LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][6], reverse=True)
        MeanStackedAccuracy = AvgAccuracy.mean(dim=0)
        MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
        print(*LabelledAccuracySorted, sep="\n")
        #torch.set_printoptions(profile="default")
        print(MeanStackedAccuracy)
        
        
        prior.save_prior()
        prior.get_top_freq_classes()
        lastPrior = prior.avg_pred_train
        print(lastPrior[:30])
        
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