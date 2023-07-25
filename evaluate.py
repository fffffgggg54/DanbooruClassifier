# run eval on a trained model and save predections and ground truth to files for visualization
# need 2 files because of some weird driver issue between matplotlib.pyplot depending on mesa and not working with nvidia drivers



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
if currGPU == 'v100': FLAGS['rootPath'] = "/media/fredo/SAMSUNG_500GB/danbooru2021/"
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


if currGPU == 'v100':



    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/gc_efficientnetv2_rw_t-448-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convnext_tiny-448-ASL_BCE-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convnext_tiny-448-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/convformer_s18-224-ASL_BCE_T-1588/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/tresnet_m-224-ASL_BCE_T-5500/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_040h-ASL_GP0_GNADAPC_-224-1588-50epoch/'
    FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/regnetz_040h-ASL_BCE_T-F1-x+80e-1-224-1588-50epoch-RawEval/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/vit_base_patch16_224-gap-ASL_BCE_T-P4-x+80e-1-224-1588-300epoch/"
    #FLAGS['modelDir'] = "/media/fredo/Storage3/danbooru_models/caformer_s18-gap-ASL_BCE_T-P4-x+80e-1-224-1588-300epoch/"
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_040h-ASL_GP1_GN5_CL005-224-1588-50epoch/'
    #FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/regnetz_b16-ASL_BCE_-_T-224-1588/'
    
    
    # post importer config

    FLAGS['chunkSize'] = 1000
    FLAGS['importerProcessCount'] = 10
    if(torch.has_mps == True): FLAGS['importerProcessCount'] = 7
    FLAGS['stopReadingAt'] = 5000

    # dataset config
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
    if(torch.has_mps == True): FLAGS['num_workers'] = 2
    if(FLAGS['device'] == 'cpu'): FLAGS['num_workers'] = 2

    # training config

    FLAGS['num_epochs'] = 50
    FLAGS['batch_size'] = 96
    FLAGS['fast_norm'] = True
    


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

def getSubsetByID(dataset, postData, lower, upper, div = 1000):
    return torch.utils.data.Subset(dataset, postData[lower <= postData['id'] % 1000][upper > postData['id'] % 1000].index.tolist())

def getData():
    startTime = time.time()

    if FLAGS['tagCount'] == 1588:
        tagData = pd.read_pickle(FLAGS['tagDFPickleFiltered'])
    elif FLAGS['tagCount'] == 5500:
        tagData = pd.read_csv(FLAGS['rootPath'] + 'selected_tags.csv')
    postData = pd.read_pickle(FLAGS['postDFPickleFilteredTrimmed'])

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
    #trimmedSet, _ = torch.utils.data.random_split(myDataset, [int(FLAGS['workingSetSize'] * len(myDataset)), len(myDataset) - int(FLAGS['workingSetSize'] * len(myDataset))], generator=torch.Generator().manual_seed(42)) # discard part of dataset if desired
    
    # TODO implement modulo-based subsets for splits to standardize train/test sets and potentially a future val set for thresholding or wtv
    
    #trainSet, testSet = torch.utils.data.random_split(trimmedSet, [int(FLAGS['trainSetSize'] * len(trimmedSet)), len(trimmedSet) - int(FLAGS['trainSetSize'] * len(trimmedSet))], generator=torch.Generator().manual_seed(42)) # split dataset
    
    testSet = getSubsetByID(myDataset, postData, 900, 930)
    
    image_datasets = {'val' : testSet}   # put dataset into a list for easy handling
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
    
    #model = timm.create_model('efficientformerv2_s0', pretrained=False, num_classes=len(classes), drop_path_rate=0.05)
    #model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True, num_classes=len(classes), drop_path_rate=0.6)
    #model = timm.create_model('gernet_l', pretrained=False, num_classes=len(classes), drop_path_rate = 0.)
    #model = timm.create_model('edgenext_small', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('davit_base', pretrained=False, num_classes=len(classes), drop_path_rate = 0.4, drop_rate = 0.05)
    #model = timm.create_model('resnet50', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1)
    #model = timm.create_model('caformer_s18', pretrained=False, num_classes=len(classes), drop_path_rate = 0.3)
    model = timm.create_model('regnetz_040_h', pretrained=False, num_classes=len(classes), drop_path_rate=0.15)
    #model = timm.create_model('regnetz_b16', pretrained=False, num_classes=len(classes), drop_path_rate=0.1)
    #model = timm.create_model('ese_vovnet99b_iabn', pretrained=False, num_classes=len(classes), drop_path_rate = 0.1, drop_rate=0.02)
    #model = timm.create_model('tresnet_m', pretrained=False, num_classes=len(classes))
    '''
    model = timm.create_model(
        'vit_base_patch16_224', 
        img_size = FLAGS['actual_image_size'], 
        patch_size = 16, 
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
    '''
    model = timm.models.VisionTransformer(
        img_size = FLAGS['actual_image_size'], 
        patch_size = 16, 
        num_classes = len(classes), 
        embed_dim=1024, 
        depth=16, 
        num_heads=16, 
        global_pool='avg', 
        class_token = False, 
        qkv_bias=False, 
        init_values=1e-6, 
        fc_norm=False)
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
    
def getDataLoader(dataset, batch_size, epoch):
    distSampler = DistributedSampler(dataset=dataset, seed=17, drop_last=True)
    distSampler.set_epoch(epoch)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler=distSampler, num_workers=FLAGS['num_workers'], persistent_workers = True, prefetch_factor=2, pin_memory = True, generator=torch.Generator().manual_seed(41))

def trainCycle(image_datasets, model):
    #print("starting training")
    startTime = time.time()

    #timm.utils.jit.set_jit_fuser("te")
    
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
        
    
    is_head_proc = not FLAGS['use_ddp'] or dist.get_rank() == 0
    
    memory_format = torch.channels_last if FLAGS['channels_last'] else torch.contiguous_format
    
    
    model = model.to(device, memory_format=memory_format)
    state_dict = torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['num_epochs'] - 1) + '.pth', map_location=torch.device('cpu'))
    #out_dict={}
    #for k, v in state_dict.items():
    #    k = k.replace('_orig_mod.', '')
    #    k = k.replace('module.', '')
    #    out_dict[k] = v
        
    model.load_state_dict(state_dict)
    
    
    
    if (FLAGS['use_ddp'] == True):
        
        model = DDP(model, device_ids=[FLAGS['device']], gradient_as_bucket_view=True)
        
    if(FLAGS['compile_model'] == True):
        model = torch.compile(model)
        
    

    print("initialized training, time spent: " + str(time.time() - startTime))
    

    
    boundaryCalculator = MLCSL.getDecisionBoundary(initial_threshold = 0.5, lr = 1e-5, threshold_min = 0.1, threshold_max = 0.9)

    boundaryCalculator.thresholdPerClass = torch.load(FLAGS['modelDir'] + 'thresholds.pth').to(device)
        
    
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()
    
    # end MLCSL code
    
    losses = []
    best = None
    tagNames = list(classes.values())
    
    
    
    if(is_head_proc):
        print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    epoch = FLAGS['num_epochs']
    
    #prior = MLCSL.ComputePrior(classes, device)
    epochTime = time.time()
    
    dataloaders = {x: getDataLoader(image_datasets[x], FLAGS['batch_size'], epoch) for x in image_datasets} # set up dataloaders

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = len(dataloaders['train'])*epoch

    
    if(is_head_proc):
        print("starting epoch: " + str(epoch))
    AP_regular = []
    AccuracyRunning = []
    AP_ema = []
    targets_running = None
    preds_running = None
    targets_running = []
    preds_running = []
    textOutput = None
    #lastPrior = None
    
    phases = ['val']
    currPhase = 0
    
    while currPhase < len(phases):
        phase = phases[currPhase]
        
        cm_tracker = MLCSL.MetricTracker()
  
            
        if phase == 'val':
            
            
            if FLAGS['val'] == False and is_head_proc:
                modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
                state_dict = model.state_dict()
                
                out_dict={}
                for k, v in state_dict.items():
                    k = k.replace('_orig_mod.', '')
                    k = k.replace('module.', '')
                    out_dict[k] = v
                
                torch.save(out_dict, modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                torch.save(boundaryCalculator.thresholdPerClass, modelDir + 'thresholds.pth')
                torch.save(optimizer.state_dict(), modelDir + 'optimizer' + '.pth')
                pd.DataFrame(tagNames).to_pickle(modelDir + "tags.pkl")
            
            
            model.eval()   # Set model to evaluate mode
            print("validation set")
            if(FLAGS['skip_test_set'] == True):
                print("skipping...")
                break;
            
            myDataset.transform = transforms.Compose([#transforms.Resize((224,224)),
                                                      transforms.ToTensor(),
                                                      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                      ])
        

        
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
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        boundary = boundaryCalculator(preds, tagBatch)
                        if FLAGS['use_ddp'] == True:
                            torch.distributed.all_reduce(boundaryCalculator.thresholdPerClass, op = torch.distributed.ReduceOp.AVG)
                            boundary = boundaryCalculator.thresholdPerClass.detach()

                    #predsModified=preds
                    #multiAccuracy = MLCSL.getAccuracy(predsModified.to(device2), tagBatch.to(device2))
                    with torch.no_grad():
                        #multiAccuracy = cm_tracker.update((preds.detach() > boundary.detach()).float().to(device), tagBatch.to(device))
                        multiAccuracy = cm_tracker.update(preds.detach().float().to(device), tagBatch.to(device))
                    
                    outputs = outputs.float()

                    if (phase == 'val'):
                        # for mAP calculation
                        # FIXME this is super slow and bottlenecked, figure out a faster way to do validation with correctly calculated metrics
                        if(FLAGS['use_ddp'] == True):
                            targets_all = None
                            preds_all = None
                            if(is_head_proc):
                                targets_all = [torch.zeros_like(tagBatch) for _ in range(dist.get_world_size())]
                                preds_all = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
                            torch.distributed.gather(tagBatch, gather_list = targets_all, async_op=True)
                            torch.distributed.gather(preds, gather_list = preds_all, async_op=True)
                            if(is_head_proc):
                                targets_all = torch.cat(targets_all).detach().cpu()
                                preds_all = torch.cat(preds_all).detach().cpu()
                        else:
                            targets_all = tags
                            preds_all = preds.detach().cpu()
                        
                        if is_head_proc:
                            targets = targets_all.numpy(force=True)
                            preds_regular = preds_all.numpy(force=True)
                            #preds_ema = output_ema.cpu().detach().numpy()
                            accuracy = MLCSL.mAP(targets, preds_regular)
                            #AP_regular.append(accuracy)
                            
                            
                            
                            targets_running.append(targets_all.detach().clone())
                            preds_running.append(preds_all.detach().clone())
                            
                            #AP_ema.append(MLCSL.mAP(targets, preds_ema))
                            #AccuracyRunning.append(multiAccuracy)
                            targets_all = None
                            preds_all = None
            
            #print(device)
            if i % stepsPerPrintout == 0:
                


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
                if(is_head_proc): print(f"[{epoch}/{FLAGS['num_epochs']}][{i}/{len(dataloaders[phase])}]\tImages/Second: {imagesPerSecond:.4f}\tAccuracy: {accuracy:.2f}\t {[f'{num:.4f}' for num in list((multiAccuracy * 100))]}\t{textOutput}")
                torch.set_printoptions(profile='default')
                #print(id[0])
                #print(currPostTags)
                #print(sorted(batchTagAccuracy, key = lambda x: x[1], reverse=True))
                
                #torch.cuda.empty_cache()
            #losses.append(loss)

                
                
        if FLAGS['use_ddp'] == True:
            torch.distributed.all_reduce(cm_tracker.running_confusion_matrix, op=torch.distributed.ReduceOp.AVG)
            
            #torch.distributed.all_reduce(criterion.gamma_neg_per_class, op = torch.distributed.ReduceOp.AVG)
        if ((phase == 'val') and (FLAGS['skip_test_set'] == False) and is_head_proc):
            #torch.set_printoptions(profile="full")
            
            #AvgAccuracy = torch.stack(AccuracyRunning)
            #AvgAccuracy = AvgAccuracy.mean(dim=0)
            AvgAccuracy = cm_tracker.get_full_metrics()
            LabelledAccuracy = list(zip(AvgAccuracy.tolist(), tagNames, boundaryCalculator.thresholdPerClass.data))
            LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][8], reverse=True)
            MeanStackedAccuracy = cm_tracker.get_aggregate_metrics()
            MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
            if(is_head_proc): print(*LabelledAccuracySorted, sep="\n")
            #torch.set_printoptions(profile="default")
            if(is_head_proc): print((MeanStackedAccuracy*100).tolist())
            
            
            #prior.save_prior()
            #prior.get_top_freq_classes()
            #lastPrior = prior.avg_pred_train
            #if(is_head_proc): print(lastPrior[:30])
            
            mAP_score_regular = MLCSL.mAP(torch.cat(targets_running).numpy(force=True), torch.cat(preds_running).numpy(force=True))
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
        FLAGS['learning_rate'] *= dist.get_world_size()
    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()
