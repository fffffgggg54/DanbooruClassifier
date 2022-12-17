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
import torchvision
import torch_optimizer

import multiprocessing

import timm
import transformers

import timm.models.layers.ml_decoder as ml_decoder
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import FastCollateMixup, Mixup

import handleMultiLabel as MLCSL
import danbooruDataset




# ================================================
#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()

FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/imagenet/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + 'data/'

FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/tf_efficientnetv2_b3/'



# device config


FLAGS['ngpu'] = torch.cuda.is_available()
FLAGS['device'] = torch.device("cuda:0" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
FLAGS['device2'] = FLAGS['device']
if(torch.has_mps == True): FLAGS['device2'] = "cpu"
FLAGS['use_AMP'] = True
FLAGS['use_scaler'] = True
#if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

# dataloader config

FLAGS['num_workers'] = 35


# training config

FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 256
FLAGS['gradient_accumulation_iterations'] = 4

FLAGS['base_learning_rate'] = 3e-3
FLAGS['base_batch_size'] = 2048
FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
FLAGS['lr_warmup_epochs'] = 6

FLAGS['weight_decay'] = 2e-2

FLAGS['resume_epoch'] = 1

FLAGS['finetune'] = False
FLAGS['channels_last'] = True

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['skip_test_set'] = False
FLAGS['stepsPerPrintout'] = 50

classes = None


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


'''
serverProcessPool = []
workQueue = multiprocessing.Queue()
'''
def getData():
    startTime = time.time()

    trainSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'train')
    testSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'val')

    global classes
    classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    
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
    
    #model = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('ghostnet_050', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=len(classes))
    model = timm.create_model('tf_efficientnetv2_b3', pretrained=False, num_classes=1588, drop_rate = 0.00, drop_path_rate = 0.0)
    
    #model = ml_decoder.add_ml_decoder_head(model)
    
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
    
    if (FLAGS['resume_epoch'] > 0):
        model.load_state_dict(torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth'), strict=False)
    
    model.reset_classifier(len(classes))
    
    if FLAGS['finetune'] == True:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        if hasattr(model, "head_dist"):
            for param in model.head_dist.parameters():
                param.requires_grad = True
    
    return model

def trainCycle(image_datasets, model):
    print("starting training")
    startTime = time.time()

    timm.utils.jit.set_jit_fuser("te")
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = False, prefetch_factor=2, pin_memory = True, drop_last=True, generator=torch.Generator().manual_seed(41)) for x in image_datasets} # set up dataloaders
    
    
    #mixup = Mixup(mixup_alpha = 0.1, cutmix_alpha = 0, label_smoothing=0)
    #dataloaders['train'].collate_fn = mixup_collate
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
    device2 = FLAGS['device2']
    
    
    memory_format = torch.channels_last if FLAGS['channels_last'] else torch.contiguous_format
    
    model = model.to(device, memory_format=memory_format)

    print("initialized training, time spent: " + str(time.time() - startTime))
    

    criterion = SoftTargetCrossEntropy()
    #criterion = nn.BCEWithLogitsLoss()

    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = torch_optimizer.Lamb(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = len(dataloaders['train'])*FLAGS['resume_epoch']
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()

    print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['resume_epoch'], FLAGS['num_epochs']):
        epochTime = time.time()
        print("starting epoch: " + str(epoch))

        image_datasets['train'].transform = transforms.Compose([transforms.Resize((256)),
            transforms.RandomHorizontalFlip(),
            RandomResizedCropAndInterpolation(size=224),
            rand_augment_transform(
                config_str='rand-m9-mstd0.5', 
                hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
            ),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            RandomErasing(probability=0.5, mode='pixel', device='cpu'),
            #transforms.GaussianBlur(kernel_size=(7, 7), sigma=(2, 10)),
            #transforms.ToPILImage(),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_datasets['val'].transform = transforms.Compose([transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.RandomCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for phase in ['train', 'val']:
        
            samples = 0
            correct = 0
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
                
                
            if phase == 'val':
                modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
                torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;

            loaderIterable = enumerate(dataloaders[phase])
            for i, (images, tags) in loaderIterable:
                
                

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    with torch.cuda.amp.autocast(enabled=FLAGS['use_AMP']):
                        #if phase == 'train':
                            #imageBatch, tagBatch = mixup(imageBatch, tagBatch)
                        
                        outputs = model(imageBatch)
                        #outputs = model(imageBatch).logits
                        #if phase == 'val':
                        preds = torch.argmax(outputs, dim=1)
                        
                        samples += len(images)
                        correct += sum(preds == tagBatch)
                        tagBatch = torch.eye(len(classes), device=device)[tagBatch]
                        
                        loss = criterion(outputs.to(device2), tagBatch.to(device2))

                        # backward + optimize only if in training phase
                        if phase == 'train' and (loss.isnan() == False):
                            if (FLAGS['use_scaler'] == True):   # cuda gpu case
                                scaler.scale(loss).backward()   #lotta time spent here
                                if(i % FLAGS['gradient_accumulation_iterations'] == 0):
                                    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad()
                            else:                               # apple gpu/cpu case
                                loss.backward()
                                if(i % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    
                if i % stepsPerPrintout == 0:
                    accuracy = 100 * (correct/(samples+1e-8))

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\ttop-1: %.2f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy))
                    torch.cuda.empty_cache()

                if phase == 'train':
                    scheduler.step()
            if phase == 'val':
                print(f'top-1: {100 * (correct/samples)}%')
        
        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        print()



def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files

    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()