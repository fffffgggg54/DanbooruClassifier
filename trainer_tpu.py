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

from io import BytesIO
from PIL import Image, ImageOps, ImageDraw

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

from accelerate import Accelerator


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
FLAGS['imageSize'] = 224

FLAGS['interpolation'] = torchvision.transforms.InterpolationMode.BICUBIC
FLAGS['crop'] = 0.900
FLAGS['image_size_initial'] = int(FLAGS['imageSize'] // FLAGS['crop'])

# training config

FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 16
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


trainTransforms = transforms.Compose([transforms.Resize((FLAGS['imageSize'],FLAGS['imageSize'])),
    transforms.RandAugment(),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    #timm.data.random_erasing.RandomErasing(probability=1, mode='pixel', device='cpu'),
    transforms.ToTensor(),
    #transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

valTransforms = transforms.Compose([
    transforms.Resize((FLAGS['image_size_initial'],FLAGS['image_size_initial']), interpolation = FLAGS['interpolation']),
    transforms.CenterCrop((int(FLAGS['imageSize']),int(FLAGS['imageSize']))),
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
        #print(image)
        if self.transform is not None:
            examples["Image"] = self.transform(examples["Image"])
        #print(examples['Image'])
        postTagList = set(examples['tag_string'].split()).intersection(set(self.lb.classes_))

        postTags = self.lb.transform([postTagList])
        examples['labels'] = torch.Tensor(postTags)
        #print(examples['labels'])
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
        .map(transformsCallable(tagList, trainTransforms))
    '''
    .shuffle(buffer_size=1000, seed=42)
    .filter(lambda x: (x['__index_level_0__'] % 10) < moduloBound) \
    .shuffle(buffer_size=1000, seed=42)
    '''
    val_ds = myDataset['train'] \
        .with_format("torch")
    '''
    .filter(lambda x: (x['__index_level_0__'] % 10) >= moduloBound) \
    .map(transformsCallable(tagList, valTransforms)) \
    .shuffle(buffer_size=1000, seed=42)
    '''
        
    global classes
    #classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    #classes = {classIndex : className for classIndex, className in enumerate(range(1000))}
    classes = {classIndex : className for classIndex, className in enumerate(tagList)}
    
    image_datasets = {'train': train_ds, 'val' : val_ds}   # put dataset into a list for easy handling
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
    #model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('edgenext_xx_small', pretrained=False, num_classes=len(classes))
    #model = timm.create_model('tf_efficientnetv2_b3', pretrained=False, num_classes=len(classes), drop_rate = 0.00, drop_path_rate = 0.0)
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=len(classes))

    
    #model = add_ml_decoder_head(model)
    
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
    
    accelerator = Accelerator()
    dl_prep_fn = accelerator.prepare_data_loader
    
    
    dataloaders = {x: dl_prep_fn(
        torch.utils.data.DataLoader(
            image_datasets[x], 
            batch_size=FLAGS['batch_size'], 
            #num_workers=FLAGS['num_workers'], 
            #persistent_workers = True, 
            prefetch_factor=2,
            pin_memory = True, 
            drop_last=True, 
            generator=torch.Generator().manual_seed(41))) for x in image_datasets} # set up dataloaders
    
    
    #mixup = Mixup(mixup_alpha = 0.1, cutmix_alpha = 0, label_smoothing=0)
    #dataloaders['train'].collate_fn = mixup_collate
    
    #dataset_sizes = {x: int((image_datasets[x].info.splits[x].num_examples / FLAGS['batch_size'])/8) for x in image_datasets}
    
    device = accelerator.device


    print("initialized training, time spent: " + str(time.time() - startTime))
    

    #criterion = SoftTargetCrossEntropy()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=0, clip=0)
    '''
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    '''
    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=FLAGS['num_epochs'], epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = FLAGS['resume_epoch']
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['resume_epoch'], FLAGS['num_epochs']):
        epochTime = time.time()
        print("starting epoch: " + str(epoch))
        '''
        image_datasets['train'].transform = transforms.Compose([
            transforms.Resize((FLAGS['imageSize'],FLAGS['imageSize'])),
            #transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            #timm.data.random_erasing.RandomErasing(probability=1, mode='pixel', device='cpu'),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_datasets['val'].transform = transforms.Compose([
            transforms.Resize((FLAGS['imageSize'],FLAGS['imageSize'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        '''
        for phase in ['train', 'validation']:
            image_datasets[phase].set_epoch(epoch)
        
            samples = 0
            correct = 0
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
                
                
            if phase == 'validation':
                modelDir = create_dir(FLAGS['modelDir'])
                torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;

            loaderIterable = enumerate(dataloaders[phase])
            for i, data in loaderIterable:
                imageBatch = data['Image']
                tagBatch = data['labels']
                #print(imageBatch.shape)
                #print(tagBatch.shape)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    #if phase == 'train':
                        #imageBatch, tagBatch = mixup(imageBatch, tagBatch)
                    
                    outputs = model(imageBatch)
                    loss = criterion(outputs, tagBatch)
                    #print("loss")
                    #print(loss)

                    # backward + optimize only if in training phase
                    if phase == 'train' and (loss.isnan() == False):
                        accelerator.backward(loss)
                        #print("backward")
                        #if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()
                        #print("optim")
                            
                                    
                if i % stepsPerPrintout == 0:
                    #print("accuracy")

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\t' % (epoch, FLAGS['num_epochs'], i, 0, loss, imagesPerSecond))

            if phase == 'train':
                scheduler.step()
        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        print()

'''
def _mp_fn(rank, flags, image_datasets, model):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    trainCycle(image_datasets, model)
'''

def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files
    print("getting datasets")
    image_datasets = getData()
    print("getting model")
    model = modelSetup(classes)
    #xmp.spawn(_mp_fn, args=(FLAGS, image_datasets, model,), nprocs=8, start_method='fork')
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()