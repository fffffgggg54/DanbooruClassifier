# based off of https://github.com/fffffgggg54/CelebA_playground/blob/main/trainer.py
# CelebA works as a small scale toy problem for multi-label classification
# too many shared utilities between Danbooru and CelebA, so moving here to share

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
import timm.layers.ml_decoder as ml_decoder
import time
import timm.optim

import numpy as np



from typing import Optional

import torch
from torch import nn
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn
import modelZooCelebA as mz
import handleMultiLabel as MLUtil

import pandas as pd

lr = 3e-3
lr_warmup_epochs = 5
num_epochs = 100
batch_size = 1024
grad_acc_epochs = 1
num_classes = 40
weight_decay = 2e-2
resume_epoch = 0
threshold_multiplier = 1

device = 'cuda:0'

def getDataLoader(dataset):
    return torch.utils.data.DataLoader(dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=16,
        persistent_workers = True,
        prefetch_factor=2, 
        pin_memory = True, 
        drop_last=True, 
        generator=torch.Generator().manual_seed(41)
    )


if __name__ == '__main__':



    train_ds = torchvision.datasets.CelebA(
        './data/',
        'train', 
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandAugment(),
            #transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )
    test_ds = torchvision.datasets.CelebA(
        './data/',
        'valid', 
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    )
    tagNames = pd.read_csv('./data/celeba/list_attr_celeba.txt', header=1, delim_whitespace=True).columns.values.tolist()


    datasets = {'train':train_ds,'val':test_ds}
    dataloaders = {x: getDataLoader(datasets[x]) for x in datasets}


    
    model = mz.resnet20w(num_classes = num_classes)
    

    #model = mz.add_ml_decoder_head(model)
    
    
    
    if (resume_epoch > 0):
        model.load_state_dict(torch.load('./models/saved_model_epoch_' + str(resume_epoch - 1) + '.pth'))
        for param in model.parameters():
            param.requires_grad = True
    
    model=model.to(device)
    #criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0.0)
    #criterion = AsymmetricLossAdaptiveWorking()
    #criterion = AsymmetricLossSigmoidMod(gamma_neg=0, gamma_pos=0, clip=0.0)
    #criterion = SPLCModified(margin = 0.0, loss_fn = nn.BCEWithLogitsLoss())
    #criterion = Hill()
    criterion = MLUtil.AdaptiveWeightedLoss(initial_weight = 1.0, lr = 1e-4, weight_limit = 1e5)
    #criterion = SymHill()
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = timm.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=lr, 
        steps_per_epoch=len(dataloaders['train']),
        epochs=num_epochs, 
        pct_start=lr_warmup_epochs/num_epochs
    )
    
    boundaryCalculator = MLUtil.getDecisionBoundary(initial_threshold = 0.5, lr = 1e-5, threshold_min = 0.1, threshold_max = 0.9)
    
    scheduler.last_epoch = len(dataloaders['train'])*resume_epoch
    cycleTime = time.time()
    epochTime = time.time()
    stepsPerPrintout = 50
    for epoch in range(resume_epoch, num_epochs):
        AP_regular = []
        AccuracyRunning = []
        for phase in ['train', 'val']:
            targets_running = []
            preds_running = []
            cm_tracker = MLUtil.MetricTracker()
            cm_tracker_unmod = MLUtil.MetricTracker()
            if phase == 'train':
                model.train()  # Set model to training mode
                #if (hasTPU == True): xm.master_print("training set")
                print("training set")
            else:
                #torch.save(model.state_dict(), './models/saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                
                print("validation set")
            for i,(image,labels) in enumerate(dataloaders[phase]):
                image = image.to(device, non_blocking=True)
                labels = labels.float().to(device)
                
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(image)
                    
                    
                    preds = torch.sigmoid(outputs)
                    targets_running.append(labels.detach().clone())
                    preds_running.append(preds.detach().clone())
                    boundary = boundaryCalculator(preds, labels)
                    predsModified = (preds > boundary).float()
                    multiAccuracy = cm_tracker.update(predsModified, labels)
                    multiAccuracyUnmod = cm_tracker_unmod.update(preds, labels)
                    accuracy = MLUtil.mAP(
                        labels.numpy(force=True),
                        outputs.sigmoid().numpy(force=True)
                    )
                    #with torch.no_grad():
                    #    targs = torch.where(preds > boundary.detach(), torch.tensor(1).to(preds), labels) # hard SPLC
                    #    targs = stepAtThreshold(preds, boundary.detach()).detach().clone() # soft SPLC
                    
                    shiftedLogits = outputs + threshold_multiplier * torch.special.logit(boundary.detach().clone(), eps=1e-12)
                    
                    #loss = criterion(outputs, labels)
                    
                    #loss = criterion(outputs, targs) if epoch > 10 else criterion(outputs, labels)
                    #loss = criterion(shiftedLogits, targs) if epoch > 0 else criterion(outputs, labels)
                    loss = criterion(shiftedLogits, labels)
                    #criterion.tau_per_class = boundary + 0.1
                    #loss = criterion(outputs, labels, epoch)
                    if loss.isnan():
                        print(outputs.cpu())
                        print(outputs.cpu().sigmoid())
                        #exit()
                    if phase == 'train':
                        loss.backward()
                        if(i % grad_acc_epochs == 0):
                            '''
                            nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                max_norm=1.0, 
                                norm_type=2
                            )
                            '''
                            optimizer.step()
                            optimizer.zero_grad()
                        scheduler.step()
                    
                    if i % stepsPerPrintout == 0:
                        
                        imagesPerSecond = (batch_size * stepsPerPrintout)/(time.time() - cycleTime)
                        cycleTime = time.time()
                        torch.set_printoptions(linewidth = 200, sci_mode = False)
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloaders[phase])}]\tLoss: {loss:.4f}\tImages/Second: {imagesPerSecond:.4f}\tAccuracy: {accuracy:.2f}\t {[f'{num:.2f}' for num in (multiAccuracy * 100).tolist()]}\t{[f'{num:.2f}' for num in (multiAccuracyUnmod * 100).tolist()]}")
                        torch.set_printoptions(profile='default')
                    
                    if phase == 'val':
                        AP_regular.append(accuracy)
                        AccuracyRunning.append(multiAccuracy)
            
            if (phase == 'val'):
                #torch.set_printoptions(profile="full")
                AvgAccuracy = cm_tracker.get_full_metrics()
                AvgAccuracyUnmod = cm_tracker_unmod.get_full_metrics()
                LabelledAccuracy = list(zip(AvgAccuracy.tolist(), AvgAccuracyUnmod.tolist(), tagNames, boundaryCalculator.thresholdPerClass))
                LabelledAccuracySorted = sorted(LabelledAccuracy, key = lambda x: x[0][8], reverse=True)
                MeanStackedAccuracy = cm_tracker.get_aggregate_metrics()
                MeanStackedAccuracyStored = MeanStackedAccuracy[4:]
                print(*LabelledAccuracySorted, sep="\n")
                #torch.set_printoptions(profile="default")
                print(MeanStackedAccuracy)
                print(cm_tracker_unmod.get_aggregate_metrics())
                
                
            mAP_score_regular = MLUtil.mAP(torch.cat(targets_running).numpy(force=True), torch.cat(preds_running).numpy(force=True))
            print("mAP score regular {:.2f}".format(mAP_score_regular))

                    

        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()
    print(model)