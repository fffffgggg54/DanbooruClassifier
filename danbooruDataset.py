import os
import torch
import torch.utils.data
import pandas as pd
import time
import shutil
import numpy as np
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageDraw
import PIL
import requests
from io import BytesIO
import io
import time
import random
import bz2
import pickle
import _pickle as cPickle
import json

from copy import deepcopy




from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

Image.MAX_IMAGE_PIXELS = None

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir
	
# TODO migrate to general data instead of danbo specific data


# TODO migrate to datapipes when mature



class DanbooruDataset(torch.utils.data.Dataset):


    def __init__(self, imageRoot, postList, tagList, transform=None, cacheRoot = None):

        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.classes = {classIndex : className for classIndex, className in enumerate(tagList)} #property of dataset?
        self.postList = postList    #dataframe with string type, not object
        self.imageRoot = imageRoot  #string
        #self.tagList = tagList
        self.tagList = pd.Series(tagList, dtype=pd.StringDtype())
        self.transform = transform  #transform, callable?
        self.cacheRoot = cacheRoot  #string
        self.callCount = 0

    def __len__(self):
        return deepcopy(len(self.postList))
    
    # TODO profile and optimize
    #@profile
    def __getitem__(self, index):
    
        if (self.callCount + 1) % 10000 == 0:
            print(gc.get_referrers(self.classes))
            print(gc.get_referrers(self.postList))
            print(gc.get_referrers(self.imageRoot))
            print(gc.get_referrers(self.tagList))
            print(gc.get_referrers(self.transform))
            print(gc.get_referrers(self.cacheRoot))
        if torch.is_tensor(deepcopy(index)):
            index = deepcopy(index.item())
        
        #startTime = time.time()
        postData = deepcopy(self.postList.iloc[index])
        #postData.tag_string = postData.tag_string.split()
        
        postID = int(deepcopy(postData.loc["id"]))
        image = torch.Tensor()
        postTags = torch.Tensor()
        
        
        
        try:
            assert deepcopy(self.cacheRoot) is not None
            cacheDir = create_dir(deepcopy(self.cacheRoot) + str(postID % 1000).zfill(4))
            cachePath = cacheDir + "/" + str(postID) + ".pkl.bz2"
            cachedSample = bz2.BZ2File(cachePath, 'rb')
            image, postTags,_ = cPickle.load(cachedSample)
            #print(f"got pickle from {cachePath}")
        except:
        
            postTagList = set(deepcopy(postData.loc["tag_string"]).split()).intersection(set(deepcopy(self.tagList.to_list())))

            # one-hot encode the tags of a given post
            # TODO find better way to find matching tags
            postTags = []
            for key in list(deepcopy(self.tagList.to_list())):
                match = False
                for tag in postTagList:
                    if tag == key:
                        match = True
                
                postTags.append(int(match))
        
            
            #metaTime = time.time() - startTime
            #startTime = time.time()
            imagePath = str(postID % 1000).zfill(4) + "/" + str(postID) + "." + deepcopy(postData.loc["file_ext"])
            #cachedImagePath = cacheRoot + imagePath
            imagePath = deepcopy(self.imageRoot) + imagePath
            
            try: 
                #path = cachedImagePath
                path = imagePath
                image = Image.open(path)    #check if file exists
                image.load()    # check if file valid
            except:     #if file doesn't exist or isn't valid, download it and save/overwrite
                imageURL = deepcopy(postData.loc["file_url"])
                #print("Getting image from " + imageURL)
                response = requests.get(imageURL)
                image = Image.open(BytesIO(response.content))
                myFile = open(path, "wb")
                myFile.write(response.content)
                myFile.close()
            
                #print("Image saved to " + path)
            # TODO implement switchable cache use
            '''
            except FileNotFoundError:
                
                try:
                    create_dir(cacheRoot + str(postID % 1000).zfill(4))
                    #print(f"copy {imagePath} to {cachedImagePath}")
                    image = Image.open(shutil.copy2(imagePath, cachedImagePath))
                    image = image.convert("RGB")
           
                except:
                    imageURL = postData.loc["file_url"]
                    print("Getting image from " + imageURL)
                    response = requests.get(imageURL)
                    image = ImageOps.pad(Image.open(BytesIO(response.content)), (512, 512))
                    image = image.convert("RGB")
                    image.save(path)
                    print("Image saved to " + path)
            '''
            #image = ImageOps.exif_transpose(image)
            #imageLoadTime = time.time() - startTime
            #startTime = time.time()
            #process our image
            
            image = image.convert("RGBA")
            
            color = (255,255,255)
            
            background = Image.new('RGB', image.size, color)
            background.paste(image, mask=image.split()[3])
            image = background
            
            
            #image = transforms.functional.pil_to_tensor(image).squeeze()
            
            image = transforms.functional.resize(image, (224,224))
            image = transforms.functional.pil_to_tensor(image)
            
            postTags = torch.Tensor(postTags)
            

            
            if(self.cacheRoot is not None):
                cacheDir = create_dir(deepcopy(self.cacheRoot) + str(postID % 1000).zfill(4))
                cachePath = cacheDir + "/" + str(postID) + ".pkl.bz2"
                with bz2.BZ2File(cachePath, 'w') as cachedSample: cPickle.dump((image, postTags, postID), cachedSample)
        
        image = transforms.functional.to_pil_image(image)
        
        if self.transform: image = self.transform(image)

        
        del postData
        
        # if(torch.utils.data.get_worker_info().id == 1):objgraph.show_growth() 
            
            
        return image, postTags, postID
        
        


def filterDanbooruData(tagData, postData, minPostCount = 10000, blockedRatings = [], blockedTags = ['animated', 'flash', 'corrupted_file', 'corrupted_metadata', 'cosplay_photo']):
    
    # get general tags with more than minPostCount posts and without any of the tags in blockedTags
    queryStartTime = time.time() 
    tagData.query('(category == 0) & (post_count > @minPostCount) & (is_deprecated == False)', inplace = True)
    print("tag query time: " + str(time.time()-queryStartTime)) #tag query time: 0.03899264335632324
    
    # get posts that have an id
    queryStartTime = time.time()
    postData.dropna(subset=["id"], inplace=True)
    print("non-NaN id query time: " + str(time.time()-queryStartTime)) #non-NaN id query time: 2.1638450622558594
    
    # get posts with a rating that is not in blockedRatings
    queryStartTime = time.time()
    postData.query("(rating not in @blockedRatings)", inplace = True)
    #postData.query("(tag_count_general > 30)", inplace = True)
    print("rating query time: " + str(time.time()-queryStartTime)) #rating query time: 6.7838006019592285
    
    # filter posts by id
    queryStartTime = time.time()
    postData.query("(id <= 5400000)", inplace = True)
    #postData.query("(id >= 5000000)", inplace = True)
    print("post age query time: " + str(time.time()-queryStartTime))
    
    # get posts that are not banned
    queryStartTime = time.time()
    postData.query("is_banned == False", inplace = True)
    blockedIDs = [5190773, 5142098, 5210705, 5344403, 5237708, 5344394, 5190771, 5237705, 5174387, 5344400, 5344397, 5174384, 4473254]
    for postID in blockedIDs: postData.query("id != @postID", inplace = True)
    print("banned post query time: " + str(time.time()-queryStartTime))
    
    
    # get posts that are not deleted
    queryStartTime = time.time()
    postData.query("is_deleted == False", inplace = True)
    print("deleted post query time: " + str(time.time()-queryStartTime))
    
    # get posts without any of the tags in blockedTags
    # TODO find something to do this without for loop, very slow
    # TODO profile this against this: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.apply.html
    queryStartTime = time.time()
    ''' loop method
    #toDelete = []
    #tagsToDelete = []
    #for index, post in postData.iterrows():
        #if any(map(lambda v: v in post.tag_string, blockedTags)): toDelete.append(index)
    
    #filter blocked tags out of tags, appears to not be working
    # TODO check effectiveness
    #toDelete = []
    #for index, tag in tagData.iterrows():
        #if any(map(lambda v: v == tag.name, blockedTags)): toDelete.append(index)
    #tagData.drop(toDelete, inplace=True)
    #toDelete = None
    '''
    for tag in blockedTags:
        #postData.drop(postData[postData.tag_string.str.contains("\b" + tag + "\b")].index, inplace=True)
        #tagData.drop(tagData[tagData.name.str.contains("\b" + tag + "\b")].index, inplace=True)
        postDataToDrop = postData[postData.tag_string.str.contains(str("\\b" + tag + "\\b"), regex=True, case=False)].index.to_list()
        tagDataToDrop = tagData[tagData.name.str.match(tag, case=False)].index.to_list()
        print(len(postDataToDrop))
        print(len(tagDataToDrop))
        postData.drop(postDataToDrop, inplace=True)
        tagData.drop(tagDataToDrop, inplace=True)
    print("blocked tag query time: " + str(time.time()-queryStartTime))
    #loop: blocked tag query time: 376.6717150211334
    #str contains: blocked tag query time: 350.0482497215271
    '''
    # convert space delimited tag string to list
    queryStartTime = time.time()
    postData.tag_string = postData.tag_string.str.split()
    print("split time: " + str(time.time()-queryStartTime)) #split time: 37.70384955406189
    '''
    return tagData, postData

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x