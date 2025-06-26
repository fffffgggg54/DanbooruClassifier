import os
import torch
import torch.utils.data
import torchvision
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
import gc

import multiprocessing

import tarfile



from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

class FileReader:
    def __init__(self, rootPath):
        self.rootPath = rootPath
    def read_file(self, file_path):
        with open(rootPath + file_path, 'rb') as f:
            return f.read()

class TarReader:
    def __init__(self, tar_path):
        self.tar_path = tar_path
        self._index = {}
        self._tar_file = None
        
        self._build_index()
    
    def _build_index(self):
        
        index_path = self.tar_path + '.TARINFO.pkl.bz2'
        if os.path.exists(index_path):
            gc.disable()
            cached_index = bz2.BZ2File(index_path, 'rb')
            self._index = cPickle.load(cached_index)
            gc.enable()
            cached_index.close()
        else:
            print(f"Building index for {self.tar_path}. This may take a while...")
            start_time = time.time()
            tar_file = tarfile.open(self.tar_path, 'r:')
            for member in tar_file.getmembers():
                if member.isfile():
                    self._index[member.name] = member
            
            with bz2.BZ2File(index_path, 'w') as cached_index: cPickle.dump(self._index, cached_index)

            end_time = time.time()
            print(f"Index built for {len(self._index)} files in {end_time - start_time:.2f} seconds.")

    def read_file(self, file_path):
        """
        Reads a single file from the archive using the index for fast access.
        Returns the file content as bytes, or None if not found.
        """
        member_info = self._index.get(file_path)
        
        if not member_info:
            print(f"Warning: File '{file_path}' not found in the archive index.")
            return None
        if self._tar_file is None:
            self._tar_file = tarfile.open(self.tar_path, 'r:')
        extracted_file = self._tar_file.extractfile(member_info)
        
        if extracted_file:
            content = extracted_file.read()
            extracted_file.close()
            return content
        return None


class CocoDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self,
        root,
        annFile,
    ):
        super().__init__(
            root, 
            annFile, 
            transform=None,
            target_transform=None,
            transforms=None
        )
        # plaintext names of each class
        self.classes = [x['name'] for x in self.coco.cats.values()]
        # lookup for coco category id to index of class
        self.id_to_idx = {x : idx for idx, x in enumerate(list(self.coco.cats.keys()))}
        self.transform = None
        
        
    def __getitem__(self, index):

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        # onehot vector
        new_target = torch.zeros(80, dtype=torch.long)
        indices = [self.id_to_idx[instance['category_id']] for instance in target]
        for idx in indices:
            new_target[idx] +=1
        target = (new_target > 0).to(torch.long)

        if self.transform is not None:
            image_out = self.transform(image)

        image.close()
        return image_out, target

class DanbooruDataset(torch.utils.data.Dataset):


    def __init__(self, imageRoot, postList, tagList, transform=None, cacheRoot = None):

        #PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        #self.classes = deepcopy({classIndex : className for classIndex, className in enumerate(deepcopy(tagList))}) #property of dataset?
        self.postList = deepcopy(postList)    #dataframe with string type, not object
        #self.imageRoot = deepcopy(imageRoot)  #string
        #self.tagList = tagList
        self.tagList = pd.Series(deepcopy(tagList), dtype=pd.StringDtype())
        self.transform = transform  #transform, callable?
        self.cacheRoot = deepcopy(cacheRoot)  #string

    def __len__(self):
        return deepcopy(len(self.postList))
    
    # TODO profile and optimize
    #@profile
    def __getitem__(self, index):
    

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
            cachedSample.close()
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
            
            
        return image, postTags
        
        

def DFServerWorkerProcess(workQueue, myDF, tagList, imageRoot, imageCacheRoot, tagCacheRoot):
    locator = myDF.loc
    while(1):
        (index, returnConnection) = workQueue.get()
        returnConnection.send((locator[index].copy(deep=True), tagList, imageRoot, imageCacheRoot, tagCacheRoot))
        returnConnection.close()

class DanbooruDatasetWithServer(torch.utils.data.Dataset):


    def __init__(self, postData, tagData, imageRoot, cacheRoot, size, serverWorkerCount, transform=None):

        #PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        #self.classes = {classIndex : className for classIndex, className in enumerate(tagList)} #property of dataset?
        self.postListLength = len(postData)
        #self.imageRoot = imageRoot  #string
        #self.tagList = tagList
        #self.tagList = pd.Series(tagList, dtype=pd.StringDtype())
        self.transform = transform  #transform, callable?
        #self.cacheRoot = cacheRoot  #string
        self.size = size
        self.num_tags = len(tagData)
        self.serverWorkerCount = serverWorkerCount
        self.serverProcessPool = []
        self.workQueue = multiprocessing.Queue()
        if cacheRoot is not None:
            imageCacheRoot = cacheRoot + 'images/' + str(size) + '/'
            #imageCacheRoot = cacheRoot + str(size) + '/'
            tagCacheRoot = cacheRoot + 'tags/' + str(self.num_tags) + '/'
        for nthWorkerProcess in range(self.serverWorkerCount):
            currProcess = multiprocessing.Process(target=DFServerWorkerProcess,
                args=(self.workQueue,
                    postData.copy(deep=True),
                    pd.Series(tagData.name.copy(deep=True), dtype=pd.StringDtype()),
                    imageRoot,
                    imageCacheRoot,
                    tagCacheRoot,),
                daemon = True)
            currProcess.start()
            self.serverProcessPool.append(currProcess)

    def __len__(self):
        return self.postListLength

    # TODO profile and optimize
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        #startTime = time.time()
        #postData = self.postList.iloc[index].copy()
        #postData.tag_string = postData.tag_string.split()

        recvConn, sendConn = multiprocessing.Pipe()

        self.workQueue.put((index, sendConn))

        (postData, tagList, imageRoot, imageCacheRoot, tagCacheRoot) = recvConn.recv()

        postID = int(postData.loc["id"])
        image = torch.Tensor()
        postTags = torch.Tensor()
        bruh = False

        try:
            tagCacheDir = create_dir(tagCacheRoot + str(postID % 1000).zfill(4))
            tagCachePath = tagCacheDir + "/" + str(postID) + ".pkl.bz2"
            with bz2.BZ2File(tagCachePath, 'rb') as cachedTags:
                postTags = cPickle.load(cachedTags)

            #print(f"got pickle from {cachePath}")
            '''
            if len(postTags) != len(tagList):
                postTagList = set(postData.loc["tag_string"].split()).intersection(set(tagList.to_list()))

                # one-hot encode the tags of a given post
                # TODO find better way to find matching tags
                postTags = []
                for key in list(tagList.to_list()):
                    match = False
                    for tag in postTagList:
                        if tag == key:
                            match = True

                    postTags.append(int(match))
                postTags = torch.Tensor(postTags)
            '''
        
        except Exception as e:
            #print(e)
            #print("cached file not found")
            postTagList = set(postData.loc["tag_string"].split()).intersection(set(tagList.to_list()))

            # one-hot encode the tags of a given post
            # TODO find better way to find matching tags
            postTags = []
            for key in list(tagList.to_list()):
                match = False
                for tag in postTagList:
                    if tag == key:
                        match = True

                postTags.append(int(match))
            
            postTags = torch.Tensor(postTags)
            
            if(tagCacheRoot is not None):    
                tagCacheDir = create_dir(tagCacheRoot + str(postID % 1000).zfill(4))
                tagCachePath = tagCacheDir + "/" + str(postID) + ".pkl.bz2"
                with bz2.BZ2File(tagCachePath, 'w') as cachedSample: cPickle.dump(postTags, cachedSample)
        
        
        try:
            imageCacheDir = create_dir(imageCacheRoot + str(postID % 1000).zfill(4))
            imageCachePath = imageCacheDir + "/" + str(postID) + ".jpeg"
            
            image = Image.open(imageCachePath)    #check if file exists
            image.load()
            
        except Exception as e:
            #metaTime = time.time() - startTime
            #startTime = time.time()
            imagePath = str(postID % 1000).zfill(4) + "/" + str(postID) + "." + postData.loc["file_ext"]
            #cachedImagePath = cacheRoot + imagePath
            imagePath = imageRoot + imagePath

            try: 
                #path = cachedImagePath
                path = imagePath
                '''
                # TODO mess with gpu decoding, needs rewrite for how stuff is handled later on
                if postData.loc["file_ext"] == 'jpg' and torch.cuda.is_available():
                    image = torchvision.io.read_file(path)
                    image = torchvision.io.decode_jpeg(image, device='cuda:0')
                    image = torchvision.transforms.functional.to_pil_image(image)
                else:
                '''
                image = Image.open(path)    #check if file exists
                image.load()    # check if file valid
            except:     #if file doesn't exist or isn't valid, download it and save/overwrite
                imageURL = postData.loc["file_url"]
                #print("Getting image from " + imageURL)
                response = requests.get(imageURL)
                image = Image.open(BytesIO(response.content))
                myFile = open(path, "wb")
                myFile.write(response.content)
                myFile.close()

                #print("Image saved to " + path)
            # TODO implement switchable cache use
            '''
            old caching and crawling
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
            image = transforms.functional.resize(image, (self.size, self.size))
            image = image.convert("RGBA")

            color = (255,255,255)

            background = Image.new('RGB', image.size, color)
            background.paste(image, mask=image.split()[3])
            image = background


            #image = transforms.functional.pil_to_tensor(image).squeeze()
            
            if(imageCacheRoot is not None):
                imageCacheDir = create_dir(imageCacheRoot + str(postID % 1000).zfill(4))
                imageCachePath = imageCacheDir + "/" + str(postID) + ".jpeg"
                image.save(imageCachePath, format='jpeg', quality=75, optimize=True)
                #with bz2.BZ2File(imageCachePath, 'w') as cachedSample: cPickle.dump((image, postTags, postID), cachedSample)


        if self.transform: image = self.transform(image)


        del postData
        # if(torch.utils.data.get_worker_info().id == 1):objgraph.show_growth() 


        return image, postTags


class DanbooruDatasetWithServerAndReader(torch.utils.data.Dataset):


    def __init__(self, postData, tagData, imageReader, tagReader, size, serverWorkerCount, transform=None):
        super().__init__()
        self.imageReader = imageReader
        self.tagReader = tagReader
        self.postListLength = len(postData)
        self.transform = transform  #transform, callable?
        self.size = size
        self.num_tags = len(tagData)
        self.serverWorkerCount = serverWorkerCount
        self.serverProcessPool = []
        self.workQueue = multiprocessing.Queue()
        for nthWorkerProcess in range(self.serverWorkerCount):
            currProcess = multiprocessing.Process(target=DFServerWorkerProcess,
                args=(self.workQueue,
                    postData.copy(deep=True),
                    pd.Series(tagData.name.copy(deep=True), dtype=pd.StringDtype()),
                    None,
                    None,
                    None,),
                daemon = True)
            currProcess.start()
            self.serverProcessPool.append(currProcess)

    def __len__(self):
        return self.postListLength

    # TODO profile and optimize
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        recvConn, sendConn = multiprocessing.Pipe()

        self.workQueue.put((index, sendConn))

        (postData, tagList, imageRoot, imageCacheRoot, tagCacheRoot) = recvConn.recv()

        postID = int(postData.loc["id"])
        image = torch.Tensor()
        postTags = torch.Tensor()

        try:
            tag_path = "tags/" + str(self.num_tags) + '/' + str(postID % 1000).zfill(4) + "/" + str(postID) + ".pkl.bz2"
            tag_bytes = self.tagReader.read_file(tag_path)
            with bz2.BZ2File(BytesIO(tag_bytes), 'rb') as cachedTags:
                postTags = cPickle.load(cachedTags)
        
        except Exception as e:
            print(e)
            print("cached file not found, running tag encode")
            postTagList = set(postData.loc["tag_string"].split()).intersection(set(tagList.to_list()))

            # one-hot encode the tags of a given post
            # TODO find better way to find matching tags
            postTags = []
            for key in list(tagList.to_list()):
                match = False
                for tag in postTagList:
                    if tag == key:
                        match = True

                postTags.append(int(match))
            
            postTags = torch.Tensor(postTags)

        try:
            imagePath = str(self.size) + '/' + str(postID % 1000).zfill(4) + "/" + str(postID) + ".jpeg"
            image_bytes = self.imageReader.read_file(imagePath)           
            image = Image.open(BytesIO(image_bytes))    #check if file exists
            image.load()

        except Exception as e:
            print(e)
            imageURL = postData.loc["file_url"]
            print("Getting image from " + imageURL)
            response = requests.get(imageURL)
            image = Image.open(BytesIO(response.content))
            image = transforms.functional.resize(image, (self.size, self.size))
            image = image.convert("RGBA")

            color = (255,255,255)

            background = Image.new('RGB', image.size, color)
            background.paste(image, mask=image.split()[3])
            image = background

        if self.transform: image = self.transform(image)

        del postData

        return image, postTags

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