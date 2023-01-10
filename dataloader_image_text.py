import os

import numpy as np
import pydicom as pdcm
import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk import word_tokenize, sent_tokenize
import nltk
import random
import pandas as pd

#import matplotlib as plt

from utility import rle_decode_modified, rle_decode

class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True,
                 dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode=None, transforms=None, resize=None,
                 img_size=256,
                 wordDict = None,
                 ngram_synonom = []):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len
        self.img_size = img_size
        self.wordDict = wordDict

        self.df_data = dataframe.values
        self.transforms = transforms
        self.mode = mode
        self.data_path = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")
        self.dir_base = dir_base
        self.resize = resize

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # text extraction
        text = str(self.text[index])
        text = " ".join(text.split())
        text = text.replace("[ALPHANUMERICID]", "")
        text = text.replace("[date]", "")
        text = text.replace("[DATE]", "")
        text = text.replace("[AGE]", "")

        text = text.replace("[ADDRESS]", "")
        text = text.replace("[PERSONALNAME]", "")
        text = text.replace("\n", "")

        #if self.wordDict != None:
        #    text = TextImageDataset.synonymsReplacement(self, text)
        #    text = TextImageDataset.shuffledTextAugmentation(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding= 'max_length',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # images data extraction
        img_name = self.row_ids[index]
        img_name = str(img_name)  # + "_mip.png"
        img_path = os.path.join(self.data_path, img_name)

        try:
            DCM_Img = pdcm.read_file(img_path)
            img_raw = DCM_Img.pixel_array
            img_norm = img_raw * (255 / np.amax(img_raw))  # puts the highest value at 255
            img = np.uint8(img_norm)

        except:
            print("can't open")
            print(img_path)

        # decodes the rle
        if self.targets[index] != str(-1):
            segmentation_mask_org = rle_decode(self.targets[index], (1024, 1024))
            #segmentation_mask_org = rle_decode_modified(self.targets[index], (1024, 1024))
            segmentation_mask_org = np.uint8(segmentation_mask_org)
        else:
            segmentation_mask_org = np.zeros((1024, 1024))
            segmentation_mask_org = np.uint8(segmentation_mask_org)

        RGB = True # should make this more rigous but switch this guy if only have 1 channel

        if self.transforms is not None:

            if self.mode == "train":
                if RGB:
                    img = Image.fromarray(img).convert("RGB")
                else:
                    img = Image.fromarray(img)
                img = np.array(img)
                transformed = self.transforms(image=img, mask=segmentation_mask_org)
                image = transformed['image']
                segmentation_mask_org = transformed['mask']
                image = Image.fromarray(np.uint8(image))  # makes the image into a PIL image
                image = self.resize(image)  # resizes the image to be the same as the model size
            else:
                if RGB:
                    img = Image.fromarray(img).convert("RGB")  # makes the image into a PIL image
                    image = self.resize(img)
                else:
                    img = Image.fromarray(img)
                    image = self.transforms(img)
        else:
            image = img

        segmentation_mask = Image.fromarray(np.uint8(segmentation_mask_org))
        segmentation_mask = self.resize(segmentation_mask)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'targets': segmentation_mask,
            'row_ids': self.row_ids[index],
            'images': image
        }


    def shuffledTextAugmentation(text):

        sentences = sent_tokenize(text)
        random.shuffle(sentences)
        shuffledText = sentences[0]
        for i in range(1, len(sentences)):
            shuffledText += " " + sentences[i]
        return shuffledText


    def synonymsReplacement(self, text):

        wordDict = self.wordDict

        newText = text
        for word in list(wordDict["synonyms"].keys()):
            if word in text:
                randValue = random.uniform(0, 1)
                if randValue <= .15:
                    randomSample = np.random.randint(low = 0, high = len(wordDict['synonyms'][word]))
                    newText = text.replace(word, wordDict["synonyms"][word][randomSample])

        return newText