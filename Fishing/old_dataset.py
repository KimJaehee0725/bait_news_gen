### dataset class 

import torch
from torch.utils.data import Dataset
import json 
import os
from glob import glob
from copy import copy
from typing import Union, Tuple
import logging
import random
from tqdm import tqdm

TRAIN_DATA_SIZE = 40_000
VAL_DATA_SIZE = 35_000
TEST_DATA_SIZE = 35_000

_logger = logging.getLogger('train')

class BaitDataset(Dataset):
    def __init__(self, config, split, tokenizer):
        
        self.tokenizer = tokenizer
        self.max_len = config['max_word_len']
        self.vocab = self.tokenizer.get_vocab() #monologg/kobert

        # special token index
        self.pad_idx = self.tokenizer.pad_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.original_title, self.original_body, self.original_title_num, self.original_file_path = self.load_dataset(data_dir=config['data_path'], bait_dir=config['bait_path'], sort=config['model_sort'], split=split)

    def __len__(self):
        return len(self.original_title_num)

    def __getitem__(self, index):
        news_id = self.original_title_num[index]
        title = self.original_title[news_id]
        body = self.original_body[news_id]
        path = self.original_file_path[news_id]

        encoding = self.tokenizer.encode_plus( # automatically pad first
            text = title,
            text_pair = body,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        label = 1 if ('Base' in path) or ("Auto" in path) else 0 #!base
        doc = {}
        doc['input_ids']=encoding['input_ids'].flatten()
        doc['attention_mask']=encoding['attention_mask'].flatten()

        return doc, label

    def load_dataset(self, data_dir, bait_dir, sort, split) -> Tuple[dict, dict, dict, dict]:
        _logger.info(f'load {split} raw data')
        
        if 'test' not in split:
            data_name = sort.split(sep='_') #['News', 'Base']
        else:
            data_name = split.split(sep='_') 
            data_name.pop(0) #['test','News', 'Base'] -> ['News','Base'] 리스트 형태 유지
            split = 'test'

        _logger.info(f"Get Category Ratio and Stratified Sampling")
        self.train_category_num, self.val_category_num, self.test_category_num = self.get_category_ratio(os.path.join(data_dir, 'train', 'News'))

        data_path = []
        for data in data_name:
            if data == 'Auto':
                train_dir = bait_dir
            else:
                train_dir = data_dir

            if split == 'train':
                for cate in list(self.train_category_num.keys()):
                    data_path = data_path + glob(os.path.join(train_dir, split, data, cate, "*"))[:self.train_category_num[cate]]
            elif split == 'validation':
                for cate in list(self.val_category_num.keys()):
                    data_path = data_path + glob(os.path.join(train_dir, split, data, cate, "*"))[:self.val_category_num[cate]]
            elif split == 'test':
                for cate in list(self.test_category_num.keys()):
                    data_path = data_path + glob(os.path.join(train_dir, split, data, cate, "*"))[:self.test_category_num[cate]]

            # data_path = data_path + glob(os.path.join(train_dir, split, data, '*/*'))                             

        news_data_path = [path for path in data_path if 'News' in path]
        bait_data_path = [path for path in data_path if 'News' not in path]

        # get news data
        news_title = {}
        news_body = {}
        news_title_num = {}
        news_file_path = {}
        for num, filename in tqdm(enumerate(news_data_path), desc = f'Load Data {split} | News : ', total=len(news_data_path)) :
            news = json.load(open(filename,'r'))
            news_id = news['sourceDataInfo']['newsID'] + "_news" #데이터명
            news_title[news_id] = news['sourceDataInfo']['newsTitle']
            news_body[news_id] = news['sourceDataInfo']['newsContent']
            news_title_num[num] = news_id
            news_file_path[news_id] = filename

        news_len = len(news_title_num)
        # get bait data
        bait_title = {}
        bait_body = {}
        bait_title_num = {}
        bait_file_path = {}
        for num, filename in tqdm(enumerate(bait_data_path), desc = f'Load Data {split} | Bait : ', total=len(bait_data_path)) :
            news = json.load(open(filename,'r'))
            news_id = news['sourceDataInfo']['newsID'] + "_bait" #데이터명
            bait_title[news_id] = news['labeledDataInfo']['newTitle']
            bait_body[news_id] = news['sourceDataInfo']['newsContent']
            bait_title_num[num + news_len] = news_id
            bait_file_path[news_id] = filename

        total_title = {**news_title, **bait_title}
        total_body = {**news_body, **bait_body}
        total_title_num = {**news_title_num, **bait_title_num}
        total_file_path = {**news_file_path, **bait_file_path}

        return total_title, total_body, total_title_num, total_file_path
    
    def load_data_path(self, data_dir, bait_dir, sort, split) :
        _logger.info(f'load {split} raw data')
        
        if 'test' not in split:
            data_name = sort.split(sep='_') #['News', 'Direct']
        else:
            data_name = split.split(sep='_') 
            data_name.pop(0) #['test','News', 'Direct'] -> ['News','Direct'] 리스트 형태 유지
            split = 'test'

        train_category_num, val_category_num, test_category_num = self.get_category_ratio(os.path.join(data_dir, 'train', 'News'))
        self.category_num = {'train' : train_category_num, 'validation' : val_category_num, 'test' : test_category_num}
        data_path = []
        for data in data_name :
            if data == 'Auto':
                train_dir = bait_dir
            else:
                train_dir = data_dir

            for cate in list(self.category_num[split].keys()):
                data_path = data_path + glob(os.path.join(train_dir, split, data, cate, "*"))[:self.category_num[split][cate]]

        return data_path


    def get_category_ratio(self, path_to_category) -> Tuple[dict, dict, dict]:
        category = {}
        categories = os.listdir(path_to_category)
        for cat in categories:
            category[cat] = len(os.listdir(os.path.join(path_to_category, cat)))

        ## get ratio
        total = sum(category.values())
        for cat in category:
            category[cat] = category[cat]/total
        
        _logger.info(f"Category Ratio : {category}")
        
        train_category_num = {cat : int(TRAIN_DATA_SIZE * category[cat]) for cat in category}
        val_category_num = {cat : int(VAL_DATA_SIZE * category[cat]) for cat in category}
        test_category_num = {cat : int(TEST_DATA_SIZE * category[cat]) for cat in category}

        return train_category_num, val_category_num, test_category_num
    
    def load_bait_news_info(self, data_dir, bait_dir, split = "train") -> dict:
        bait_data_path = self.load_data_path(data_dir, bait_dir, 'Auto', split)
        DATA_LIST = ['train', 'validation', 'test']
        bait_title = {}
        bait_file_path = {}
        news_title = {}
        for split in DATA_LIST:
            for num, filename in tqdm(enumerate(bait_data_path), desc = f'Load Data {split} | Bait : ', total=len(bait_data_path)) :
                news = json.load(open(filename,'r'))
                news_id = news['sourceDataInfo']['newsID'] #데이터명
                bait_title[news_id] = news['labeledDataInfo']['newTitle']
                news_title[news_id] = news['sourceDataInfo']['newsTitle']
                bait_file_path[news_id] = filename

        return bait_title, news_title, bait_file_path