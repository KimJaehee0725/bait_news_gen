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

_logger = logging.getLogger('train')

class BaitDataset(Dataset):
    def __init__(self, config, split, tokenizer):
        
        self.tokenizer = tokenizer
        self.max_len = config['max_word_len']
        self.vocab = self.tokenizer.get_vocab() #monologg/kobert

        # special token index
        self.pad_idx = self.tokenizer.pad_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.original_title, self.original_body, self.original_title_num, self.original_file_path = self.load_dataset(data_dir=config['data_path'], bait_dir=config['bait_path'], sort=config['sort'], split=split)

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
        
        label = 1 if ('Direct' in path) or ("Auto" in path) else 0
        doc = {}
        doc['input_ids']=encoding['input_ids'].flatten()
        doc['attention_mask']=encoding['attention_mask'].flatten()

        return doc, label
    

    def load_dataset(self, data_dir, bait_dir, sort, split) -> Tuple[dict, dict, dict, dict]:
        _logger.info(f'load {split} raw data')
        
        if 'test' not in split:
            data_name = sort.split(sep='_') #['News', 'Direct']
        else:
            data_name = split.split(sep='_') 
            data_name.pop(0) #['test','News'] -> ['News'] 리스트 형태 유지
            split = 'test'

        data_path = []
        for data in data_name:
            if data == 'Auto':
                train_dir = bait_dir
            else:
                train_dir = data_dir
        
            data_path = data_path + glob(os.path.join(train_dir, split, data, '*/*'))                             

        news_data_path = [path for path in data_path if 'News' in path]
        bait_data_path = [path for path in data_path if 'News' not in path]

        # get news data
        news_title = {}
        news_body = {}
        news_title_num = {}
        news_file_path = {}
        for num, filename in tqdm(enumerate(news_data_path), desc = f'Load Data {split} | News : ', total=len(news_data_path)) :
            news = json.load(open(filename,'r'))
            news_id = news['sourceDataInfo']['newsID'] #데이터명
            news_title[news_id] = news['sourceDataInfo']['newsTitle']
            news_body[news_id] = news['sourceDataInfo']['newsContent']
            news_title_num[num] = news_id
            news_file_path[news_id] = filename

        # get bait data
        bait_title = {}
        bait_body = {}
        bait_title_num = {}
        bait_file_path = {}
        for num, filename in tqdm(enumerate(bait_data_path), desc = f'Load Data {split} | Bait : ', total=len(bait_data_path)) :
            news = json.load(open(filename,'r'))
            news_id = news['sourceDataInfo']['newsID'] #데이터명
            bait_title[news_id] = news['labeledDataInfo']['newTitle']
            bait_body[news_id] = news['sourceDataInfo']['newsContent']
            bait_title_num[num] = news_id
            bait_file_path[news_id] = filename

        # original_title = {}
        # original_body = {}
        # original_title_num = {}
        # original_file_path = {}
        # for num, filename in tqdm(enumerate(data_path), desc = f'Load Data {split} : ') :
        #     news = json.load(open(filename,'r'))
        #     news_id = news['sourceDataInfo']['newsID'] #데이터명
        #     if "News" in data_path : 
        #         original_title[news_id] = news['sourceDataInfo']['newsTitle']
        #     else :
        #         original_title[news_id] = news['labeledDataInfo']['newTitle']
        #     original_body[news_id] = news['sourceDataInfo']['newsContent']
        #     original_title_num[num] = news_id
        #     original_file_path[news_id] = filename

        total_title = {**news_title, **bait_title}
        total_body = {**news_body, **bait_body}
        total_title_num = {**news_title_num, **bait_title_num}
        total_file_path = {**news_file_path, **bait_file_path}

        return total_title, total_body, total_title_num, total_file_path

