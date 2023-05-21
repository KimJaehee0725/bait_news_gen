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

import pandas as pd

TRAIN_DATA_SIZE = 40_000
VAL_DATA_SIZE = 13_000
TEST_DATA_SIZE = 13_000

_logger = logging.getLogger('train')

class BaitDataset(Dataset):
    def __init__(self,config, split, tokenizer):
        
        self.tokenizer = tokenizer
        self.max_len = config['max_word_len']

        self.id_list, self.title_list, self.body_list, self.label_list = self.load_dataset(
            data_dir=config['data_path'], 
            bait_dir=config['bait_sort'], 
            sort=config['model_sort'], 
            split=split
            )


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        title = self.title_list[index] 
        body = self.body_list[index] 
        label = self.label_list[index]

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
        self.train_category_num, self.val_category_num, self.test_category_num = self.get_category_ratio(os.path.join(data_dir, 'Real', 'train.csv'))
        #! real 데이터의 train의 비율만 가지고 train/val/test 모두 처리하는 것인가?

        data_df = pd.DataFrame()
        for data in data_name:
            if data == 'Auto':
                train_dir = data_dir + bait_dir #fake
            else:
                train_dir = data_dir + '/Real' #real

            if split == 'train':
                df = pd.read_csv(os.path.join(train_dir, 'train.csv')) #data load
                loc_dict = self.train_category_num

            elif split == 'validation':
                df = pd.read_csv(os.path.join(train_dir, 'val.csv'))
                loc_dict = self.val_category_num

            elif split == 'test':
                df = pd.read_csv(os.path.join(train_dir, 'test.csv'))
                loc_dict = self.test_category_num
            

            for cate in list(self.train_category_num.keys()):
                cate_df = df.loc[df.category == cate][:loc_dict[cate]] #category loc
                print(f'sampled {cate} size : ', len(cate_df))
                data_df = pd.concat([data_df, cate_df], ignore_index=True) #sampled data
        
        print('data_df length : ', len(data_df))

        id_list = list(data_df['news_id'])
        title_list = list(data_df['original_title']) + list(data_df['bait_title'])
        body_list = list(data_df['content'])
        label_list = list(data_df['label'])

        return id_list, title_list, body_list, label_list
    

    def load_data_path(self, data_dir, bait_dir, sort, split) :
        _logger.info(f'load {split} raw data')
        
        if 'test' not in split:
            data_name = sort.split(sep='_') #['News', 'Direct']
        else:
            data_name = split.split(sep='_') 
            data_name.pop(0) #['test','News', 'Direct'] -> ['News','Direct'] 리스트 형태 유지
            split = 'test'

        train_category_num, val_category_num, test_category_num = self.get_category_ratio(os.path.join(data_dir, 'Real', 'train.csv'))
        self.category_num = {'train' : train_category_num, 'validation' : val_category_num, 'test' : test_category_num}
       

    def get_category_ratio(self, real_dir) -> Tuple[dict, dict, dict]:
        
        data = pd.read_csv(real_dir)
        category = data['category'].unique()
        cate_count = data['category'].value_counts()
        cate_dict = {}

        ## get ratio
        total = len(data)
        for cat in category:
            cate_dict[cat] = cate_count[cat]/total
        
        _logger.info(f"Categories : {category}")
        
        train_category_num = {cat : int(TRAIN_DATA_SIZE * cate_dict[cat]) for cat in category}
        val_category_num = {cat : int(VAL_DATA_SIZE * cate_dict[cat]) for cat in category}
        test_category_num = {cat : int(TEST_DATA_SIZE * cate_dict[cat]) for cat in category}

        print('train size : ', TRAIN_DATA_SIZE)   
        print('val size : ', VAL_DATA_SIZE)     

        return train_category_num, val_category_num, test_category_num
    
