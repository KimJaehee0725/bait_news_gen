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

        data_df = pd.DataFrame()
        for data in data_name:
            if data == 'Auto':
                train_dir = data_dir + bait_dir #fake
            else:
                train_dir = data_dir + '/Real' #real

            if split == 'train':
                df = pd.read_csv(os.path.join(train_dir, 'train.csv')) #data load

            elif split == 'validation':
                df = pd.read_csv(os.path.join(train_dir, 'val.csv'))

            elif split == 'test':
                df = pd.read_csv(os.path.join(train_dir, 'test.csv'))

            data_df = pd.concat([data_df,df], ignore_index=True) #train data

        print(f'{split} : ', len(data_df))

        id_list = list(data_df['news_id'])
        title_list = list(data_df['original_title']) + list(data_df['bait_title'])
        body_list = list(data_df['content'])
        label_list = list(data_df['label'])

        return id_list, title_list, body_list, label_list
