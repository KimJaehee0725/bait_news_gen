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
            bait_sort=config['bait_sort'],  
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


    def load_dataset(self, data_dir, bait_sort, split) -> Tuple[dict, dict, dict, dict]:
        
        bait = bait_sort.split('/')[1]
        _logger.info(f'load {bait} raw data')
        
        data_df = pd.DataFrame()
        
        real_dir = data_dir + 'Real'
        bait_dir = data_dir + bait_sort
        
        for dir in [real_dir, bait_dir]:
            if split == 'train':
                df = pd.read_csv(os.path.join(dir, 'train.csv')) #data load

            elif split == 'validation':
                df = pd.read_csv(os.path.join(dir, 'val.csv'))

            elif split == 'test':
                df = pd.read_csv(os.path.join(dir, 'test.csv'))
            
            data_df = pd.concat([data_df,df], ignore_index=True) #train data


        print(f'{split} : ', len(data_df))

        id_list = list(data_df['news_id'])
        title_list = list(data_df['original_title']) + list(data_df['bait_title'])
        body_list = list(data_df['content'])
        label_list = list(data_df['label'])

        return id_list, title_list, body_list, label_list
