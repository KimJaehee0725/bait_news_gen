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
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

_logger = logging.getLogger('train')
def sampling_data(file_df) :
    """
    train : 40_000
    validation : 13_000
    test : 13_000
    """
    train_sampler = StratifiedShuffleSplit(n_splits=1, test_size=40_000, random_state=42)
    train_sampler = train_sampler.split(file_df, file_df['category'])
    dropped_idx, selected = next(train_sampler)
    train_df = file_df.iloc[selected].reset_index(drop=True)
    dropped_df = file_df.iloc[dropped_idx].reset_index(drop=True)

    valid_sampler = StratifiedShuffleSplit(n_splits=1, test_size=13_000, random_state=42)
    valid_sampler = valid_sampler.split(dropped_df, dropped_df['category'])
    dropped_idx, selected = next(valid_sampler)
    valid_df = dropped_df.iloc[selected].reset_index(drop=True)

    test_sampler = StratifiedShuffleSplit(n_splits=1, test_size=13_000, random_state=42)
    test_sampler = test_sampler.split(dropped_df, dropped_df['category'])
    dropped_idx, selected = next(test_sampler)
    test_df = dropped_df.iloc[selected].reset_index(drop=True)
    return train_df, valid_df, test_df

class BaitDataset(Dataset):
    def __init__(self,config, split, tokenizer):
        
        self.tokenizer = tokenizer
        self.max_len = config['max_word_len']

        self.id_list, self.title_list, self.body_list, self.label_list = self.load_dataset(
            data_dir=config['data_path'], 
            fake_path=config['fake_path'],
            fake_name=config['fake_name'],  
            split=split
            )

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        title = self.title_list[index] 
        body = self.body_list[index] 
        label = self.label_list[index]
        encoding = self.tokenizer.encode_plus( # automatically pad first
            text = str(title),
            text_pair = str(body),
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


    def load_dataset(self, data_dir, fake_path, fake_name, split) -> Tuple[dict, dict, dict, dict]:
        
        _logger.info(f'load {os.path.join(fake_path, fake_name)} raw data')
        
        data_df = pd.DataFrame()
        
        real_dir = os.path.join(data_dir, 'Real')
        bait_dir = os.path.join(data_dir, fake_path)
        if not os.path.exists(os.path.join(bait_dir, 'train.csv')):
            self.split_dataset(bait_dir, fake_name)

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
        title_list = []
        label_list = []
        for row in tqdm(data_df.iterrows(), desc = 'load data') :
            row = row[1]
            title_list.append(row['original_title']) if row['label'] == 1 else title_list.append(row['fake_title'])
            label_list.append(row['label'])
        body_list = list(data_df['original_content'])

        return id_list, title_list, body_list, label_list

    def split_dataset(self, data_dir, fake_name) :
        data = pd.read_csv(os.path.join(data_dir, fake_name))
        train_df, valid_df, test_df = sampling_data(data)
        train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        valid_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)