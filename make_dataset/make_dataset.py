from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random
import time

from tqdm.auto import tqdm

def update_label_info(file: dict, new_title: str) -> dict:
    '''
    update label information in dictionary 
    '''

    file['labeledDataInfo'] = {
        'newTitle': new_title,
        'clickbaitClass': 0,
        'referSentenceInfo': [
            {'sentenceNo':i+1, 'referSentenceyn': 'N'} for i in range(len(file['sourceDataInfo']['sentenceInfo']))
        ]
    }
    
    return file


def make_fake_title(file_list: list, save_list: list, sim_filepath_dict: dict = None) -> None:
    '''
    make fake title using selected method
    '''

    for file_path, save_path in tqdm(zip(file_list, save_list), total=len(file_list)):

        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))
        #print(category_name)

        # load source file
        source_file = json.load(open(file_path, 'r'))
        
        # extract fake title
        #file_path = file_path[3:]
        sim_filepath = sim_filepath_dict[category_name][file_path]

        # target file
        target_file = json.load(open(sim_filepath, 'r'))
        fake_title = target_file['sourceDataInfo']['newsTitle']

        # update label infomation
        source_file = update_label_info(file=source_file, new_title=fake_title)

        # save source file
        json.dump(
            obj          = source_file, 
            fp           = open(save_path, 'w', encoding='utf-8'), 
            indent       = '\t',
            ensure_ascii = False
        )

def main(args):
    # index_dir = os.path.join('../data-direct', args.index_dir)
    # with open(f"../data-direct/{args.index_dir}", 'r') as f:
    with open(args.index_dir, 'r') as f:
        sim_index_dict = json.load(f)
    print(">>> Load sim_index_dict", len(sim_index_dict))
    # load file list
    file_list = glob(os.path.join(args.data_dir, '*/Auto/*/*')) #auto 불러옴
    save_list = [p.replace(args.data_dir, args.save_dir) for p in file_list]
    print(len(file_list), len(save_list))
    print(file_list[0])
    print(save_list[0])
    
    # make directory to save files
    partition_path = glob(os.path.join(args.data_dir, '*/Auto/*'))
    partition_path = [p.replace(args.data_dir, args.save_dir) for p in partition_path]
    for path in partition_path:
        os.makedirs(path, exist_ok=True)    
    
    # make fake title
    make_fake_title(file_list, save_list, sim_filepath_dict = sim_index_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, default='sim_index_content_content.json')
    parser.add_argument('--data_dir', type=str, default='../../data/Bait/original')
    parser.add_argument('--save_dir', type=str, default='../../data/Bait/tfidf_content_content')

    args = parser.parse_args()
    main(args)
