from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random

from tqdm.auto import tqdm
from methods import get_similar_filepath_dict, extract_nouns, extract_text

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



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


def make_fake_title(file_list: list, save_list: list, cfg_method: dict,) -> None:
    '''
    make fake title using selected method
    '''

    for file_path, save_path in tqdm(zip(file_list, save_list), total=len(file_list)):

        # source file name and category
        category_name = os.path.basename(os.path.dirname(file_path))

        # load source file
        source_file = json.load(open(file_path, 'r'))
        
        # get false negative title
        fake_title = source_file['sourceDataInfo']['newsTitle']

        # update label infomation
        source_file = update_label_info(file=source_file, new_title=fake_title)
        
        # save source file
        json.dump(
            obj          = source_file, 
            fp           = open(save_path, 'w', encoding='utf-8'), 
            indent       = '\t',
            ensure_ascii = False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename', default='configs/false_negative.yaml')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg['SEED'])

    # update save directory
    cfg['savedir'] = os.path.join(cfg['savedir'], cfg['METHOD']['select_name'])

    # load file list
    file_list = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Auto/*/*'))
    save_list = [p.replace(cfg['datadir'], cfg['savedir']) for p in file_list]

    # make directory to save files
    partition_path = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Auto/*'))
    partition_path = [p.replace(cfg['datadir'], cfg['savedir']) for p in partition_path]

    ## train, validation 폴더 생성
    for path in partition_path:
        os.makedirs(path, exist_ok=True)    

    # run
    make_fake_title(
        file_list      = file_list, 
        save_list      = save_list, 
        cfg_method     = cfg['METHOD'],
    )
