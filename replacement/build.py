from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random
import time
import pandas as pd

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


def make_fake_title(data : pd.DataFrame, savedir, top_k, sim_filepath_dict: dict = None) -> None:
    '''
    make fake title using selected method
    '''
    data['bait_content'] = pd.Series()
    for category, src_sim_pairs_dict in tqdm(sim_filepath_dict.items()):
        for src_path, similar_path in tqdm(src_sim_pairs_dict.items()):
            data.loc[data['news_id'] == src_path, 'sim_news_id']  = similar_path
            data.loc[data['news_id'] == src_path, 'bait_title']   = data.loc[data['news_id'] == similar_path, 'original_title'].values[0]
            data.loc[data['news_id'] == src_path, 'bait_content'] = data.loc[data['news_id'] == similar_path, 'content'].values[0]
    data.to_csv(os.path.join(savedir, f'fake_title_{top_k}.csv'), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename', default='/workspace/code/bait_news_gen/replacement/configs/tfidf/full_full.yaml')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg['SEED'])

    # update save directory
    os.makedirs(os.path.join(cfg['savedir'], cfg["METHOD"]["select_name"]), exist_ok=True)
    cfg['savedir'] = os.path.join(cfg['savedir'], cfg["METHOD"]["select_name"])

    os.makedirs(os.path.join(cfg['savedir'], 'generated'), exist_ok=True)
    cfg['savedir'] = os.path.join(cfg['savedir'], 'generated')
    
    # load file list
    df = pd.read_csv(os.path.join(cfg['datadir'], 'fake.csv'))
    print(">>> Load dataset, size : ",len(df)) 

    #find article index most similar to article and save indices
    file_list = df['news_id'].tolist()
    sim_filepath_dict = None
    if cfg['METHOD']['name'] != 'random':
        sim_filepath_dict = get_similar_filepath_dict(
            method_name          = cfg['METHOD']['name'],
            make_sim_matrix_func = __import__('methods').__dict__["tfidf_sim_matrix"] if 'overlap' in cfg['METHOD']['name'] \
                                else __import__('methods').__dict__[f"{cfg['METHOD']['name']}_sim_matrix"],
            extract_text_func    = extract_text if ('dense' in cfg['METHOD']['name']) or (cfg['METHOD']['extract'] == 'all') else extract_nouns,
            data                 = df,     
            file_list            = file_list,    
            category_list        = df['category'].unique().tolist(),
            target               = cfg['METHOD']['target'],
            source               = cfg['METHOD']['source'],
            savedir              = cfg['savedir'],
            top_k                = cfg['METHOD']['topk']
        )

    json.dump(sim_filepath_dict, open(os.path.join(cfg['savedir'], 'sim_filepath_dict.json'), 'w'), indent=4)

    # run
    make_fake_title(
        data              = df,
        savedir           = cfg['savedir'],
        top_k             = cfg['METHOD']['topk'],
        sim_filepath_dict = sim_filepath_dict
    )