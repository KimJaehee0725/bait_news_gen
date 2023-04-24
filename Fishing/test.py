import numpy as np
import wandb
import json
import logging
import os
import torch
import argparse
import yaml

from transformers import AutoConfig
from kobert_transformers import get_tokenizer

from transformers import get_cosine_schedule_with_warmup
from train import training, evaluate

from log import setup_default_logging
from model import BERT
from dataset import BaitDataset
from torch.utils.data import DataLoader

import pandas as pd

_logger = logging.getLogger('train')


def torch_seed(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def run(cfg):
    # setting seed and device
    setup_default_logging()
    torch_seed(223)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    #* save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['sort'])
    os.makedirs(savedir, exist_ok=True)

    #* make TRAIN data
    tokenizer = get_tokenizer() #monologg/kobert
    trainset = BaitDataset(
            cfg['DATASET'],
            'train',
            tokenizer = tokenizer
        )
    validset = BaitDataset(
            cfg['DATASET'],
            'validation',
            tokenizer = tokenizer
        )
    
    _logger.info('load raw data completely')


    #* load MODEL -------------------

    model_config = AutoConfig.from_pretrained('monologg/kobert')
    model = BERT( # model.py class
        config          = model_config,
        num_classes     = 2
    )
    model.load_state_dict(torch.load(cfg['TEST']['saved_model_path'])) # load pre-trained model
    model.to(device)
    

    #* TEST -------------------
    _logger.info('TEST start')

    for split in cfg['MODE']['test_list']:
        _logger.info('{} evaluation'.format(split.upper()))
        
        if split == 'train':
            dataset = trainset
        elif split == 'validation':
            dataset = validset
        else:
            dataset = BaitDataset(
                cfg['DATASET'],
                split,
                tokenizer = tokenizer
            )

        testloader = DataLoader(
            dataset, 
            batch_size  = cfg['TEST']['batch_size']
        )

        #* testing Model
        
        criterion = torch.nn.CrossEntropyLoss()

        metrics, exp_results = evaluate(
            model        = model, 
            dataloader   = testloader, 
            criterion    = criterion,
            log_interval = cfg['TEST']['log_interval'],
            device       = device,
            sample_check = True
        )
                
        # save exp result
        exp_results = pd.concat([pd.DataFrame({'filename':list(dataset.original_file_path.values())}), pd.DataFrame(exp_results)], axis=1)
        exp_results['label'] = exp_results['filename'].apply(lambda x: 1 if ('Direct'or'Auto') in x else 0)
        exp_results.to_csv(os.path.join(savedir, f'exp_results_{split}.csv'), index=False)

        # save result metrics
        json.dump(metrics, open(os.path.join(savedir, f"{split}.json"),'w'), indent='\t')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bait News Generation')
    parser.add_argument('--base_config', type=str, default=None, help='exp config file')    
    parser.add_argument('--bait_path', type=str, default=None, help='bait path')
    parser.add_argument('--sort', type=str, default=None, help='sort')
    parser.add_argument('--saved_model_path', type=str, default=None, help='saved_model_path')
    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    cfg['DATASET']['bait_path'] = args.bait_path
    cfg['DATASET']['sort'] = args.sort
    cfg['TEST']['saved_model_path'] = args.saved_model_path

    print("Config:")
    print(json.dumps(cfg, indent=2))
    run(cfg)