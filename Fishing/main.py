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
    tokenizer = get_tokenizer() # monologg/kobert
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

    trainloader = DataLoader(
        trainset,
        batch_size = cfg['TRAIN']['batch_size'],
        shuffle = True
    )
    validloader = DataLoader(
        validset,
        batch_size = cfg['TEST']['batch_size'],
        shuffle = False
    )

    #* TRAIN -------------------

    model_config = AutoConfig.from_pretrained('monologg/kobert')
    model = BERT( # model.py class
        config          = model_config,
        num_classes     = 2
    )
   
    model.to(device)

    _logger.info('# of trainable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    # wandb
    if cfg['TRAIN']['use_wandb']:
        wandb.init(
            name=os.path.join(cfg['DATASET']['bait_path'].split('/')[-1], cfg['DATASET']['sort']), 
            project='Bait-News-Detection', 
            config=cfg
            )

    # Set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()), 
        lr           = cfg['TRAIN']['OPTIMIZER']['lr'], 
        weight_decay = cfg['TRAIN']['OPTIMIZER']['weight_decay']
    )

    # scheduler
    if cfg['TRAIN']['SCHEDULER']['use_scheduler']:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps   = int(cfg['TRAIN']['num_training_steps'] * cfg['TRAIN']['SCHEDULER']['warmup_ratio']), 
            num_training_steps = cfg['TRAIN']['num_training_steps'])
    else:
        scheduler = None


    #* fitting Model
    _logger.info('TRAIN start')
        
    train_model = training(
        model              = model, 
        num_training_steps = cfg['TRAIN']['num_training_steps'], 
        trainloader        = trainloader, 
        validloader        = validloader, 
        criterion          = criterion, 
        optimizer          = optimizer, 
        scheduler          = scheduler,
        log_interval       = cfg['TRAIN']['log_interval'],
        eval_interval      = cfg['TRAIN']['eval_interval'],
        savedir            = savedir,
        accumulation_steps = cfg['TRAIN']['accumulation_steps'],
        device             = device,
        use_wandb          = cfg['TRAIN']['use_wandb']
    )

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
        metrics, exp_results = evaluate(
            model        = train_model, 
            dataloader   = testloader, 
            criterion    = criterion,
            log_interval = cfg['TEST']['log_interval'],
            device       = device,
            sample_check = True
        )
                
        # save exp result
        exp_results = pd.concat([pd.DataFrame({'filename':list(dataset.original_file_path.values())}), pd.DataFrame(exp_results)], axis=1)
        exp_results['label'] = exp_results['filename'].apply(lambda x: 1 if ('Direct' in x) or ("Auto" in x) else 0)
        exp_results.to_csv(os.path.join(savedir, f'exp_results_{split}.csv'), index=False)

        # save result metrics
        json.dump(metrics, open(os.path.join(savedir, f"{split}.json"),'w'), indent='\t')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bait News Generation')
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml', help='exp config file')    
    parser.add_argument('--bait_path', type=str, default='../data/generated/tfidf_avg_category_select')
    parser.add_argument('--sort', type=str, default='News_Direct')

    args = parser.parse_args()

    cfg = yaml.load(open(args.base_config,'r'), Loader=yaml.FullLoader)
    cfg['DATASET']['bait_path'] = args.bait_path
    cfg['DATASET']['sort'] = args.sort

    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['bait_path'].split('/')[-1], cfg['DATASET']['sort'])
    os.makedirs(savedir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(savedir, 'Logs.log')
        )

    _logger = logging.getLogger(__name__)

    # config
    _logger.info('Config:')
    _logger.info(json.dumps(cfg, indent=2))
    run(cfg)
