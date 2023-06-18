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
    
    #Fake/tfidf/train.csv, test.csv, val.csv
                              
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

    print('cfg[TEST][saved_model_path] : ', cfg['TEST']['saved_model_path'])

    if cfg['TEST']['saved_model_path'] == 'None':

        #* save directory Fake/auto
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0])
                                # 학습시 : results / 학습 모델 이름 / topN 종류
                                # 테스트시 : results / 학습 모델 이름 / 학습 데이터 topN 종류 / 테스트 데이터 이름 / 테스트 데이터 topN 종류
        print('model savedir : ', savedir)

        os.makedirs(savedir, exist_ok=True)

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
                name=os.path.join(cfg['DATASET']['fake_path'].split('/')[1]), 
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

        #* save directory Fake/auto
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0])
                                # 학습시 : results / 학습 모델 이름 / topN 종류
                                # 테스트시 : results / 학습 모델 이름 / 학습 데이터 topN 종류 / 테스트 데이터 이름 / 테스트 데이터 topN 종류
        print('result savedir : ', savedir)
        os.makedirs(savedir, exist_ok=True)

    else :        
        #* load MODEL -------------------

        #* save directory Fake/auto
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['TEST']['saved_model_path'].split('/')[-3], cfg['TEST']['saved_model_path'].split('/')[-2], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0])
                                # 학습시 : results / 학습 모델 이름 / topN 종류
                                # 테스트시 : results / 학습 모델 이름 / 학습 데이터 topN 종류 / 테스트 데이터 이름 / 테스트 데이터 topN 종류
        print('result savedir : ', savedir)
        os.makedirs(savedir, exist_ok=True)

        model_config = AutoConfig.from_pretrained('monologg/kobert')
        train_model = BERT( # model.py class
            config          = model_config,
            num_classes     = 2
        )
        train_model.load_state_dict(torch.load(cfg['TEST']['saved_model_path'])) # load pre-trained model
        train_model.to(device)

        criterion = torch.nn.CrossEntropyLoss()


    #* TEST -------------------
    _logger.info('TEST start')

    bait = cfg['DATASET']['fake_path'].split('/')[1]
    _logger.info(f'{bait} evaluation')
        
    testset = BaitDataset(
        cfg['DATASET'],
        'test',
        tokenizer = tokenizer
    )

    testloader = DataLoader(
        testset, 
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
    exp_results = pd.concat([pd.DataFrame({'news_id':testset.id_list}), pd.DataFrame(exp_results)], axis=1)
    exp_results['label'] = testset.label_list
    exp_results.to_csv(os.path.join(savedir, f'exp_results.csv'), index=False)

    # save result metrics
    json.dump(metrics, open(os.path.join(savedir, f"exp_metrics.json"),'w'), indent='\t')
        



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bait News Generation')
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml', help='exp config file')    
    parser.add_argument('--fake_path', type=str, default='Fake/content_summarization_forward/filtered')
    parser.add_argument('--fake_name', type=str, default='fake_top1_90_99.csv')
    parser.add_argument('--saved_model_path', type=str, default='None', help='saved_model_path')

    args = parser.parse_args()

    cfg = yaml.load(open(args.base_config,'r'), Loader=yaml.FullLoader)
    cfg['DATASET']['fake_path'] = args.fake_path
    cfg['DATASET']['fake_name'] = args.fake_name
    cfg['TEST']['saved_model_path'] = args.saved_model_path

    if cfg['TEST']['saved_model_path'] == 'None':
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0])
        # results / 학습 모델 이름 / topN 종류
    else:
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['TEST']['saved_model_path'].split('/')[-3], cfg['TEST']['saved_model_path'].split('/')[-2], cfg['DATASET']['fake_path'].split('/')[1], cfg['DATASET']['fake_name'].split('.')[0])
        # results / 학습 모델 이름 / topN 종류 / 테스트 데이터 이름 / topN 종류

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
