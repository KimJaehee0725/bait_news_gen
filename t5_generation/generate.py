from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5TokenizerFast
import json
from glob import glob
import os
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, load_metric
import nltk
import numpy as np
import argparse
from methods import generation
import torch
from accelerate import Accelerator

accelerator = Accelerator()

def main(args):

    with open(args.index_dir, 'r') as f:
        index = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model = accelerator.prepare(model)
    
    if args.method=='summarization':
        sum_model_path = "lcw99/t5-base-korean-text-summary"
        sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_path)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_path)
        sum_model = accelerator.prepare(sum_model)

    else:
        sum_tokenizer = None
        sum_model = None       
    
    prefix = 'summarize: '
    
    args.save_dir = os.path.join(args.save_dir, '{}/{}/{}_{}_{}'.format(args.model_name, args.index_name, args.use_metadata, args.method, args.direction))
    
    # load file list
    file_list = glob(os.path.join(args.data_dir, '[!sample]*/Auto/*/*'))
    save_list = [p.replace(args.data_dir, args.save_dir) for p in file_list]
    
    
    # make directory to save files
    partition_path = glob(os.path.join(args.data_dir, '[!sample]*/Auto/*'))
    partition_path = [p.replace(args.data_dir, args.save_dir) for p in partition_path]
    for path in partition_path:
        os.makedirs(path, exist_ok=True)    
    
    category_list = list(index.keys())
    for category in tqdm(category_list, desc='category iteration'):
        source_list = []
        target_list = []
        source_title_list = []
        target_title_list = []
        source_content_list = []
        target_content_list = []
        
        source_file_list = []
        
        for source, target in tqdm(index[category].items(), desc = 'load_file iteration'):
            source_file = json.load(open(source, 'r'))
            target_file = json.load(open(target, 'r'))
            
            source_file_list.append(source_file)
            
            source_title = source_file['sourceDataInfo']['newsTitle']
            source_content = source_file['sourceDataInfo']['newsContent']
            
            target_title = target_file['sourceDataInfo']['newsTitle']
            target_content = target_file['sourceDataInfo']['newsContent']
            
            source_list.append(source)
            target_list.append(target)
            source_title_list.append(source_title)
            target_title_list.append(target_title)
            source_content_list.append(source_content)
            target_content_list.append(target_content)

        generated_title_list = generation(source_title_list,
                                            source_content_list,
                                            target_title_list,
                                            target_content_list,
                                            prefix,
                                            model,
                                            tokenizer,
                                            accelerator,
                                            sum_model,
                                            sum_tokenizer,
                                            batch_size=args.batch_size,
                                            use_metadata=args.use_metadata,
                                            method=args.method,
                                            direction=args.direction,
                                            max_input_length=args.max_input_length)

        for source_file, source, generated_title in tqdm(zip(source_file_list, source_list, generated_title_list), desc='save_title iteration'):
            source_file = update_label_info(file=source_file, new_title=generated_title)
            
            # save source file
            save_path = save_list[file_list.index(source)]
            json.dump(
                obj          = source_file, 
                fp           = open(save_path, 'w', encoding='utf-8'), 
                indent       = '\t',
                ensure_ascii = False
            )
                

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


if __name__== "__main__":  
    
    parser = argparse.ArgumentParser()
    
    # data path
    parser.add_argument("--data_dir", type=str, default='../data/Part1') 
    parser.add_argument("--save_dir", type=str, default='../data-Auto')
    parser.add_argument("--index_dir", type=str, default='../data-Auto/index/sim_index_tfidf_content.json')
    
    # model
    parser.add_argument("--model_name", type=str, default='t5_base')
    parser.add_argument("--index_name", type=str, default='tfidf_content')
    parser.add_argument("--model_path", type=str, default='./Models/ke-t5-base-newslike_epoch7')
    
    # generation parameters
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--use_metadata", type=str, default='content')
    parser.add_argument("--method", type=str, default='chunking')
    parser.add_argument("--direction", type=str, default='backward')
    parser.add_argument("--batch_size", type=int, default=12)
    # parser.add_argument("--num_beams", type=int, default=8)
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)