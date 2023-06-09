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
import pandas as pd
from accelerate import Accelerator

accelerator = Accelerator()

def main(args):    
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
    
    # load data
    fake_original = pd.read_csv('../../data/Fake/fake_original.csv')
    fake_tfidf = pd.read_csv('../../data/Fake/tfidf/generated/fake_{}.csv'.format(args.index_rank))
    
    generated_title_list = generation(
        fake_tfidf['original_title'].to_list(),
        fake_tfidf['original_content'].to_list(),
        fake_tfidf['sim_news_title'].to_list(),
        fake_tfidf['sim_news_content'].to_list(),
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
        max_input_length=args.max_input_length
        )
    
    fake_original['fake_title'] = generated_title_list
    fake_original['sim_news_content'] = fake_tfidf['sim_news_content'].to_list()
    fake_original['sim_news_id'] = fake_tfidf['sim_news_id'].to_list()
    fake_original['sim_news_title'] = fake_tfidf['sim_news_title'].to_list()

    os.makedirs('../../data/Fake/{}_{}_{}/generated'.format(args.use_metadata, args.method, args.direction), exist_ok = True)
    fake_original.to_csv('../../data/Fake/{}_{}_{}/generated/fake_{}.csv'.format(args.use_metadata, args.method, args.direction, args.index_rank), index=False)

if __name__== "__main__":  
    parser = argparse.ArgumentParser()
    
    # data path
    parser.add_argument("--index_rank", type=str, default='top3') 
    
    # model path
    parser.add_argument("--model_path", type=str, default='../finetuning/Models/ke-t5-base-newslike_epoch7_30k')
    
    # generation parameters
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--use_metadata", type=str, default='content')
    parser.add_argument("--method", type=str, default='chunking')
    parser.add_argument("--direction", type=str, default='backward')
    parser.add_argument("--batch_size", type=int, default=12)
    # parser.add_argument("--num_beams", type=int, default=8)
    args = parser.parse_args()
    print(args)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)