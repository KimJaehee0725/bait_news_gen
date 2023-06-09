from KoBERTScore.KoBERTScore import BERTScore
from transformers import AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

import pandas as pd
import torch
import argparse
import os
from tqdm.auto import tqdm

def run(args):
    def get_bertscore(ref, cand) :
        bert_scorer = BERTScore(model_name_or_path = args.model_name)
        score = bert_scorer.score(ref, cand)
        return score

    df = pd.read_csv(args.data_path)
    df = df.set_index('news_id')
    for sim_id, news_id in tqdm(zip(df['sim_news_id'].tolist(), df.index), total = len(df), desc=">>> preprocessing data frame"):
        df.loc[news_id, 'sim_news_title'] = df.loc[sim_id, 'original_title']
        df.loc[news_id, 'sim_news_content'] = df.loc[sim_id, 'content']
    
    ## 1. (ref) original content vs (cand) original title
    print(">>> Original Content vs Original Title")
    score = get_bertscore(df['content'].tolist(), df['original_title'].tolist())
    df['org_org_bertscore'] = score
    print(f">>> Original Content vs Original Title : {torch.mean(torch.Tensor(score))}")
    ## 2. (ref) original content vs (cand) fake title
    print(">>> Original Content vs Fake Title")
    score = get_bertscore(df['content'].tolist(), df['bait_title'].tolist())
    df['org_fake_bertscore'] = score
    print(f">>> Original Content vs Fake Title : {torch.mean(torch.Tensor(score))}")
    ## 3. (ref) similar content vs (cand) similart title
    print(">>> Similar Content vs Similar Title")
    score = get_bertscore(df['sim_news_content'].tolist(), df['sim_news_title'].tolist())
    df['sim_sim_bertscore'] = score
    ## 4. (ref) similar content vs (cand) fake title
    print(">>> Similar Content vs Fake Title")
    score = get_bertscore(df['sim_news_content'].tolist(), df['bait_title'].tolist())
    df['sim_fake_bertscore'] = score
    print(f">>> Similar Content vs Fake Title : {torch.mean(torch.Tensor(score))}")

    df['news_id'] = df.index
    # sort column names
    # df = df[['news_id', 'original_title', 'original_content',  'sim_news_id', 'sim_news_title', 'sim_news_content', 'fake_title', 'org_org_bertscore', 'org_fake_bertscore', 'sim_sim_bertscore', 'sim_fake_bertscore']]
    df.index = range(len(df))
    save_path = os.path.join(args.data_path, os.path.basename(args.data_path).split('.')[0])
    df.to_csv(save_path+'_eval.csv', index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "/workspace/codes/02_DSBA_Project/bait_news_gen/data/Fake/content_chunking_forward/generated/fake_top3.csv")
    args = parser.parse_args()

    args.model_name = 'klue/roberta-large'

    run(args)