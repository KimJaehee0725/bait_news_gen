from KoBERTScore.KoBERTScore import BERTScore
from transformers import AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

import pandas as pd
import torch
import argparse
import os
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import logging



COLUMN_LIST = ['original_title', 'original_content', 'sim_news_id', 'fake_title', 'category', 'label', 'sim_news_content', 'sim_news_title', 'filter_bertscore']


def sampling_data(file_df, sample_size = 50_000) :
    """
    train : 40_000
    validation : 13_000
    test : 13_000
    """
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
    sampler = sampler.split(file_df, file_df['category'])
    dropped_idx, selected = next(sampler)
    df = file_df.iloc[selected].reset_index(drop=True)

    return df


def run(args):
    def get_bertscore(ref, cand) :
        bert_scorer = BERTScore(model_name_or_path = args.model_name)
        score = bert_scorer.score(ref, cand)
        return score

    df = pd.read_csv(args.data_path)
    df = df.set_index('news_id')

    assert sum(col in COLUMN_LIST for col in df.columns) == len(COLUMN_LIST), ">>> Column names are not matched"
    # df.index = df.news_id 

    # for sim_id, news_id in tqdm(zip(df['sim_news_id'].tolist(), df.index), total = len(df), desc=">>> preprocessing data frame"):
    #     df.loc[news_id, 'sim_news_title'] = df.loc[sim_id, 'original_title']
    #     df.loc[news_id, 'sim_news_content'] = df.loc[sim_id, 'original_content']
    
    df = sampling_data(df, sample_size = 50_000)
        
    ## 1. (ref) original content vs (cand) original title
    print(">>> Original Content vs Original Title")
    score_org_org = get_bertscore(df['original_content'].tolist(), df['original_title'].tolist())
    df['org_org_bertscore'] = score_org_org
    ## 2. (ref) original content vs (cand) fake title
    print(">>> Original Content vs Fake Title")
    score_org_fake = get_bertscore(df['original_content'].tolist(), df['fake_title'].tolist())
    df['org_fake_bertscore'] = score_org_fake
    ## 3. (ref) similar content vs (cand) similart title
    print(">>> Similar Content vs Similar Title")
    score_sim_sim = get_bertscore(df['sim_news_content'].tolist(), df['sim_news_title'].tolist())
    df['sim_sim_bertscore'] = score_sim_sim
    ## 4. (ref) similar content vs (cand) fake title
    print(">>> Similar Content vs Fake Title")
    score_sim_fake = get_bertscore(df['sim_news_content'].tolist(), df['fake_title'].tolist())
    df['sim_fake_bertscore'] = score_sim_fake

    print(f">>> Original Content vs Original Title : {round(torch.mean(torch.Tensor(score_org_org)).item(), 4)}")
    print(f">>> Original Content vs Fake Title : {round(torch.mean(torch.Tensor(score_org_fake)).item(), 4)}")
    print(f">>> Similar Content vs Similar Title : {round(torch.mean(torch.Tensor(score_sim_sim)).item(), 4)}")
    print(f">>> Similar Content vs Fake Title : {round(torch.mean(torch.Tensor(score_sim_fake)).item(), 4)}")

    # print(f">>> Top 5 Score Examples")
    # index = torch.sort(torch.Tensor(score), descending = True)[1]
    # for i in range(5):
    #     index_ = index[i].item()
    #     print(f"Ref : {refs[index_]}")
    #     print(f"Pred : {preds[index_]}")
    #     print(f"Score : {score[index_]}")
    #     print("="*50)
    # print("\n")

    # print(">>> Lowest 5 Score Examples")
    # for i in range(5) :
    #     index_ = index[-i-1].item()
    #     print(f"Ref : {refs[index_]}")
    #     print(f"Pred : {preds[index_]}")
    #     print(f"Score : {score[index_]}")
    #     print()
    
    # #---- filter by threshold
    # df_filtered = df[df['BERTScore'] < args.threshold]

    # #---- save score as csv
    # os.makedirs(args.save_path, exist_ok = True)
    # save_path = os.path.join(args.save_path, os.path.splitext(args.data_path)[0])
    # df_filtered.to_csv(save_path+'_filtered.csv', index = False) 
    # df['news_id'] = df.index
    # sort column names
    # df = df[COLUMN_LIST]
    # df.index = range(len(df))
    df.to_csv(args.data_path[:-4] + "sampled.csv", index = False)

    result_path = args.data_path[:-4] + "sampled_result.csv"
    with open(result_path, "w") as f:
        f.write(f"Original Content vs Original Title : {round(torch.mean(torch.Tensor(score_org_org)).item(), 4)}\n")
        f.write(f"Original Content vs Fake Title : {round(torch.mean(torch.Tensor(score_org_fake)).item(), 4)}\n")
        f.write(f"Similar Content vs Similar Title : {round(torch.mean(torch.Tensor(score_sim_sim)).item(), 4)}\n")
        f.write(f"Similar Content vs Fake Title : {round(torch.mean(torch.Tensor(score_sim_fake)).item(), 4)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "../data/Fake/tfidf/generated/fake_top3.csv")
    args = parser.parse_args()

    args.model_name = 'klue/roberta-large'
    args.sort = "News_Direct"

    run(args)