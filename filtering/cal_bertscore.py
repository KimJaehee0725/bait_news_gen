from KoBERTScore.KoBERTScore import BERTScore
from transformers import AutoTokenizer

import pandas as pd
import torch
import argparse
import os

def run(args):
    #tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    df = pd.read_csv(args.data_path)
    refs = df['content'].tolist()
    preds = df['bait_content'].tolist()

    bert_scorer = BERTScore(model_name_or_path = args.model_name)
    score = bert_scorer.score(refs, preds)
    df['BERTScore'] = score
    print(f">>> BERT Score : {torch.mean(torch.Tensor(score))}")
    
    print(f">>> Top 5 Score Examples")
    index = torch.sort(torch.Tensor(score), descending = True)[1]
    for i in range(5):
        index_ = index[i].item()
        print(f"Ref : {refs[index_]}")
        print(f"Pred : {preds[index_]}")
        print(f"Score : {score[index_]}")
        print("="*50)
    print("\n")

    print(">>> Lowest 5 Score Examples")
    for i in range(5) :
        index_ = index[-i-1].item()
        print(f"Ref : {refs[index_]}")
        print(f"Pred : {preds[index_]}")
        print(f"Score : {score[index_]}")
        print()
    
    #---- filter by threshold
    df_filtered = df[df['BERTScore'] < args.threshold]

    #---- save score as csv
    os.makedirs(args.save_path, exist_ok = True)
    save_path = os.path.join(args.save_path, os.path.splitext(args.data_path)[0])
    df_filtered.to_csv(f'{save_path}_{args.threshold}_content.csv', index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "/workspace/code/bait_news_gen/data/Fake/tfidf_full_full_all/generated/fake_top3.csv")
    parser.add_argument("--save_path", type=str, default = "/workspace/code/bait_news_gen/data/Fake/tfidf_full_full_all/filtered")
    parser.add_argument("--threshold", type=float, default = 1.0)
    args = parser.parse_args()

    args.max_word_len = 128
    args.model_name = 'klue/roberta-large'
    args.sort = "News_Direct"

    run(args)