from KoBERTScore.KoBERTScore import BERTScore
from transformers import AutoTokenizer

import pandas as pd
import torch
import argparse
import os

def run(args):
    #tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    df = pd.read_csv(args.data_path)
    refs = df['original_title'].tolist()
    preds = df['fake_title'].tolist()

    bert_scorer = BERTScore(model_name_or_path = args.model_name)
    score = bert_scorer.score(refs, preds)
    df['filter_bertscore'] = score
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
    # df_filtered = df[args.threshold_under < df['BERTScore'] < args.threshold_upper]
    df_filtered = df[args.threshold_under < df['filter_bertscore'] ][df['filter_bertscore'] < args.threshold_upper]

    #---- save score as csv
    os.makedirs(args.save_path, exist_ok = True)
    save_path = os.path.join(args.save_path, os.path.basename(args.data_path).split('.')[0])
    df_filtered.to_csv(save_path+'_'+str(int(args.threshold_under*100))+'_'+str(int(args.threshold_upper*100))+'.csv', index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "/workspace/code/bait_news_gen/data/Fake/tfidf_content_content_all/generated/fake_top3.csv")
    parser.add_argument("--save_path", type=str, default = "/workspace/code/bait_news_gen/data/Fake/tfidf_content_content_all/filtered")
    # parser.add_argument("--file_name", type=str, default = "fake_top3.csv")
    parser.add_argument("--threshold_under", type=float, default = 0.0)
    parser.add_argument("--threshold_upper", type=float, default = 0.99)
    args = parser.parse_args()

    args.max_word_len = 128
    args.model_name = 'klue/roberta-large'

    run(args)