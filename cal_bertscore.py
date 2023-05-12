from Fishing.dataset import BaitDataset
from KoBERTScore.KoBERTScore import BERTScore
from transformers import AutoTokenizer

import pandas as pd
import torch
import argparse

MODEL_NAME = 'klue/roberta-large'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bait_path", type=str, default = "/workspace/codes/02_DSBA_Project/bait_news_gen/joonghoon/t5_base/tfidf_content/content_chunking_forward")
    parser.add_argument("-data_path", type=str, default = "/workspace/codes/02_DSBA_Project/bait_news_gen/data/original")
    args = parser.parse_args()
    args.max_word_len = 128
    args.model_name = MODEL_NAME
    args.sort = "News_Direct"

    ## args to dictionary
    args = vars(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])

    dataset = BaitDataset(
        args,
        split = "train",
        tokenizer = tokenizer
        )
    
    bait_title, news_title, bait_file_path = dataset.load_bait_news_info(
        data_dir = args['data_path'],
        bait_dir = args['bait_path'], 
        split = "train")
    
    refs = list(news_title.values())
    preds = list(bait_title.values())

    bert_scorer = BERTScore(model_name_or_path = args['model_name'])
    score = bert_scorer.score(refs, preds)

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
    
    ## save score as csv
    path = list(bait_file_path.values())
    score_df = pd.DataFrame({"path" : path, "ref" : refs, "pred" : preds, "score" : score})
    score_df.to_csv("score.csv", index = False)

if __name__ == "__main__":
    main()