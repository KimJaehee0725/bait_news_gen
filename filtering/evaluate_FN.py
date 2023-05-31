import pandas as pd
from konlpy.tag import Mecab
from tqdm import tqdm
import re
import json
import argparse
import os
from KoBERTScore.KoBERTScore import BERTScore
import kss
 
def clean_text(text):
  text_removed = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', text)
  return text_removed

def find_token_for_fake_news(df):
    mecab = Mecab()
    titles = df['fake_title'].tolist()
    contents = df['original_content'].tolist()

    title_n_list = []
    content_n_list = []
    for idx in tqdm(range(len(titles)), desc=f'Extract Morphs', total=len(titles), leave=False):
        title_cleaned = clean_text(titles[idx])
        content_cleaned = clean_text(contents[idx])
        title_n_list.append([m for m in mecab.morphs(title_cleaned) if len(m) > 1])
        content_n_list.append([m for m in mecab.morphs(content_cleaned) if len(m) > 1])

    print(">>> Find Token for Fake News")
    fake = []
    tokens_for_fakenews = []
    for title, content in zip(title_n_list, content_n_list):
        fake_token = ''
        f = 0 #flag for no false negative
        for token in title:
            # 가설 : 가짜 제목이면 본문에 없는 단어가 포함될 것이다.
            content_joined = ' '.join(content)
            if content_joined.count(token) == 0:
                f = 1
                fake_token += token + ' '
        if f == 1:
            fake.append(1)
        else:
            fake.append(0)
        tokens_for_fakenews.append(fake_token)
    df['fake'] = fake
    df['tokens_for_fake'] = tokens_for_fakenews
    return df

def get_max_score(df, real=True):
    print(">>> Get Max Score")
    max_score = []
    for idx in tqdm(range(len(df))):
        content_sentences = kss.split_sentences(df.loc[idx, 'original_content'])
        if real == True:
            title_duplicated = [df.loc[idx, 'original_title']] * len(content_sentences)
        else:
            title_duplicated = [df.loc[idx, 'fake_title']] * len(content_sentences)
        bert_scorer = BERTScore(model_name_or_path = 'klue/roberta-large')
        score = bert_scorer.score(title_duplicated, content_sentences)
        max_score.append(max(score))
    df['max_score'] = max_score
    return df


def run(args):
    df = pd.read_csv(args.data_path)

    df_eval_token = find_token_for_fake_news(df)
    df_eval_bertscore = get_max_score(df_eval_token, real=args.real)
    df_eval_bertscore.to_csv(os.path.join(args.savedir, f"exp.csv"), index=False)
    
    fake_news_cnt = len(df[df_eval_bertscore['fake']==1])
    bert_score_mean = df_eval_bertscore['max_score'].mean()
    
    metrics = {
        'false_negative' : len(df) - fake_news_cnt,
        'fake_news' : fake_news_cnt,
        'bert_score_mean' : bert_score_mean
    }
    json.dump(metrics, open(os.path.join(args.savedir, f"exp_metrics.json"),'w'), indent='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bait News Generation')
    parser.add_argument('--data_path', type=str, default='../data/Fake/content_rotation_forward/filtered/fake_top1_90_99.csv')
    parser.add_argument('--savedir', type=str, default='../data/Fake/content_rotation_forward/evaluation_FN')
    parser.add_argument('--real', type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    run(args)

