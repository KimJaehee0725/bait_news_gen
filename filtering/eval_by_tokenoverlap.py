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
    if 'content' in df.columns:
        contents = df['content'].tolist()
    if 'original_content' in df.columns:
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

# 제목과 본문내 각 문장 사이의 BERTScore를 구함
def get_max_score(df, real=True):
    print(">>> Get Max Score")
    max_score = []
    content_n_list = []
    title_n_list = []
    len_list = []
    for idx in tqdm(range(len(df))):
        content_sentences = kss.split_sentences(df.loc[idx, 'original_content'])
        content_n_list += content_sentences
        if real == True:
            title_duplicated = [df.loc[idx, 'original_title']] * len(content_sentences)
        else:
            title_duplicated = [df.loc[idx, 'fake_title']] * len(content_sentences)
        title_n_list += title_duplicated
        len_list.append(len(content_sentences))
    bert_scorer = BERTScore(model_name_or_path = 'klue/roberta-large')
    score = bert_scorer.score(content_n_list, title_n_list, batch_size=64)
    
    start_idx = 0
    for length in tqdm(len_list):
        max_score.append(max(score[start_idx : start_idx+length]))
        start_idx += length
    df['max_score'] = max_score
    return df


def run(args):
    metrics = {}
    df = pd.read_csv(args.data_path)
    df_eval = find_token_for_fake_news(df)
    if args.BertScore == True:
        df_eval = get_max_score(df_eval, real=args.real)
    
    # -------filtering
    df_save_path = os.path.join('../data/Fake',args.method, 'filtered', os.path.basename(args.data_path))
    df_filtered = df_eval[df_eval['fake']==1]
    df_filtered.to_csv(df_save_path+'_token.csv', index = False) 
    
    # -------save results
    fake_news_cnt = len(df_filtered)
    if args.BertScore == True:
        bert_score_mean = df_eval['max_score'].mean()
        metrics['bert_score_mean'] = bert_score_mean

    metrics['false_negative'] = len(df) - fake_news_cnt
    metrics['fake_news'] = fake_news_cnt

    json.dump(metrics, open(os.path.join(args.savedir, f"exp_metrics.json"),'w'), indent='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bait News Generation')

    parser.add_argument('--method', type=str, default='content_chunking_backward')
    parser.add_argument('--file_name', type=str, default='fake_top1.csv')
    parser.add_argument('--real', type=bool, default=False)
    parser.add_argument('--BertScore', type=bool, default=False)
    
    args = parser.parse_args()
    # file_path에 맞춰줘야 함.
    args.data_path = os.path.join('/workspace/code/bait_news_gen/data/Fake',args.method, 'generated', args.file_name)
    args.savedir = os.path.join('/workspace/code/bait_news_gen/data/Fake', args.method, 'eval_FN')
    os.makedirs(args.savedir, exist_ok=True)
    run(args)

