import numpy as np
import pandas as pd

import os
import json

from tqdm.auto import tqdm
from typing import List
from methods.tfidf import overlap_token

def score_overlap_acc(query_ids, token_overlap_sorted_idx):
    top_1_accuracy = 0.0
    for query_id, indices in zip(query_ids, token_overlap_sorted_idx):
        if query_id == indices[0]: #query와 document로 사용하는 text가 달라야 유의미함.
            top_1_accuracy += 1.0
    top_1_accuracy = round(top_1_accuracy/len(query_ids), 5)

    top_K_accuracy = 0.0
    for query_id, indices in zip(query_ids, token_overlap_sorted_idx):
        if query_id in indices:
            top_K_accuracy += 1.0
            
    top_K_accuracy = round(top_K_accuracy/len(query_ids), 5)
    return top_1_accuracy, top_K_accuracy

def score_tfidf_acc(sim_matrix, category):
    target_ids = list(range(len(sim_matrix)))
    top_1 = sum(np.argmax(sim_matrix, axis=1) == target_ids)
    top_1_accuracy = round(top_1/len(sim_matrix), 5)
    print(f"{category} TF-IDF ACC : {top_1_accuracy}")
    return top_1_accuracy


def check_overlap(tfidf, overlap_matrix, category, target_text_list, source_text_list):
    print(f"############ Category : {category} #################")
    # 1. tfidf top-1과 overlap top-1 비교
    same_tfidf_and_overlap = sum(np.argmax(tfidf, axis=1) == np.argmax(overlap_matrix, axis=1))
    diff_tfidf_and_overlap = sum(np.argmax(tfidf, axis=1) != np.argmax(overlap_matrix, axis=1))

    print('TFIDF top-1과 Overlap top-1이 같은 target의 개수 : {}'.format(same_tfidf_and_overlap))
    print('TFIDF top-1과 Overlap top-1이 다른 target의 개수 : {}'.format(diff_tfidf_and_overlap))

    # 2. # of token_overlap / # of token in source text
    prop = []
    intersection = []
    count_list = []
    for t_idx, target_text in tqdm(enumerate(target_text_list), desc='Count ovelap token', total=len(target_text_list)):
        # 2-1. 'token1 token2 ...' -> ['token1', 'token2',..] -> 중복 제거
        target_token_list = list(set(target_text.split())) 
        for s_idx in overlap_matrix[t_idx]:
            source_text = source_text_list[s_idx]
            cnt_list = [source_text.count(token) for token in target_token_list]
            count = sum(cnt_list)
            count_list.append(count)
            prop.append(count/len(source_text.split()))

            tmp = len(set(target_text.split())& set(source_text.split()))
            intersection.append(tmp)
    print('AVERAGE of (# of token_overlap / # of token in source text) : {}'.format(np.mean(prop)))
    print('AVERAGE of intersection : {}'.format(np.mean(intersection)))
    print("***********************************************")
    return same_tfidf_and_overlap, diff_tfidf_and_overlap, np.mean(count_list), np.mean(prop), np.mean(intersection)


def score_overlap(
    method_name : str, make_sim_matrix_func, extract_text_func, 
    file_list: list, category_list: list, target: str, savedir: str, source : str = None, top_k : int = None,) :

    # define progress bar
    pbar = tqdm(category_list, total=len(category_list))

    results = dict()
    for category in pbar:
        pbar.set_description(f'Category: {category}')
        results[category] = {}
        # extract file path in category
        file_list_cat = [f for f in file_list if category in f]

        # mask similarity matrix(sparse, n by n) or top-k rank matrix(dense, n by k)
        if source is None : # if target and source is the same.
            sim_matrix = make_sim_matrix_func(
                text    = extract_text_func(file_list=file_list_cat, target=target),
                target  = target
            )

        else : # if different texts are used as target and source
            target_text_list, source_text_list = extract_text_func(file_list = file_list_cat, target = target, source = source)
            sim_matrix = make_sim_matrix_func(
                target_text_list = target_text_list,
                source_text_list = source_text_list
            )
        
        #tfidf score 측정
        tfidf_acc = score_tfidf_acc(sim_matrix, category)

        if 'dense' not in method_name: 
            sim_matrix[np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[0])] = -1
            tfidf_sim_matrix = sim_matrix

        if 'overlap' in method_name:
            sim_matrix = overlap_token(sim_matrix, target_text_list, source_text_list, method_name, top_k)

        #TFIDF top1과 TFIDF-overlap top1이 얼마나 다른가
        same_tfidf_and_overlap, diff_tfidf_and_overlap, count, prop, intersection = check_overlap(tfidf_sim_matrix, sim_matrix, category, target_text_list, source_text_list)
        results[category] = {'tfidf' : tfidf_acc,
                            'same_comparison_tfidf_overlap' : same_tfidf_and_overlap,
                            'diff_comparison_tfidf_overlap' : diff_tfidf_and_overlap,
                            'count of overlap tokens' : count,
                            'prop. of overlap' : prop,
                            '# of intersection' : intersection
                            }
    
    return results