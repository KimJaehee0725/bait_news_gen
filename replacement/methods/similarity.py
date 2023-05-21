import numpy as np
import pandas as pd
import os
import json

from tqdm.auto import tqdm
from konlpy.tag import Mecab
from typing import List


def get_similar_filepath_dict(
    method_name : str, make_sim_matrix_func, extract_text_func, 
    data : pd.DataFrame, file_list: list, category_list: list, target: str, savedir: str, source : str = None, top_k : int = None) -> None:

    # load sim_filepath_dict
    sim_filepath_dict = dict()

    # define progress bar
    pbar = tqdm(category_list, total=len(category_list))

    for category in pbar:
        pbar.set_description(f'Category: {category}')

        # run if category not in sim_filepath_dict
        if not category in sim_filepath_dict.keys():
            # set default
            sim_filepath_dict[category] = dict()

            # extract file path in category
            file_list_cat = [f for f in file_list if category in f]

            # mask similarity matrix(sparse, n by n) or top-k rank matrix(dense, n by k)
            if source is None : # if target and source is the same.
                sim_matrix = make_sim_matrix_func(
                    text    = extract_text_func(data = data, file_list=file_list_cat, target=target),
                    target  = target
                )

            else : # if different texts are used as target and source
                target_text_list, source_text_list = extract_text_func(data = data, file_list = file_list_cat, target = target, source = source)
                sim_matrix = make_sim_matrix_func(
                    target_text_list = target_text_list,
                    source_text_list = source_text_list
                )
            
            # masking
            if 'dense' not in method_name: 
                sim_matrix[np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[0])] = -1

            # update similarity matrix
            sim_filepath_dict = extract_sim_filepath(
                method_name       = method_name,
                sim_matrix        = sim_matrix,
                file_list         = file_list_cat,
                category          = category,
                sim_filepath_dict = sim_filepath_dict,
                top_k             = top_k
            )

    return sim_filepath_dict


def extract_sim_filepath(
    method_name : str, sim_matrix: np.ndarray, file_list: list, category: str, sim_filepath_dict: dict, top_k: int = 3) -> None:
    """
    extract filepath most similar to filepath1 using ngram similarity
    """
    # find argmax
    if top_k == 1:
        if ('overlap' in method_name) or ('dense' in method_name) :
            sim_index = list(sim_matrix[:, 0])
        else :
            sim_index = sim_matrix.argmax(axis=1)
    # get top-k indices
    else :
        sim_index = np.argsort(sim_matrix, axis=1)[:,-top_k]
    
    # update sim_filepath_dict
    for file_path, idx in zip(file_list, sim_index):
        sim_filepath_dict[category][file_path] = file_list[idx]
    
    return sim_filepath_dict


def extract_nouns(file_list: list, target: str, source : str = None, join: bool = True) -> List[list]:
    """
    extract nouns from target text
    
    """
    # extract morphs
    mecab = Mecab()

    # define list
    target_n_list = []
    source_n_list = []

    if source == None : 
        target_list = extract_text(file_list=file_list, target=target, source=source)
    else :
        target_list, source_list = extract_text(file_list=file_list, target=target, source=source)


    for idx in tqdm(range(len(target_list)), desc=f'Extract Nouns({target})', total=len(file_list), leave=False):
        if join:
            if source != None:
                source_n_list.append(' '.join(mecab.nouns(source_list[idx])))
            target_n_list.append(' '.join(mecab.nouns(target_list[idx])))
        else:
            if source != None:
                source_n_list.append(mecab.nouns(source_list[idx]))
            target_n_list.append(mecab.nouns(target_list[idx]))

    return target_n_list if source is None else (target_n_list, source_n_list) #['token1 token2 ... tokenN', ...]


def extract_text(data : pd.DataFrame, file_list: list, target: str, source : str = None) -> List[list]:
    """
    extract target text
    """

    # define list
    target_list = []
    if source is not None :
        source_list = []

    for file_path in tqdm(file_list, desc=f'Extract Morphs({target})', total=len(file_list), leave=False):
        # load source file
        # source_file = json.load(open(file_path, "r"))
        if target == 'title':
            target_text = data[data['news_id']==file_path]['original_title'].values[0]
        elif target == 'content':
            target_text = data[data['news_id']==file_path]['content'].values[0]
        elif target == 'title-content' :
            target_text = data[data['news_id']==file_path]['original_title'].values[0] + " " + data[data['news_id']==file_path]['content'].values[0]
        
        target_list.append(target_text)
        
        if source is not None :
            if source == 'title':
                source_text = data[data['news_id']==file_path]['original_title'].values[0]
            elif source == 'content':
                source_text = data[data['news_id']==file_path]['content'].values[0]
            elif source == 'title-content' :
                source_text = data[data['news_id']==file_path]['original_title'].values[0] + " " + data[data['news_id']==file_path]['content'].values[0]
            
            source_list.append(source_text)

    return target_list if source is None else (target_list, source_list)

if __name__ == '__main__':
    df = pd.read_csv('/workspace/code/bait_news_gen/data/Real/val_news.csv')
    target_list, source_list = extract_text(data=df, file_list=df['news_id'].tolist(), target='content', source='content')
    print(target_list[0])