import numpy as np
import os
import json

from tqdm.auto import tqdm
from konlpy.tag import Mecab
from typing import List
from .tfidf import overlap_token 

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from utils import score_tfidf_acc,score_overlap_acc
#TODO : 이거 어떻게 넣는 게 좋을까?




def get_similar_filepath_dict(
    method_name : str, make_sim_matrix_func, extract_text_func, 
    file_list: list, category_list: list, target: str, savedir: str, source : str = None, top_k : int = None, test = False) -> None:

    # define save path
    savepath = os.path.join(savedir, f'sim_index_{target}.json')

    # load sim_filepath_dict
    if not os.path.isfile(savepath):
        sim_filepath_dict = dict()
    else:
        sim_filepath_dict = json.load(open(savepath, 'r'))

    # define progress bar
    pbar = tqdm(category_list, total=len(category_list))

    for category in pbar:
        pbar.set_description(f'Category: {category}')

        # run if category not in sim_filepath_dict
        if not category in sim_filepath_dict.keys():
            # set default
            sim_filepath_dict.setdefault(category, {})

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
            if test :
                score_tfidf_acc(sim_matrix, category)

            if 'dense' not in method_name: 
                sim_matrix[np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[0])] = -1
                tfidf_sim_matrix = sim_matrix

            # TODO : add <overlap> func
            if 'overlap' in method_name:
                sim_matrix = overlap_token(method_name, sim_matrix, target_text_list, source_text_list, top_k = top_k)

            if test : #TFIDF top1과 TFIDF-overlap top1이 얼마나 다른가
                check_overlap(tfidf, target_text_list, source_text_list)

            # update similarity matrix
            sim_filepath_dict = extract_sim_filepath(
                method_name       = method_name,
                sim_matrix        = sim_matrix,
                file_list         = file_list_cat,
                category          = category,
                sim_filepath_dict = sim_filepath_dict
            )

            # save sim_filepath_dict
            json.dump(sim_filepath_dict, open(savepath, 'w'), indent=4)

    return sim_filepath_dict


def extract_sim_filepath(
    method_name : str, sim_matrix: np.ndarray, file_list: list, category: str, sim_filepath_dict: dict) -> None:
    """
    extract filepath most similar to filepath1 using ngram similarity
    """

    # find argmax
    if ['dense', 'overlap'] in method_name :
        sim_index = list(sim_matrix[:, 0])
    else :
        sim_index = sim_matrix.argmax(axis=1)

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


    for idx in tqdm(range(len(target_list)), desc=f'Extract Morphs({target})', total=len(file_list), leave=False):
        if join:
            if source != None:
                source_n_list.append(' '.join(mecab.nouns(source_list[idx])))
            target_n_list.append(' '.join(mecab.nouns(target_list[idx])))
        else:
            if source != None:
                source_n_list.append(mecab.nouns(source_list[idx]))
            target_n_list.append(mecab.nouns(target_list[idx]))

    return target_n_list if source is None else (target_n_list, source_n_list) #['token1 token2 ... tokenN', ...]


def extract_text(file_list: list, target: str, source : str = None) -> List[list]:
    """
    extract target text
    """

    # define list
    target_list = []
    if source is not None :
        source_list = []

    for file_path in tqdm(file_list, desc=f'Extract Morphs({target})', total=len(file_list), leave=False):
        # load source file
        source_file = json.load(open(file_path, "r"))

        if target == 'title':
            target_text = source_file['sourceDataInfo']['newsTitle']
        elif target == 'content':
            target_text = source_file['sourceDataInfo']['newsContent']
        elif target == 'title-content' :
            target_text = source_file['sourceDataInfo']['newsTitle'] + " " + source_file['sourceDataInfo']['newsContent']
        
        target_list.append(target_text)
        
        if source is not None :
            if source == 'title':
                source_text = source_file['sourceDataInfo']['newsTitle']
            elif source == 'content':
                source_text = source_file['sourceDataInfo']['newsContent']
            elif source == 'title-content' :
                source_text = source_file['sourceDataInfo']['newsTitle'] + " " + source_file['sourceDataInfo']['newsContent']
            
            source_list.append(source_text)

    return target_list if source is None else (target_list, source_list)