import numpy as np
import os
import json

from tqdm.auto import tqdm
from konlpy.tag import Mecab
from typing import List




def get_similar_filepath_dict(
    select_name: str,
    make_sim_matrix_func, # 000_sim_matrix
    extract_text_func, # extract_nouns / extract_text
    file_list: list, 
    category_list: list, 
    query : str,
    key : str,
    fit_data : str,
    savedir: str) -> None:

    # define save path
    savepath = os.path.join(savedir, f'sim_index_{query}.json')

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

            file_list_train = [x for x in file_list_cat if 'train' in x] # train 데이터
            file_list_val = [x for x in file_list_cat if 'validation' in x] # validation 데이터
            file_list_test = [x for x in file_list_cat if 'test' in x] # test 데이터

            fit_data_text = extract_text_func(file_list = file_list_train, extract_type = fit_data) #fit_data는 train 데이터만 사용
            query_train_text = extract_text_func(file_list = file_list_train, extract_type = query) #train 데이터
            query_val_text = extract_text_func(file_list = file_list_val, extract_type = query) #validation 데이터
            query_test_text = extract_text_func(file_list = file_list_test, extract_type = query) #test 데이터

            if "-" in key : # key로 복수의 값을 사용할 경우 sim matrix 평균 이용
                key_list = key.split("-")
                sim_matrix_list = []
                sim_matrix_for_val_list = []
                sim_matrix_for_test_list = []

                for _key in key_list: #["title", "content"]

                    key_train_text = extract_text_func(file_list = file_list_train, extract_type = _key) #train 데이터
                    key_val_text = extract_text_func(file_list = file_list_val, extract_type = _key) #validation 데이터
                    key_test_text = extract_text_func(file_list = file_list_test, extract_type = _key) #test 데이터
                    
                    sim_matrix = make_sim_matrix_func(
                        text_fit        = fit_data_text,
                        query_text_list = query_train_text + query_val_text + query_test_text, 
                        key_text_list   = key_train_text + key_val_text + key_test_text
                    )

                    sim_matrix_for_val = make_sim_matrix_func(
                        text_fit        = fit_data_text,
                        query_text_list = query_val_text,
                        key_text_list   = key_val_text
                    )

                    sim_matrix_for_test = make_sim_matrix_func(
                        text_fit        = fit_data_text,
                        query_text_list = query_test_text,
                        key_text_list   = key_test_text
                    )

                    sim_matrix_list.append(sim_matrix)
                    sim_matrix_for_val_list.append(sim_matrix_for_val)
                    sim_matrix_for_test_list.append(sim_matrix_for_test)

                sim_matrix = np.mean(sim_matrix_list, axis = 0)
                sim_matrix_for_val = np.mean(sim_matrix_for_val_list, axis = 0)
                sim_matrix_for_test = np.mean(sim_matrix_for_test_list, axis = 0)
                
            else : 
                key_train_text = extract_text_func(file_list = file_list_train, extract_type = key) #train 데이터
                key_val_text = extract_text_func(file_list = file_list_val, extract_type = key) #validation 데이터
                key_test_text = extract_text_func(file_list = file_list_test, extract_type = key) #test 데이터

                sim_matrix = make_sim_matrix_func(
                    text_fit        = fit_data_text,
                    query_text_list = query_train_text + query_val_text + query_test_text, 
                    key_text_list   = key_train_text + key_val_text + key_test_text
                )

                sim_matrix_for_val = make_sim_matrix_func(
                        text_fit        = fit_data_text,
                        query_text_list = query_val_text,
                        key_text_list   = key_val_text
                    )                

                sim_matrix_for_test = make_sim_matrix_func(
                    text_fit        = fit_data_text,
                    query_text_list = query_test_text,
                    key_text_list   = key_test_text
                )

            if 'dense' not in select_name :
                ## get top-1 and top-10 accuracy
                # get valid data num
                valid_num = sim_matrix_for_val.shape[0]
                # get top-1 accuracy
                top1_acc = np.sum(np.argmax(sim_matrix_for_val, axis=1) == np.arange(valid_num)) / valid_num
                # get top-10 accuracy
                top10_acc = np.sum(np.argsort(sim_matrix_for_val, axis=1)[:, -10:] == np.arange(valid_num)[:, None]) / valid_num
                
                # check if savedir exists
                if os.path.exists(os.path.join(savedir, 'top1_top10_acc.txt')):
                    mode = 'a'
                else:
                    mode = 'w'
                with open(os.path.join(savedir, 'top1_top10_acc.txt'), mode) as f:
                    f.write(f'{category} {top1_acc} {top10_acc} \n')

                sim_matrix[np.arange(sim_matrix.shape[1]-1), np.arange(sim_matrix.shape[1]-1)] = -1

            # make sim_filepath_dict
            sim_filepath_dict = extract_sim_filepath(
                select_name       = select_name,
                sim_matrix        = sim_matrix,
                file_list         = file_list_cat,
                category          = category,
                sim_filepath_dict = sim_filepath_dict
            )

            # save sim_filepath_dict
            json.dump(sim_filepath_dict, open(savepath, 'w'), indent=4)

    return sim_filepath_dict

def extract_sim_filepath(
    select_name: str, sim_matrix: np.ndarray, file_list: list, category: str, sim_filepath_dict: dict) -> None:
    """
    extract filepath most similar to filepath1 using ngram similarity
    """

    # find argmax
    if 'dense' in select_name :
        sim_index = list(sim_matrix[:, 0])
    else :
        sim_index = sim_matrix.argmax(axis=1)

    # update sim_filepath_dict
    for file_path, idx in zip(file_list, sim_index):
        sim_filepath_dict[category][file_path] = file_list[idx]
    
    return sim_filepath_dict


def extract_nouns(file_list: list, extract_type: str, join: bool = True) -> List[list]:
    """
    extract nouns from query text
    """
    # extract morphs
    mecab = Mecab()

    # define list
    nouns_list = []

    for file_path in tqdm(file_list, desc=f'Extract Morphs from ({extract_type})', total=len(file_list), leave=False):
        # load source file
        source_file = json.load(open(file_path, "r"))

        if extract_type == 'title':
            text = source_file['sourceDataInfo']['newsTitle']
        elif extract_type == 'content':
            text = source_file['sourceDataInfo']['newsContent']
        elif extract_type == 'title+content' :
            text = source_file['sourceDataInfo']['newsTitle'] + source_file['sourceDataInfo']['newsContent']

        if join:
            nouns_list.append(' '.join(mecab.nouns(text)))
        else:
            nouns_list.append(mecab.nouns(text))

    return nouns_list


def extract_text(file_list: list, extract_type: str) -> List[list]:
    """
    extract query text
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    # define list
    text_list = []

    for file_path in tqdm(file_list, desc=f'Extract Whole Text from ({extract_type})', total=len(file_list), leave=False):
        # load source file
        source_file = json.load(open(file_path, "r"))


        if extract_type == 'title':
            _text = source_file['sourceDataInfo']['newsTitle']
        elif extract_type == 'content':
            _text = source_file['sourceDataInfo']['newsContent']
        elif extract_type == 'title+content' :
            _text = source_file['sourceDataInfo']['newsTitle'] + " " + source_file['sourceDataInfo']['newsContent']
        
        tokenized_text = tokenizer.tokenize(_text)[:1000]
        _text = " ".join(tokenized_text)
        text_list.append(_text)

    return text_list