from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random
import time

from tqdm.auto import tqdm
from methods import get_similar_filepath_dict, extract_nouns, extract_text, tfidf_category_select
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def tfidf_sim_matrix_noun(target_text_list: list, source_text_list : list, **kwargs) -> np.ndarray:
    """
    make similarity matrix using tfidf similarity
    """
    tf_idf_model = TfidfVectorizer().fit(source_text_list)
    source_tfidf = tf_idf_model.transform(source_text_list).toarray()
    target_tfidf = tf_idf_model.transform(target_text_list).toarray()
    start = time.time()
    cos_sim = cosine_similarity(target_tfidf, source_tfidf)    
    end = time.time() - start
    print(f">>> TFIDF scoring takes {end}s.")
    return cos_sim

def tfidf_sim_matrix(target_text_list: list, source_text_list : list, **kwargs) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

    tok_fit = [tokenizer.tokenize(t)[:1000] for t in source_text_list]
    tok_a = [tokenizer.tokenize(t)[:1000] for t in source_text_list]
    tok_b = [tokenizer.tokenize(t)[:1000] for t in target_text_list]

    list_fit = [' '.join(s for s in temp) for temp in tok_fit]
    list_a = [' '.join(s for s in temp) for temp in tok_a]
    list_b = [' '.join(s for s in temp) for temp in tok_b]

    tf_idf_model = TfidfVectorizer().fit(list_fit) #train 데이터만 사용해서 fit

    tf_idf_df_a = tf_idf_model.transform(list_a).toarray()
    tf_idf_df_b = tf_idf_model.transform(list_b).toarray()

    cos_sim = cosine_similarity(tf_idf_df_a, tf_idf_df_b)
    
    return cos_sim

def score_tfidf_acc(sim_matrix, category, method_name):
    target_ids = list(range(len(sim_matrix)))
    top_1 = sum(np.argmax(sim_matrix, axis=1) == target_ids)
    top_1_accuracy = round(top_1/len(sim_matrix), 5)
    print(f"{method_name}_{category} TF-IDF ACC : {top_1_accuracy}")
    return top_1_accuracy


def main(method_name : str, 
         extract_text_func, 
         file_list: list, 
         category_list: list, 
         target: str, 
         savedir: str, 
         source : str = None,
         top_k : int = 3
         ) -> None:

    # define save path
    savepath = os.path.join(savedir, f'sim_index_{target}_123.json')

    # load sim_filepath_dict
    if not os.path.isfile(savepath):
        sim_filepath_dict = dict()
    else:
        sim_filepath_dict = json.load(open(savepath, 'r'))

    # define progress bar
    pbar = tqdm(category_list, total=len(category_list))

    tfidf_acc_list = []
    for category in pbar:
        pbar.set_description(f'Category: {category}')

        # run if category not in sim_filepath_dict
        if not category in sim_filepath_dict.keys():
            # set default
            sim_filepath_dict.setdefault(category, {})

            # extract file path in category
            print(">>> Extract tokenized text from file list.")
            file_list_cat = [f for f in file_list if category in f]
            target_text_list, source_text_list = extract_text_func(file_list = file_list_cat, target = target, source = source)
            
            print(">>> Calculate TF-IDF similarity matrix.")
            sim_matrix = tfidf_sim_matrix(
                target_text_list = target_text_list,
                source_text_list = source_text_list
            )
            
            # get tfidf acc
            tfidf_acc = score_tfidf_acc(sim_matrix, category, method_name)
            tfidf_acc_list.append(tfidf_acc)

            # mask diagonal
            sim_matrix[np.arange(sim_matrix.shape[0]), np.arange(sim_matrix.shape[0])] = -1

            # update similarity matrix
            # define save path
            savepath = os.path.join(savedir, f'sim_index_{target}.json')

            # load sim_filepath_dict
            if not os.path.isfile(savepath):
                sim_filepath_dict = dict()
            else:
                sim_filepath_dict = json.load(open(savepath, 'r'))
            
            sim_filepath_dict.setdefault(category, dict())
            
            # find i-th index
            sim_index = np.argsort(sim_matrix, axis=1)[:,top_k]
            
            # update sim_filepath_dict
            for file_path, idx in zip(file_list_cat, sim_index):
                sim_filepath_dict[category][file_path] = file_list_cat[idx]

            # save sim_filepath_dict
            json.dump(sim_filepath_dict, open(savepath, 'w'), indent=4)
    print(f"TFIDF top-1 acc mean: {np.mean(tfidf_acc_list)}")
    return sim_filepath_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename', default='configs/tfidf/title_content.yaml')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg['SEED'])

    # update save directory
    cfg['savedir'] = os.path.join(cfg['savedir'], cfg['METHOD']['select_name'])
    # load file list
    file_list = glob(os.path.join(cfg['datadir'], '[!sample]*/Auto/*/*'))
    save_list = [p.replace(cfg['datadir'], cfg['savedir']) for p in file_list]
    print(len(file_list), len(save_list))

    # make directory to save files
    partition_path = glob(os.path.join(cfg['datadir'], '[!sample]*/Auto/*'))
    partition_path = [p.replace(cfg['datadir'], cfg['savedir']) for p in partition_path]

    ## train, validation, test 폴더 생성
    for path in partition_path:
        os.makedirs(path, exist_ok=True)    

    sim_filepath_dict = main(
        method_name          = cfg['METHOD']['name'],
        extract_text_func    = extract_text if ('dense' in cfg['METHOD']['name']) or (cfg['METHOD']['extract'] == 'all') else extract_nouns,
        file_list            = file_list,
        category_list        = os.listdir(os.path.join(cfg['savedir'],'train/Auto')),
        target               = cfg['METHOD']['target'],
        source               = cfg['METHOD']['source'],
        savedir              = cfg['savedir'],
        top_k                = cfg['METHOD']['topk'],
    )

