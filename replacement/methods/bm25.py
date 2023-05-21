import numpy as np
import json

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
import time
import json 
from tqdm import tqdm



# ========================
# select
# ========================

def bm25_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using tfidf similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle']

    return fake_title

def bm25_title_title_category_select(sim_filepath: str) -> str:
    return bm25_category_select(sim_filepath=sim_filepath)

def bm25_title_content_category_select(sim_filepath: str) -> str:
    return bm25_category_select(sim_filepath=sim_filepath)

def bm25_title_all_category_select(sim_filepath: str) -> str:
    return bm25_category_select(sim_filepath=sim_filepath)

def bm25_content_content_category_select(sim_filepath: str) -> str:
    return bm25_category_select(sim_filepath=sim_filepath)

def bm25_content_all_category_select(sim_filepath: str) -> str:
    return bm25_category_select(sim_filepath=sim_filepath)

# ========================
# similarity
# ========================


def bm25_sim_matrix(target_text_list: list, source_text_list : list, **kwargs) -> np.ndarray:
    """
    make similarity matrix using bm25 similarity
    """
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    tok_target = [tokenizer.tokenize(t)[:1000] for t in target_text_list]
    tok_source = [tokenizer.tokenize(t)[:1000] for t in source_text_list]
    print(">>> BM25 Indexing Start")
    start = time.time()
    bm25 = BM25Okapi(tok_source)
    end = time.time() - start
    print(f">>> BM25 Indexing takes {end}s.")
    
    bm25_sim = np.zeros((len(target_text_list), len(target_text_list)), dtype=np.float16)
    for row, tokenized_query in enumerate(tqdm(tok_target, desc='>>> BM25 Ranking', dynamic_ncols=True)):
        doc_scores = bm25.get_scores(tokenized_query)
        bm25_sim[row] = doc_scores
    return bm25_sim

