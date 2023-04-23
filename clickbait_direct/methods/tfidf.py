import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer



# ========================
# select
# ========================

def tfidf_category_select(sim_filepath: str) -> str:
    """
    select news title among file list using tfidf similarity
    """
    # target file
    target_file = json.load(open(sim_filepath, 'r'))
    fake_title = target_file['sourceDataInfo']['newsTitle'] #fake title 반환

    return fake_title

def tfidf_title_category_select(sim_filepath: str) -> str:
    return tfidf_category_select(sim_filepath=sim_filepath)


def tfidf_content_category_select(sim_filepath: str) -> str:
    return tfidf_category_select(sim_filepath=sim_filepath)


def tfidf_avg_category_select(sim_filepath: str) -> str:
    return tfidf_category_select(sim_filepath=sim_filepath)



# ========================
# similarity
# ========================


def tfidf_sim_matrix(text_fit: list, query_text_list: list, key_text_list: list, **kwargs) -> np.ndarray:
    """
    make similarity matrix using tfidf similarity
    """
    print(">>> start making tfidf_sim_matrix")
    tf_idf_model = TfidfVectorizer().fit(text_fit) #train 데이터만 사용해서 fit
    query_tf_idf = tf_idf_model.transform(query_text_list).toarray() #train, validation 모두 transform
    key_tf_idf = tf_idf_model.transform(key_text_list).toarray() #train, validation 모두 transform
    cos_sim = cosine_similarity(query_tf_idf, key_tf_idf) # query by key similarity matrix

    return cos_sim


def tfidf_avg_sim_matrix(text_fit: list, text_a: list, text_b:list, **kwargs) -> np.ndarray:
    print(">>> start making tfidf_avg_sim_matrix")
    
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

    tok_fit = [tokenizer.tokenize(t)[:1000] for t in text_fit]
    tok_a = [tokenizer.tokenize(t)[:1000] for t in text_a]
    tok_b = [tokenizer.tokenize(t)[:1000] for t in text_b]

    list_fit = [' '.join(s for s in temp) for temp in tok_fit]
    list_a = [' '.join(s for s in temp) for temp in tok_a]
    list_b = [' '.join(s for s in temp) for temp in tok_b]

    tf_idf_model = TfidfVectorizer().fit(list_fit) #train 데이터만 사용해서 fit

    tf_idf_df_a = tf_idf_model.transform(list_a).toarray()
    tf_idf_df_b = tf_idf_model.transform(list_b).toarray()

    cos_sim = cosine_similarity(tf_idf_df_a, tf_idf_df_b)
    
    return cos_sim
