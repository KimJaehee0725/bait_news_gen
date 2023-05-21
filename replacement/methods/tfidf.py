import numpy as np
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer


# ========================
# similarity
# ========================

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

def overlap_token(cos_sim, target_text_list: list, source_text_list : list, method_name=None, top_k = None):
    # 1. 유사도 점수 기반 topk개 뽑기
    topkindex = np.argpartition(cos_sim,-top_k,axis=1)[:,-top_k:] #not sorted

    token_overlapped = dict() #q_id : {d_id : # of overlapped tokens}

    # 2. overlap token (intersection : 중복 없이 겹치는 토큰 개수, count : 중복 포함 겹치는 토큰 개수)
    if 'intersection' in method_name :
        for t_idx, target_text in enumerate(target_text_list):        
            target_token_list = set(target_text.split())
            token_overlapped[t_idx] = {}
            for s_idx in topkindex[t_idx]:
                source_text = source_text_list[s_idx]
                source_token_list = set(source_text.split()) 
                count = len(target_token_list & source_token_list)
                token_overlapped[t_idx][s_idx] = count

    if 'count' in method_name:
        # 2. 각 target마다 topk개의 source 마다 겹치는 토큰 개수 세기
        for t_idx, target_text in enumerate(target_text_list):
            # 2-1. 'token1 token2 ...' -> ['token1', 'token2',..] -> 중복 제거
            target_token_list = list(set(target_text.split())) 
            token_overlapped[t_idx] = {}
            for s_idx in topkindex[t_idx]:
                source_text = source_text_list[s_idx]
                cnt_list = [source_text.count(token) for token in target_token_list]
                count = sum(cnt_list)
                token_overlapped[t_idx][s_idx] = count  
    # 3. sorting
    token_overlapped_sorted = dict()
    for t_idx, cnt_overlapped_dict in token_overlapped.items():
        token_overlapped_sorted[t_idx] = dict(sorted(cnt_overlapped_dict.items(), key=lambda item: item[1], reverse=True))

    # 4. return format에 맞추기 : 정렬한 순으로 source index로 구성된 리스트
    sorted_idx = [list(source_cnt_dict.keys()) for _, source_cnt_dict in token_overlapped_sorted.items()] 
    sorted_idx = np.array(sorted_idx)
    return sorted_idx
