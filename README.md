# bait_news_gen

## 🍀 data directory 구성
```bash
├── data : 
│   ├── T5 : train_t5.csv
│   ├── Real :
│   │   ├── train_news.csv
│   │   ├── val_news.csv
│   │   └── test_news.csv
│   └── Fake :
│       ├── fake_original.csv 
│       ├── content_chunking_forward
│       │   ├── generated
│       │   │   ├── fake_top1.csv
│       │   │   └── fake_top2.csv
│       │   └── filtered : fake_topk_threshold.csv
│       │   │   ├── fake_top1_95.csv
│       │   │   └── fake_top2_95.csv
│       ├── rotation_chunking_backward
│       │   ├── generated
│       │   │   ├── fake_top1.csv
│       │   │   └── fake_top2.csv
│       │   └── filtered : fake_topk_threshold.csv
│       │   │   ├── fake_top1_95.csv
│       │   │   └── fake_top2_95.csv
│       └── tfidf
│           ├── generated
│           │   ├── fake_top1.csv
│           │   └── fake_top2.csv
│           └── filtered : fake_topk_threshold.csv
│               ├── fake_top1_95.csv
│               └── fake_top2_95.csv
│
├── replacement : 교체 방법론 (tfidf, random, ...)
│   ├── 
│   └──
│ 
├── T5 :
│   ├── finetuning : T5 fine-tuning
│   │   ├── hf_dataset  
│   │   ├── Models  
│   │   └── train.py
│   └── generation : 가짜 뉴스 생성
│       ├── generate.py  
│       ├── methods.py  
│       └── generate.sh
│
├── filtering : false negative 필터링
│   ├── 
│   └──
│
├── detection : 탐지 모델 학습
│   ├── 
│   └── 
│
│
├── results : 학습된 모델, 학습 결과, 테스트 결과 저장
│   └── tfidf
│       ├── best_model.pt
│       ├── best_score.json : train 결과
│       ├── latest_model.pt
│       └── content_chunking_forward
│           ├── exp_metrics.json : test f1, acc, loss
│           ├── exp_results.csv : test 결과
│           └── main_results.csv : paper table 용 - B>C / D / false_negative
│
│  
│
└──    
```

## 💚 py파일
- main.py
- dataset.py
- model.py
- train.py
- log.py


## 🔫 실행

#### < 실행 전 확인 사항 >
1.  본인의 데이터 폴더 형태 확인

    : 위 directory 구성 참고
    - bait_sort 지정 시, ex) Fake/content_chunking_forward 와 같이 넣기 위해

2. config 파일 내 data_path 수정
    
    : data_path → 모든 데이터가 포함되어 있는 폴더 경로, 끝 부분에 '/' 붙여야 함




#### 기본 실행
- run.sh에서 fake_path와 saved_model_path 수정해서 사용
    - fake_path : 학습 시 사용할 fake 데이터 폴더 경로 - Fake/{bait 종류}
    - saved_model_path
        - train부터 test까지 하는 경우 - 'None'
        - 이미 저장된 모델로 test만 하는 경우 - saved_model/{방법론}/best_model.pt
```
bash run.sh
```

- 실행 방법

    - train~test
    1. [run.sh](http://run.sh/) 파일 내의 fake_path에 학습하고자 하는 fake 데이터 지정 (ex. Fake/tfidf)
    2. saved_model_path에 “None”
    -> train data: news+A / test data: news+A

    - only test
    1. [run.sh](http://run.sh/) 파일 내의  fake_path에 테스트하고자 하는 fake 데이터 지정 (ex. Fake/tfidf)
    2. saved_model_path에 사용할 모델 지정 (ex. saved_model/A/best_model.pt)
    -> model: A데이터로 학습된 모델 / test data: news+B





