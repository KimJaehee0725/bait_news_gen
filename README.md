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
└── detection : 탐지 모델 학습
    ├── 
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
1.  데이터 폴더 형태 확인

    : 위 directory 구성 참고
    - bait_sort 지정 시, ex) Fake/content_chunking_forward 와 같이 넣기 위해

2. config 파일 내 data_path 수정
    
    : data_path → 모든 데이터가 포함되어 있는 폴더 경로, 끝 부분에 '/' 붙여야 함




#### 기본 실행
- run.sh에서 model_sort와 bait_sort수정해서 사용하면 됩니다.
    - model_sort : 학습 시 사용할 데이터 유형 - News_Base / News_Auto 
    - bait_sort : 학습 시 사용할 bait 데이터 폴더 경로 - Fake/{bait 종류}
```
bash run.sh
```

```
python main.py --base_config ./configs/{데이터명}.yaml
```

- train 시킨 모델로 test만 하고 싶을 때

```
python test.py --base_config ./configs/{데이터명}.yaml
```
 → 사용시, 원하는 학습 모델 경로로 test.py내 checkpoint 수정 필요

---------------

## 🍈 동작
### train + test
1. python main.py --base_config ./configs/{데이터명}.yaml 실행
2. config 파일 내 data_path와 bait_path에서 데이터 로드
3. config 파일 내 sort에 지정된 데이터 종류로 모델 학습
4. checkpoint 저장
5. 테스트용 데이터 로드 : News, Direct, Auto
6. 학습 모델 테스트
7. 테스트 결과 저장


### train없이, only test
1. test.py 내 ckeckpoint 경로 수정
2. python test.py --base_config ./configs/{데이터명}.yaml 실행
3. checkpoint 모델 로드
4. 테스트용 데이터 로드 : News, Direct, Auto 
5. 학습 모델 테스트
6. 테스트 결과 저장


---------------
## 🍏 예시

#### News_Direct 데이터로 학습 시키고 싶다면?
```
python main.py --base_config ./configs/News_Direct.yaml
```
(config 내 sort는 News_Direct로 되어 있어야 함)


#### News_Direct 데이터로 학습 시킨 모델에 새로운 Auto 데이터를 테스트 하고 싶다면?
→ 새로운 config 파일 생성 필요

bait_path : 새로운 Auto 경로로 수정 (나머지 항목 수정 필요x)
```
python test.py --base_config ./configs/{새로운 config}.yaml
```
