# bait_news_gen

## 🍀 data directory 구성
- data 
    - train
        - News
        - Direct
        - Auto
    - validation
        - train과 구성 동일
    - test
        - train과 구성 동일
- data-auto
    - method 이름
        - train
            - Auto
        - validation
            - Auto
        - test
            - Auto

## 💚 py파일
- main.py
- dataset.py
- model.py
- train.py
- test.py
- log.py


## 🔫 실행

#### < 실행 전 확인 사항 >
1.  데이터 폴더명 변경

    : NonClickbait_Auto → News / Clickbait_Direct → Direct / Clickbait_Auto → Auto

    - 데이터 경로에 Direct, Auto 포함되는지 여부 체크 필요
    - 라벨링 부분에서 Direct, Auto 포함 여부로 판단하므로, News 데이터의 경로엔 해당 단어 없어야 함.

2. config 파일 내 data_path, bait_path 수정
    
    : data_path → News, Direct 경로 / bait_path → Auto 경로로 수정




#### 기본 실행
- main.sh에서 bait_path와 sort_list수정해서 사용하면 됩니다.
    - bait_path : auto로 만들어진 가짜뉴스 데이터 폴더
    - sort_list : News_Auto(auto로 BERT 학습), News_Direct(direct로 BERT 학습)
```
bash main.sh
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
