# bait_news_gen

---------------

## data

1. train
- News_Direct
- News_Auto
- News_Direct_Auto

2. test
- News
- Direct
- Auto

## py파일
- main.py
- dataset.py
- model.py
- train.py
- log.py

---------------

## 실행

#### < 실행 전 확인 사항 >
1.  데이터 폴더명 변경

    : NonClickbait_Auto → News / Clickbait_Direct → Direct / Clickbait_Auto → Auto

    - 데이터 경로에 Direct, Auto 포함되는지 여부 체크 필요
    - 라벨링 부분에서 Direct, Auto 포함 여부로 판단하므로, News 데이터의 경로엔 해당 단어 없어야 함.

2. config 파일 내 data_path, bait_path 수정
    
    : data_path → News, Direct 경로 / bait_path → Auto 경로로 수정





```
python main.py --yaml_config ./configs/{데이터명}.yaml
```

- train 시킨 모델로 test만 하고 싶을 때

```
python forTest.py --yaml_config ./configs/{데이터명}.yaml
```
 → 사용시, 원하는 학습 모델 경로로 forTest.py내 checkpoint 수정 필요

---------------

## 동작

1. python main.py --yaml_config ./configs/{데이터명}.yaml 실행
2. config 파일 내 data_path와 bait_path에서 데이터 로드
3. config 파일 내 sort에 지정된 데이터 종류로 모델 학습
4. checkpoint 저장
5. 테스트용 데이터 로드 : News, Direct, Auto
6. 학습 모델 테스트
7. 테스트 결과 저장


- train 시킨 모델로 test만 하고 싶을 때
1. forTest.py 내 ckeckpoint 경로 수정
2. python forTest.py --yaml_config ./configs/{데이터명}.yaml 실행
3. checkpoint 모델 로드
4. 테스트용 데이터 로드 : News, Direct, Auto 
5. 학습 모델 테스트
6. 테스트 결과 저장

