# bait_news_gen

---------------

<data>
1. train
- News_Direct
- News_Auto
- News_Direct_Auto

2. test
- News
- Direct
- Auto

<py파일>
- main.py
- dataset.py
- model.py
- train.py
- log.py

---------------

<실행>
```
python main.py --yaml_config ./configs/{데이터명}.yaml
```
---------------
- train만 되고 test가 안된 경우
1.  main.py 에서 line 86 ~ 124까지 주석처리
2. line 81에 *model.load_state_dict(torch.load('../saved_model/News_Direct/best_model.pt'))* 추가 → 학습시킨 데이터로 중간 폴더명 변경
3. line 129에 *criterion = torch.nn.CrossEntropyLoss()* 추가
4. line 151에 train_model을 model로 변경
5. python main.py --yaml_config ./configs/News_Direct.yaml 실행 → test부터 시작됨

--->> forTest.py 파일 실행으로 해결. 사용시, 원하는 학습 모델 경로로 수정 필요**

