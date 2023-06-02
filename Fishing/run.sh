fake_path="Fake/content_chunking_forward/filtered"
fake_name='fake_top1_90_99.csv'
saved_model_path='/root/code/bait_news_gen/results/tfidf/fake_top1_90_99/best_model.pt'


echo $fake_path $fake_name
echo $saved_model_path
python3 main.py --fake_path $fake_path --fake_name $fake_name --saved_model_path $saved_model_path



