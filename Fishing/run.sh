fake_path="Fake/tfidf/filtered"
fake_name='fake_top1_90_99.csv'
saved_model_path='None'


echo $fake_path $fake_name
echo $saved_model_path
python3 main.py --fake_path $fake_path --fake_name $fake_name --saved_model_path $saved_model_path



