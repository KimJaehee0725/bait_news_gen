fake_path_list="Fake/tfidf_full_full_all/generated"
saved_model_path='None'

for fake_path in $fake_path_list
do
    echo $sort
    python3 main.py --fake_path $fake_path --saved_model_path $saved_model_path
done