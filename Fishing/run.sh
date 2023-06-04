fake_path_list="Fake/tfidf/filtered Fake/content_chunking_backward/filtered Fake/content_chunking_forward/filtered Fake/content_rotation_backward/filtered" 
saved_model_path='None'

for fake_path in $fake_path_list
do
    echo $sort
    python3 main.py --fake_path $fake_path --saved_model_path $saved_model_path
done
