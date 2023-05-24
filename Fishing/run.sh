fake_path_list="Fake/auto/type"
saved_model_path='None'

for fake_path in $fake_path_list
do
    echo $sort
    python3 main.py --fake_path $fake_path --saved_model_path $saved_model_path
done