model_sort_list="News_Base"
bait_sort_list="Fake/auto/type"
saved_model_path='None'

for model_sort in $model_sort_list
do
    for bait_sort in $bait_sort_list
    do
        echo $sort
        python3 main.py --model_sort $model_sort --bait_sort $bait_sort --saved_model_path $saved_model_path
    done
done