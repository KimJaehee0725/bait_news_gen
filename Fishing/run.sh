bait_sort_list="Fake/auto/type"
saved_model_path='None'

for bait_sort in $bait_sort_list
do
    echo $sort
    python3 main.py --bait_sort $bait_sort --saved_model_path $saved_model_path
done