bait_sort_list="Fake/auto/type"
saved_model_path='/root/code/bait_news_gen/saved_model_old/original/News_Auto/best_model.pt'

for bait_sort in $bait_sort_list
do
    echo $sort
    python3 main.py --bait_sort $bait_sort --saved_model_path $saved_model_path
done