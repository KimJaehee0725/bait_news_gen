bait_path='../data/generated/tfidf_avg_category_select'
sort_list="News_Auto News_Direct"

for sort in $sort_list
do
    echo $sort
    python3 main.py --bait_path $bait_path --sort $sort
done