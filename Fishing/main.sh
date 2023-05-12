bait_path='../../data/Bait/original'
sort_list="News_Auto"

for sort in $sort_list
do
    echo $sort
    python3 main.py --bait_path $bait_path --sort $sort
done