bait_path='../data-auto/false-negative'
sort_list="News_Direct"

for sort in $sort_list
do
    echo $sort
    python3 main.py --bait_path $bait_path --sort $sort
done