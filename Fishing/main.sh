<<<<<<< HEAD
bait_path='../data-auto/false-negative'
sort_list="News_Direct"
=======
bait_path='../../data/Bait/original'
sort_list="News_Base"
>>>>>>> 971733263ead4f26987c0dadac08104b343257bf

for sort in $sort_list
do
    echo $sort
    python3 main.py --bait_path $bait_path --sort $sort
done