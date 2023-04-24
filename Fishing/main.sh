bait_path="/workspace/code/Fake-News-Detection-Dataset/data-Auto/t5_base/tfidf_content/"
bait_type="content_chunking_forward content_rotation_forward content_summarization_forward"
# bait_type="content_chunking_forward"
# sort_list="News_Direct"
sort_list="News_Auto News_Direct_Auto"
# News_Direct

for type in ${bait_type}
do
    for sort in ${sort_list}
    do
        python main.py --bait_path ${bait_path}${type} --sort ${sort}
    done
done

