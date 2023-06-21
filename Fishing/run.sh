fake_path_list="Fake/content_chunking_forward/generated Fake/content_chunking_backward/generated Fake/content_rotation_forward/generated Fake/content_rotation_backward/generated Fake/content_summarization_forward/generated Fake/content_summarization_backward/generated"
fake_name_list='fake_top1.csv'

for fake_path in $fake_path_list
do
    for fake_name in $fake_name_list
    do
        python3 main.py --fake_path $fake_path --fake_name $fake_name
    done
done