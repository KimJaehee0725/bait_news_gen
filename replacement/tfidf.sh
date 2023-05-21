cfg_list="configs/tfidf/content_content.yaml"
for cfg in $cfg_list
do
    python3 get_sim_index_topk.py --yaml_config $cfg
done