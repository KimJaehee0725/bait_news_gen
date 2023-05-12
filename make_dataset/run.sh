cfg_list=("sim_index_content_content.json")
save_list=("../data/bait/tfidf_content_content")

for ((i=0; i<${#cfg_list[@]}; i++)); do
    cfg=${cfg_list[$i]}
    save=${save_list[$i]}
    python3 make_dataset.py —index_dir $cfg —save_dir $save
done

