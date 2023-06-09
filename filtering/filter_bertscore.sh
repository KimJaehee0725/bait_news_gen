# python filter_bertscore.py \
# --data_path ../data/Fake/content_chunking_backward/generated/fake_top3.csv \
# --save_path ../data/Fake/content_chunking_backward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

# python filter_bertscore.py \
# --data_path ../data/Fake/content_chunking_forward/generated/fake_top3.csv \
# --save_path ../data/Fake/content_chunking_forward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

# python filter_bertscore.py \
# --data_path ../data/Fake/content_rotation_backward/generated/fake_top3.csv \
# --save_path ../data/Fake/content_rotation_backward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

# python filter_bertscore.py \
# --data_path ../data/Fake/content_chunking_backward/generated/fake_top1.csv \
# --save_path ../data/Fake/content_chunking_backward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

# python filter_bertscore.py \
# --data_path ../data/Fake/content_chunking_forward/generated/fake_top1.csv \
# --save_path ../data/Fake/content_chunking_forward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

# python filter_bertscore.py \
# --data_path ../data/Fake/content_rotation_backward/generated/fake_top1.csv \
# --save_path ../data/Fake/content_rotation_backward/filtered \
# --threshold_under 0.9 \
# --threshold_upper 0.99

python filter_bertscore.py \
--data_path ../data/Fake/content_rotation_forward/generated/fake_top1.csv \
--save_path ../data/Fake/content_rotation_forward/filtered \
--threshold_under 0.9 \
--threshold_upper 0.99

python filter_bertscore.py \
--data_path ../data/Fake/content_rotation_forward/generated/fake_top3.csv \
--save_path ../data/Fake/content_rotation_forward/filtered \
--threshold_under 0.9 \
--threshold_upper 0.99