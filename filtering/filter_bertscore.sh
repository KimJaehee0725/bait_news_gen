python filter_bertscore.py \
--data_path ../data/Fake/content_summarization_forward/generated/fake_top1.csv \
--save_path ../data/Fake/content_summarization_forward/filtered \

python filter_bertscore.py \
--data_path ../data/Fake/content_summarization_backward/generated/fake_top1.csv \
--save_path ../data/Fake/content_summarization_backward/filtered \


python filter_bertscore.py \
--data_path ../data/Fake/content_chunking_backward/generated/fake_top1.csv \
--save_path ../data/Fake/content_chunking_backward/filtered \


python filter_bertscore.py \
--data_path ../data/Fake/content_chunking_forward/generated/fake_top1.csv \
--save_path ../data/Fake/content_chunking_forward/filtered \


python filter_bertscore.py \
--data_path ../data/Fake/content_rotation_backward/generated/fake_top1.csv \
--save_path ../data/Fake/content_rotation_backward/filtered \

python filter_bertscore.py \
--data_path ../data/Fake/content_rotation_forward/generated/fake_top1.csv \
--save_path ../data/Fake/content_rotation_forward/filtered \

