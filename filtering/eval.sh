method_list="content_chunking_backward content_chunking_forward content_rotation_backward content_rotation_forward content_summarization_backward content_summarization_forward" 

for method in $method_list
do
    echo $method
    python3 eval_by_tokenoverlap.py --method $method
done
