bait_path='../../main_generation/data-Auto/tfidf_avg_category_select'
sort="News_Direct"
saved_model_path='../saved_model/News_Direct/best_model.pt'

python main.py --bait_path $bait_path --sort $sort --saved_model_path $saved_model_path