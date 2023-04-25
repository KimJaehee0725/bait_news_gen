bait_path= '../../Fake-News-Detection-Dataset/data/Part1' 
sort="News_Auto"
saved_model_path='../saved_model/News_Direct/best_model.pt'

python3 test.py --bait_path ../../Fake-News-Detection-Dataset/data/Part1 --sort $sort --saved_model_path $saved_model_path