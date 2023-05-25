import pandas as pd
import numpy as np
import json



def make_results(fake_name):

    # B
    with open(f'../results/tfidf/{fake_name}/exp_metrics.json', 'r') as f:
        B_data = json.load(f)
    B_score = B_data['acc']

    # C
    with open(f'../results/{fake_name}/tfidf/exp_metrics.json', 'r') as f:
        C_data = json.load(f)
    C_score = C_data['acc']

    # D
    with open(f'../results/{fake_name}/{fake_name}/exp_metrics.json', 'r') as f:
        D_data = json.load(f)
    D_score = D_data['acc']

    result = pd.DataFrame({
        'C/B' : [C_score / B_score],
        'D' : [D_score],
        'false negative': [0]
    })

    result.to_csv(f'../results/{fake_name}/main_results.csv', index=False)



fake_name = 'content_chunking_forward' #! 필요한 fake 종류로 변경
make_results(fake_name)


