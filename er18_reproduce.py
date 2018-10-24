"""
Script to reproduce results in
"Maass, W. & Shcherbatyi, I. Inductive Discovery By Machine Learning for
Identification of Structural Models, The 37th International Conference on
Conceptual Modeling (ER), 2018."
"""

import os
from allrelations.interface import extract_1_to_1

dataset_path = os.path.join('datasets', 'gender_discrimination', 'gend_disc.csv')
results_path = os.path.join('experimental_results', 'gender_discrimination')

for model in ['lasso', 'knn', 'gbrt', 'tree', 'ann']:
    for use_resp_data in [False, True]:
        extract_1_to_1(dataset_path, results_path, model, use_resp_data)