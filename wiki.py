'''
parses and plots wiki dataset
@author: iaroslav
'''

from dataset_parser import read_dataset
import relations_backend as rbc
import fit_approximator as fx
import pickle
import os
import numpy as np

### SETTINGS ###

dataset_csv = "datasets/wiki.csv" # see wiki.csv to understand format used
dataset_extra_csv = None # extra data e.g. about persons age added for every relation pair, can be used in further work
results_folder = "results" # this folder should exist

### CODE STARTS HERE ###

dataset_name = os.path.basename(dataset_csv)[:-4] # remove the .csv at the end of dataset name
extension = ("_prefix" if not dataset_extra_csv is None else "")

# compute the IRGs
for approximator in [fx.KNN_approximator, fx.SVR_approximator, fx.AdaBoost_approximator, fx.ANN_approximator]:

    fname = dataset_name + "_" + approximator + extension

    results_bin = os.path.join(results_folder, fname + ".bin")
    results_csv = os.path.join(results_folder, fname  + ".csv")

    # establish relationships if not given
    if not os.path.exists(results_bin):
        c, p = read_dataset(dataset_csv, dataset_extra_csv)
        relations = rbc.Extract_1_to_1_Relations(c, approximator, p)

        with open(results_bin, 'wb') as handle:
            pickle.dump(relations, handle)

    ### everything below saves the data and produces the csvs
    with open(results_bin, 'rb') as handle:
        relations = pickle.load(handle)

    print relations

    fid = None

    pr = ""

    concept_names = []
    csv_lines = []

    for item in relations:
        A, B, W = item

        if A != fid:
            if fid is not None:
                csv_lines.append(pr)
            # below storeds the concept names in proper order
            concept_names.append(A)
            pr = A
            fid = A
        pr = pr + "," + (("%.2f" % W) if not W == 1000.0 else "")
    csv_lines.append(pr)

    csv_lines.insert(0, "," + ",".join([name for name in concept_names]))
    csv_result = "\n".join(csv_lines)

    print csv_result

    with open(results_csv, "w") as f:
        f.write(csv_result)

# compute the average over IRGs for different classes for all relations
all_relations = {}

for approximator in [fx.KNN_approximator, fx.SVR_approximator, fx.AdaBoost_approximator, fx.ANN_approximator]:

    fname = dataset_name + "_" + approximator + extension
    results_bin = os.path.join(results_folder, fname + ".bin")

    with open(results_bin, 'rb') as handle:
        relations = pickle.load(handle)

    for A, B, W in relations:
        # relation go from A to B with weight W
        if A == B:
            continue

        key = A + "->" + B
        if not key in all_relations:
            all_relations[key] = []

        all_relations[key].append(W)

# compute average IRG over different predictive classes for relations
all_relations = [{'relation': key, 'IRG': np.mean(all_relations[key])} for key in all_relations]

all_relations.sort(key= lambda value: value['IRG'], reverse=True)

ranking = "\n".join([value['relation'] + "," + str(value['IRG']) for value in all_relations])

# save the csv
ranking_csv = os.path.join(results_folder, dataset_name + "_ranking" + extension + ".csv")

with open(ranking_csv, "w") as f:
    f.write(ranking)