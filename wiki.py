'''
parses and plots wiki dataset
@author: iaroslav
'''

from dataset_parser import read_dataset
import relations_backend as rbc
import fit_approximator as fx
import pickle
import os

### SETTINGS
dataset_name = "wiki"
use_prefix = False
threshold = 1.5

#for approximator in [fx.KNN_approximator, fx.SVR_approximator, fx.AdaBoost_approximator, fx.ANN_approximator]:
for approximator in [fx.SVR_approximator, fx.AdaBoost_approximator, fx.ANN_approximator]:

    ### CODE START
    fname = dataset_name + "_" + approximator + ("_prefix" if use_prefix else "")

    results_array_file = fname + ".bin"
    results_csv = fname + ".csv"

    # establish relationships if not given
    if not os.path.exists(results_array_file):
        c, p = read_dataset(dataset_name, use_prefix)
        relations = rbc.Extract_1_to_1_Relations(c, approximator, p)

        with open(results_array_file, 'wb') as handle:
            pickle.dump(relations, handle)

    # plot relationships
    with open(results_array_file, 'rb') as handle:
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
        pr = pr + "," + ( ("%.2f" % W) if not W == 1000.0 else "" )
    csv_lines.append(pr)

    csv_lines.insert(0, "," + ",".join([name for name in concept_names]))
    csv_result = "\n".join(csv_lines)

    print csv_result

    with open(results_csv, "w") as f:
        f.write(csv_result)

    #plot_relations(relations, threshold)