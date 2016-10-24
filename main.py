'''
parses and plots wiki dataset
@author: iaroslav
'''

import pandas as ps
from backend import relations as rbc
from backend import fitter as fx
from backend import parser as drd

import pandas as ps
import pickle
import os
import numpy as np

if __name__ == "__main__": # all the code is in main so that it works properly on windows in shell
    """
    Settings:
    """

    dataset_csv = "datasets/utaut.csv" # see wiki.csv to understand format used
    results_folder = "results" # this folder should exist
    model_classes = [fx.KNN_approximator, fx.SVR_approximator, fx.AdaBoost_approximator, fx.ANN_approximator] # - use if TensorFlow is installed

    """
    What happens below is as follows:
    1. The dataset is split into statistical learning and evaluation parts. Statistical learning
        part is used to establish weights for all pairs of concepts, and evaluation part can be
        used to evaluate the model extracted from the weights for all pairs of conecpts.
        The split of the dataset is stored in the results_folder.

    2. The statistical learning part is used to come up with weights for every pair of concepts.

    3. For the computed relation weights for every selected class, the average is computed, which
        is used to come up with an order of relations from the strongest ones to weakest ones.

    """

    """ 1. Split the data"""

    dataset_name = os.path.basename(dataset_csv)[:-4] # remove the .csv at the end of dataset name

    # create folder with dataset name
    results_folder = os.path.join(results_folder, dataset_name)

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # load and split the dataset: stat. training part 2/3, rest eval part
    dataset = ps.read_csv(dataset_csv, skip_blank_lines = True)

    # replace missing values with -1
    dataset = dataset.replace("?", "-1")
    dataset = dataset.replace("null", "-1")
    dataset = dataset.replace("Null", "-1")
    dataset = dataset.fillna(-1)

    stat_train_part = dataset[:-int(len(dataset) / 3)]
    eval_part = dataset[-int(len(dataset) / 3):]

    # save dataset parts
    eval_part.to_csv(os.path.join(results_folder, "for_evaluation.csv"), index = False)

    dataset_csv = os.path.join(results_folder, "dataset.csv")
    stat_train_part.to_csv(dataset_csv, index = False)

    """ 2. Compute the IRGs"""
    for model_class in model_classes:

        fname = dataset_name + "_" + model_class

        results_bin = os.path.join(results_folder, fname + ".bin")
        results_csv = os.path.join(results_folder, fname + ".csv")

        # establish relationships if not given
        if not os.path.exists(results_bin):
            rlt = rbc.Extract_1_to_1_Relations(drd.read_dataset(dataset_csv), model_class)

            # store results_paper in a .bin file
            with open(results_bin, 'wb') as handle:
                pickle.dump(rlt, handle)

        # load computed results_paper
        with open(results_bin, 'rb') as handle:
            rlt = pickle.load(handle)

        # 2d array representing csv table
        names = rlt.keys()
        header = [[""] + names]
        rows = header + [[A] + ["%.2f" % rlt[A][B] if not rlt[A][B] is None else "" for B in names] for A in names]

        # convert csv array to string
        csv_result = "\n".join([",".join(row) for row in rows])

        # save the csv
        with open(results_csv, "w") as f:
            f.write(csv_result)

        print csv_result

    """ 3. compute the average over IRGs for different classes for all relations """
    all_relations = {}

    for model_class in model_classes:

        fname = dataset_name + "_" + model_class
        results_bin = os.path.join(results_folder, fname + ".bin")

        with open(results_bin, 'rb') as handle:
            rlt = pickle.load(handle)

        for A in rlt.keys():
            for B in rlt.keys():
                # relation go from A to B with weight W
                if A == B:
                    continue

                W = rlt[A][B]

                key = A + "->" + B
                if not key in all_relations:
                    all_relations[key] = []

                all_relations[key].append(W)

    # compute average IRG over different predictive classes for relations
    all_relations = [{'relation': key, 'IRG': np.mean(all_relations[key])} for key in all_relations]
    all_relations.sort(key= lambda value: value['IRG'], reverse=True)

    ranking = "\n".join([value['relation'] + "," + str(value['IRG']) for value in all_relations])

    # save the csv
    ranking_csv = os.path.join(results_folder, dataset_name + "_ranking.csv")

    with open(ranking_csv, "w") as f:
        f.write(ranking)
