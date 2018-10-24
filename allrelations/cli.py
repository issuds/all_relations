"""Command line interface (CLI) for extraction of weights of
all one to one relations of concepts given some dataset. See
https://github.com/InformationServiceSystems/all-relations
for more details.

Usage:
  main.py --help
  main.py --dataset=<path> --model=<class> --saveto=<path> [--userfeatures]

Options:
  --dataset=<file>       Path of the CSV file, containing a dataset.
  --model=<class>        A class of models to use. Supported classes
                         are ['lasso', 'ann', 'knn', 'gbrt', 'tree'].
                         The options correspond to the following
                         models:
                         'lasso' stands for Lasso Regression,
                         'ann' stands for Artificial Neural Network,
                         'knn' stands for k Nearest Neighbors,
                         'gbrt' is Gradient Boosting Regression Trees,
                         'tree' is Regression Decision tree model.
                         Warning: training ANNs takes considerably
                         more time than for other models.
  --saveto=<folder>      A folder where to save the results of computations.
  --userfeatures         Whether to use additional user features, specified
                         in the dataset, in addition to the features of the
                         concepts themselves.
  -h --help              Show this screen.
  --version              Show version.

"""

import pandas as pd
import os

from allrelations.skapi import preprocess_dataset, all_1_to_1
from allrelations.visualization import render_relations

from docopt import docopt


def main():
    arguments = docopt(__doc__, version='Oct 2018, ER18')

    dataset = arguments['--dataset=<file>']
    model = arguments['--model=<class>']
    saveto = arguments['--saveto=<folder>']
    userfeatures = arguments['--userfeatures']

    # read the CSV file of the dataset
    data = pd.read_csv(dataset)

    # separate dataset into concepts and user information
    concepts, respdata = preprocess_dataset(data)

    if not userfeatures:
        respdata = None  # ignore respondent data if required

    # extract all 1 to 1 relations
    relations = all_1_to_1(concepts, prefix=respdata, models_subset=model)

    # name of dataset without .csv at the end
    dfname = os.path.basename(dataset)
    if dfname.endswith('.csv'):
        dfname = dfname[:-4]

    # output name: dataset file name + model name [+ userfeatures] .json
    result_name = dfname + '_' + model
    if userfeatures:
        result_name += '_userfeatures'

    result_path = os.path.join(saveto, result_name + '.json')

    import json

    json.dump(
        relations,
        open(os.path.join('results', result_path+'.json'), 'w'),
        indent=2,
    )

    render_relations(result_path)


if __name__ == "__main__":
    main()
