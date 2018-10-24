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

from allrelations.interface import extract_1_to_1
from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__, version='Oct 2018, ER18')

    extract_1_to_1(
        dataset = arguments['--dataset=<file>'],
        model = arguments['--model=<class>'],
        saveto = arguments['--saveto=<folder>'],
        use_resp_data= arguments['--userfeatures']
    )