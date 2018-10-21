"""Executes training of the all - relations model to replicate
results in the "Justifying gender discrimination in the
workplace: The mediating role of motherhood myths" study.

Usage:
  main.py --help
  main.py <model>
  main.py <model> --prefix

Options:
  <model>       A class of models to use. Supported classes
                are ['lasso', 'ann', 'knn', 'gbrt', 'tree']. 
                gbrt stands for the gradient boosted regression
                tree model.
  --prefix      Specifies whether to use the features of the 
                respondent to particular survey question in
                order to predict the answers to other questions.
                Could potentiall lead to more accurate models,
                but also to overfitting. Possible values are
                true or false.
                This was called prefix as historically the 
                extra features of the respondent were added
                as extra columns at the beginning of the main
                response matrix, thus adding a "prefix" to the
                features that represent the 
  -h --help     Show this screen.
  --version     Show version.

"""

from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__, version='ER18 version 2.0')
    subset = arguments['<model>']
    prefix_notify = '_prefix' if arguments['--prefix'] else ""
    print(prefix_notify)

    import numpy as np
    import pandas as ps

    from backend.skapi import pandas_to_concepts, all_1_to_1, all_n_to_1
    from visualize import render
    from datasets import read_gender_discrimination_dataset, read_utaut

    reader_func = read_gender_discrimination_dataset

    survey, extra_data = read_gender_discrimination_dataset()
    concepts = pandas_to_concepts(survey)
    #concepts = {c:concepts[c] for c in ['Q1_', 'Q3_']}

    prefix = None
    if arguments['--prefix']:
        prefix = extra_data
        print('Using user features of a shape ' + str(extra_data.shape))

    relations = all_1_to_1(concepts, prefix=prefix, models_subset=[subset])

    # sort from highest weight to the lowest weight
    relations.sort(reverse=True, key=lambda x: x[-1])

    name = "gender_disc_test"+subset+".json"

    import os
    import json

    json.dump(
        relations,
        open(os.path.join('results', name), 'w'),
        indent=2,
    )

    from pprint import pprint

    render(name, subset + prefix_notify)