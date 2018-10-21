"""Executes training of the all - relations model to replicate
results in the "Justifying gender discrimination in the
workplace: The mediating role of motherhood myths" study.

Usage:
  main.py --help
  main.py <model>
  main.py <model> --prefix

Options:
  <model>       A class of models to use. Supported classes
                are ['lasso', 'ann', 'knn', 'gbrt']. gbrt 
                stands for the gradient boosted regression
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

    subset = subset + prefix_notify
    name = "gender_disc_test"+subset+".json"

    import os
    import json

    json.dump(
        relations,
        open(os.path.join('results', name), 'w'),
        indent=2,
    )

    from pprint import pprint

    pprint(relations)

    # make visualizations here

    import numpy as np
    from pprint import pprint
    import os
    import json
    import pydot
    import pandas as pd

    fpath = os.path.join('results', name)

    skip_w = -100.0

    # optional abbreviations json, which contains descriptions to concept abbreviations
    # set to None if no are available
    abbs = True
    if abbs:
        abbs = os.path.join('datasets', 'gender_discrimination_workplace.info')
        abbs = json.load(open(abbs, 'r'))

    relations = json.load(open(fpath, 'r'))


    relations.sort(reverse=True, key=lambda x: x[-1])

    dataframe = pd.DataFrame()
    graph_data = ["digraph {"]


    for A, B, w in relations:
        for a in A:
            for b in B:
                w = np.round(w, 3)

                an, bn = a, b

                if abbs:
                    an = abbs[an]
                    bn = abbs[bn]

                # write the relation in the csv
                dataframe.at[an, bn] = w

                if w < skip_w:
                    continue

                line = '"%s" -> "%s"[label="%s"]' % (an, bn, w)
                graph_data.append(line)

    graph_data.append("}")

    graph_data = "\n".join(graph_data)

    # generate dot file
    (graph,) = pydot.graph_from_dot_data(graph_data)
    graph.write_svg('visualization_'+subset+'.svg')

    dataframe = dataframe.sort_index(axis=0)
    dataframe = dataframe.sort_index(axis=1)

    # make clear from where the relationship starts
    idx = dataframe.index.tolist()
    idx = ["From " + v for v in idx]
    dataframe.index = idx

    dataframe.to_csv('visualization_'+subset+'.csv')