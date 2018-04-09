import numpy as np
from pprint import pprint
import os
import json
import pydot
import pandas as pd

fpath = os.path.join('results', 'gender_discrimination.json')

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
graph.write_svg('visualization.svg')

dataframe = dataframe.sort_index(axis=0)
dataframe = dataframe.sort_index(axis=1)

dataframe.to_csv('visualization.csv')
