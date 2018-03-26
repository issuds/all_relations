import numpy as np
from pprint import pprint
import os
import json
import pydot

fpath = os.path.join('results', 'gender_discrimination.json')

# optional abbreviations json, which contains descriptions to concept abbreviations
# set to None if no are available
abbs = os.path.join('datasets', 'gender_discrimination_workplace.info')
if abbs:
    abbs = json.load(open(abbs, 'r'))

relations = json.load(open(fpath, 'r'))
relations.sort(reverse=True, key=lambda x: x[-1])

graph_data = ["digraph {"]

for A, B, w in relations:
    for a in A:
        for b in B:
            an, bn = a, b

            if abbs:
                an = abbs[an]
                bn = abbs[bn]

            w = np.round(w, 3)

            line = '"%s" -> "%s"[label="%s"]' % (an, bn, w)
            graph_data.append(line)

graph_data.append("}")

graph_data = "\n".join(graph_data)

# generate dot file
(graph,) = pydot.graph_from_dot_data(graph_data)
graph.write_svg('visualization.svg')