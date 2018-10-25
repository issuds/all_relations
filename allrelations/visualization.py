import numpy as np
import os
import json
import pydot
import pandas as pd


def render_relations(results_json, min_weight_edge=0.0):
    """
    Converts obtained set of relations to CSV and to .SVG formats,
    for ease of comprehension. The .csv and .svg files are stored
    in the same folder as results_json, with the file name of
    results json where the extension is switched to csv or svg
    accordingly.

    Parameters
    ----------
    results_json: str
        Path where the results json is stored.

    min_weight_edge: float
        Minimum value of the edge weight, at which to visualize
        the edges of the relations graph.
    """

    relations = json.load(open(results_json, 'r'))
    relations.sort(reverse=True, key=lambda x: x[-1])

    dataframe = pd.DataFrame()
    graph_data = ["digraph {"]

    for A, B, w in relations:
        for a in A:
            for b in B:
                # weight of the relationship
                w = np.round(w, 3)

                # relation, from a to b
                an, bn = a, b

                # write the relation in the csv
                dataframe.at[an, bn] = w

                # skip the weights which are too small
                if w < min_weight_edge:
                    continue

                line = '"%s" -> "%s"[label="%s"]' % (an, bn, w)
                graph_data.append(line)

    graph_data.append("}")
    graph_data = "\n".join(graph_data)

    # path except for extension where to store the visualizations.
    fpath = results_json
    if fpath.endswith('.json'):
        fpath = fpath[:-5]

    # generate visualization usin pydot
    (graph,) = pydot.graph_from_dot_data(graph_data)
    graph.write_svg(fpath + '.svg')

    dataframe = dataframe.sort_index(axis=0)
    dataframe = dataframe.sort_index(axis=1)

    # make clear from where the relationship starts
    idx = dataframe.index.tolist()
    idx = ["From " + v for v in idx]
    dataframe.index = idx
    dataframe.to_csv(fpath + '.csv')