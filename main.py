import pandas as ps

from backend.skapi import pandas_to_concepts, all_1_to_1, all_n_to_1

data = ps.read_csv('datasets/wiki.csv')
data = ps.DataFrame(data, dtype='float')
data = data.dropna()
concepts = pandas_to_concepts(data)

relations = all_n_to_1(concepts)
#relations = all_1_to_1(concepts)

import os
import pickle as pc

pc.dump(relations, open(os.path.join('results', 'nto1.b'), 'wb'))