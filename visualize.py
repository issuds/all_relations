from pprint import pprint
import os
import pickle as pc

results = pc.load(open(os.path.join('results', 'nto1.b'), 'rb'))

results.sort(reverse=True, key=lambda x: x[-1])

pprint(results)