"""
Example usage of the api.
Use it in command line regime.
"""

if __name__ == "__main__":

    import numpy as np
    import pandas as ps

    from backend.skapi import pandas_to_concepts, all_1_to_1, all_n_to_1

    from datasets import read_gender_discrimination_dataset, read_utaut

    reader_func = read_gender_discrimination_dataset

    survey, extra_data = read_gender_discrimination_dataset()
    concepts = pandas_to_concepts(survey)
    #concepts = {c:concepts[c] for c in ['Q1_', 'Q3_']}

    prefix = extra_data
    prefix = None

    # relations = all_n_to_1(concepts)
    relations = all_1_to_1(concepts, prefix=prefix, models_subset='linear')

    # sort from highest weight to the lowest weight
    relations.sort(reverse=True, key=lambda x: x[-1])

    name = "gender_disc_test.json"

    import os
    import json

    json.dump(
        relations,
        open(os.path.join('results', name), 'w'),
        indent=2,
    )

    from pprint import pprint

    pprint(relations)
