"""
Example usage of the api.
Use it in command line regime.
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', nargs="?", default='read_gender_discrimination_dataset',
        type=str, help="Name of the dataset function to use form "
                       "the __init__.py in 'datasets' folder")

    parser.add_argument(
        '--prefix', action='store_true', help="Whether to use descriptors of a person.")

    args = parser.parse_args()

    import numpy as np
    import pandas as ps

    from backend.skapi import pandas_to_concepts, all_1_to_1, all_n_to_1

    from datasets import read_gender_discrimination_dataset, read_utaut

    reader_func = read_gender_discrimination_dataset

    survey, extra_data = read_gender_discrimination_dataset()
    concepts = pandas_to_concepts(survey)

    prefix = None
    prefix = extra_data

    # relations = all_n_to_1(concepts)
    relations = all_1_to_1(concepts, prefix=prefix)

    # sort from highest weight to the lowest weight
    relations.sort(reverse=True, key=lambda x: x[-1])

    name = reader_func.__name__ + "_prefix.json"

    import os
    import json

    json.dump(
        relations,
        open(os.path.join('results', name), 'w'),
        indent=2,
    )

    from pprint import pprint

    pprint(relations)
