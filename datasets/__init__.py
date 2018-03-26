import pandas as pd
import os
import numpy as np

# location of script, as well as datasets
script_path = os.path.dirname(os.path.realpath(__file__))


def read_gender_discrimination_dataset():
    path = os.path.join(script_path, 'gender_discrimination_workplace.csv')
    data = pd.read_csv(path)
    data = data.replace(8, np.nan)
    data = data.replace(9, np.nan)
    data = pd.DataFrame(data, dtype='float')
    #I = np.random.rand(len(data)) < 0.1
    #data = data[I]
    return data


def read_utaut():
    path = os.path.join(script_path, 'utaut.csv')
    data = pd.read_csv(path)
    data = pd.DataFrame(data, dtype='float')
    return data


def read_wiki():
    path = os.path.join(script_path, 'wiki.csv')
    data = pd.read_csv(path)
    data = pd.DataFrame(data, dtype='float')
    return data
