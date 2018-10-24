import pandas as pd
import os
import numpy as np

# location of script, as well as datasets
script_path = os.path.dirname(os.path.realpath(__file__))


def read_gender_discrimination_dataset():
    path = os.path.join(script_path, 'gender_discr_train.csv')
    data = pd.read_csv(path)

    #data = data[np.random.rand(len(data)) < 0.1]

    extra_data = data[['V1', 'PAYS', 'SEX', 'SexRec', 'YEAR']]

    # one hot encoded values
    dummies = pd.get_dummies(data['V3'])

    extra_data = pd.concat([extra_data, dummies], axis=1)

    survey = pd.DataFrame()
    survey['Q1_1'] = data['Q1bRec']
    survey['Q1_2'] = data['Q1cRec']
    survey['Q2_1'] = data['Q2bRec']
    survey['Q3_1'] = data['Q3aRec']
    survey['Q3_2'] = data['Q3bRec']

    # separate prefix vs actual survey data
    #I = np.random.rand(len(data)) < 0.1
    #data = data[I]

    survey = survey.replace(' ', np.nan)
    extra_data = extra_data.replace(' ', np.nan)

    return survey, extra_data


def read_utaut():
    path = os.path.join(script_path, 'utaut.csv')
    data = pd.read_csv(path)
    data = pd.DataFrame(data, dtype='float')
    return data, None


def read_wiki():
    path = os.path.join(script_path, 'wiki_hicss.csv')
    data = pd.read_csv(path)
    data = pd.DataFrame(data, dtype='float')
    return data, None