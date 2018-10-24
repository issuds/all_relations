"""
This procedure is used to split the gender discrimination dataset into training and testing parts.
Training part: random 80% of the data;
Testing part: rest of 20% of the data.
"""

import numpy as np
import pandas as pd

data = pd.read_csv('gender_discrimination_workplace.csv', sep=';')

# the value of "true" is encountered in I with 80% of chance
I = np.random.rand(len(data)) < 0.8

data_train = data[I]
data_test = data[~I]

print("Duplicates:", np.sum(pd.concat([data_train, data_test]).duplicated()), np.sum(data.duplicated()))
print("Train len:", len(data_train))
print("Test len:", len(data_test))
print("Total len:", len(data))

data_train.to_csv('gender_discr_train.csv', index=False)
data_test.to_csv('gender_discr_test.csv', index=False)