'''
parses and plots wiki dataset
@author: iaroslav
'''

from dataset_parser import read_dataset
from relations_backend import Extract_1_to_1_Relations, plot_relations
import pickle
import os

### SETTINGS
dataset_name = "wiki"
use_prefix = False
threshold = 1.5
approximator = "SVR"

### CODE START
save_to = dataset_name + ("_prefix.bin" if use_prefix else ".bin")

# establish relationships if not given
if not os.path.exists(save_to):
    c, p = read_dataset(dataset_name, use_prefix)
    relations = Extract_1_to_1_Relations(c, approximator, p)
    
    with open(save_to, 'wb') as handle:
        pickle.dump(relations, handle)

# plot relationships
with open(save_to, 'rb') as handle:
    relations = pickle.load(handle)

print relations

fid = ""
pr = ""

for item in relations:
    A, B, W = item
    if A != fid:
        print pr
        pr = A
        fid = A
    pr = pr + "," + ("%.2f" % W)
print pr

plot_relations(relations, threshold)