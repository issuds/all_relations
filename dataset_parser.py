'''
Parses the dataset consisting of 
@author: iaroslav
'''

import numpy as np
import csv
import os

def file_to_columns(filename):
    
    headers = None
    columns = []
    
    with open(filename, 'rb') as csvfile:
        
        spamreader = csv.reader(csvfile, delimiter=',')
        first = True;
        
        for row in spamreader:
            
            if first:
                headers = row;
                first = False;
                for header in headers:
                    columns.append([])
                continue;
            
            for i in range(len(row)):
                columns[i].append(float(row[i]))
            
        # convert to numbers
        for i in range(len(columns)):
            columns[i] = np.array(columns[i]);
        
        result = {}
        
        # create dictionary with headers
        for hdr, clm in zip(headers, columns):
            result[hdr] = clm;
        
    return result

def read_dataset(name, load_prefix = True):
    """
    It is assumed that the dataset is located in the datasets folder
    """
    
    folder = "datasets"
    cfile = os.path.join(folder, name + "_concepts.csv")
    pfile = os.path.join(folder, name + "_prefix.csv")
    
    prefix = None
    
    if os.path.exists(pfile) and load_prefix:# concatenate prefix into matrix
        columns = file_to_columns(pfile)
        prefix = [value for value in columns.values()]
        prefix = np.column_stack(prefix)
    
    # create list of concepts
    data = file_to_columns(cfile)
    concepts = {}
    
    for column in data.keys():
        name = column[:-1]
        if name in concepts:
            concepts[name].append( data[column] )
        else:
            concepts[name] = [data[column]]
    
    # concatenate concept features
    for name in concepts.keys():
        concepts[name] = np.column_stack(concepts[name])
        
    return concepts, prefix
    
        
    
    
            
    