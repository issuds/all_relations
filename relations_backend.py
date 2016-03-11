'''
Contains functions for processing of all relations
@author: Euler
'''
import csv
import numpy as np
from sklearn.svm import SVR


def Read_CSV_Columns(filename):
    
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

def Relation_Generalization(X,y):
    # establishes how well relation between inputs and outputs can generalize
    # X : input matrix
    # y : output vector
    
    # split data in half
    Xtr, Xts = np.array_split(X,2)
    ytr, yts = np.array_split(y,2)    
    
    # train 
    
    best_predictor = None;
    best_score = -1e+10;
    
    for c in [0.1, 1, 10, 100]:
        for e in [0.1,0.05,0.01]:
            predictor = SVR(C=c, epsilon=e)  
            predictor.fit(Xtr, ytr)
            score = predictor.score(Xtr, ytr)
            if score > best_score:
                best_predictor, best_score = predictor, score;
                
    # evaluate
    result = best_predictor.score(Xts, yts)
    
    return result;

def Extract_1_to_1_Relations(columns, threshold):
    # return arrays of size 3 of the form colx, coly, relation strength
    
    result = [];
    
    for A in columns.keys():
        for B in columns.keys():
            
            if A == B:
                continue;
            
            X =  np.expand_dims( columns[A], axis = 1);
            y = columns[B];
            w = Relation_Generalization(X, y)
            
            if w > threshold:
                print "relation",A,"->",B,"..."
                result.append([A,B,w]);
    
    return result
