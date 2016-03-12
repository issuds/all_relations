'''
Contains functions for processing of all relations
@author: Euler
'''
import csv
import numpy as np
from sklearn.svm import SVR
from ff_train import train_evaluate

import networkx as nx
import matplotlib.pyplot as plt
import time

from multiprocessing import Process, Manager

def plot_relations(relations, thr):
    
    
    G=nx.DiGraph()

    for relation in relations:
        if relation[-1] > thr:
            G.add_edge(relation[0],relation[1],weight=relation[2])
    
    
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) ]
    
    pos=nx.spring_layout(G) # positions for all nodes
    nx.draw_networkx_nodes(G,pos,node_size=700)
    nx.draw_networkx_edges(G,pos,edgelist=elarge,
                        width=6)
    
    # labels
    nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
    #nx.draw_networkx_edge_labels(G,pos,font_size=10,font_family='sans-serif')
    
    plt.axis('off')
    plt.show() # display

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

def diff_measure(Y, Yp):
    return np.mean( np.abs( Y - Yp ) )

def improvement_over_guessing(Ytr, Ytst, Ypr):
    pr_obj = diff_measure(Ytst, Ypr)
    
    rnd_objs = []
    for i in range(10):
        I = np.random.choice(len(Ytr), len(Ytst))
        Yrnd = Ytr[I,]
        rnd_objs.append( diff_measure(Ytst, Yrnd) )
    
    rnd_obj = np.mean(rnd_objs)
    
    return rnd_obj / pr_obj

def Relation_Generalization(x,y):
    # establishes how well relation between inputs and outputs can generalize
    # x : input matrix
    # y : output vector
    
    # train 
    best_val = 0.0;
    best_tst = 0;
    for neurons in [5,10,20,40]:
        for layers in [1,2,3,4]:
            val, tst = train_evaluate(x, y, improvement_over_guessing, [neurons, layers])
            if val > best_val:
                best_val = val;
                best_tst = tst;
        
    return best_tst;

def Relation_Generalization_WRP(X, Y, procnum, return_dict):
    w = Relation_Generalization(X, Y)
    return_dict[procnum] = w;

def Extract_1_to_1_Relations(concepts, prefix = None):
    # return arrays of size 3 of the form colx, coly, relation strength
    
    result = [];
    
    idx = 1;
    N = len(concepts);
    avg_time = None
    
    manager = Manager()
    
    for A in concepts.keys():
        
        return_dict = manager.dict()
        jobs = []
        
        start_time = time.time()
        
        for B in concepts.keys():
            
            if A == B:
                continue;
            
            X = concepts[A];
            Y = concepts[B];
            
            if not (prefix is None):
                X = np.column_stack((prefix, X))
            
            p = Process(target=Relation_Generalization_WRP, args=(X,Y,B,return_dict))
            jobs.append(p)
            p.start()
                    
        for proc in jobs:
            proc.join()
        
        for B in return_dict.keys():      
            result.append([A,B, return_dict[B] ]);
        
        #print "relation",A,"->",B,":",w,";",idx,"/",N,"est. time:", avg_time
        #idx = idx + 1
        
        N = N - 1
        
        est_time = (time.time() - start_time)
        avg_time = est_time if avg_time is None else avg_time*0.8 + 0.2*est_time
        print "est. time:", avg_time*N
    
    return result
