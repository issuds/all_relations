'''
Contains functions for processing of all relations
@author: Euler
'''
import csv
import numpy as np
import fitter as fx

import time
from multiprocessing import Pool



def plot_relations(relations, thr):

    import networkx as nx
    import matplotlib.pyplot as plt
    
    G=nx.DiGraph()

    for relation in relations:
        if relation[-1] > thr:
            G.add_edge(relation[0],relation[1],weight=relation[2])
    
    
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) ]
    
    pos=nx.spring_layout(G) # positions for all nodes
    nx.draw_networkx_nodes(G,pos,node_size=9000)
    nx.draw_networkx_edges(G,pos,edgelist=elarge,
                        width=6)
    
    # labels
    nx.draw_networkx_labels(G,pos,font_size=50,font_family='sans-serif')
    #nx.draw_networkx_edge_labels(G,pos,font_size=10,font_family='sans-serif')
    
    plt.axis('off')
    plt.show() # display

def Read_CSV_Columns(filename):
    
    headers = None
    columns = []
    
    with open(filename, 'rb') as csvfile:
        
        spamreader = csv.reader(csvfile, delimiter=',')
        first = True
        
        for row in spamreader:
            
            if first:
                headers = row
                first = False
                for header in headers:
                    columns.append([])
                continue
            
            for i in range(len(row)):
                columns[i].append(float(row[i]))
            
        # convert to numbers
        for i in range(len(columns)):
            columns[i] = np.array(columns[i])
        
        result = {}
        
        # create dictionary with headers
        for hdr, clm in zip(headers, columns):
            result[hdr] = clm
        
    return result

def diff_measure(Y, Yp):
    #return np.mean( np.abs( Y - Yp ) )
    return np.sqrt(np.mean((Y - Yp)**2))

def improvement_over_guessing(Ytr, Ytst, Ypr):
    pr_obj = diff_measure(Ytst, Ypr)
    
    rnd_objs = []
    for i in range(100):
        I = np.random.choice(len(Ytr), len(Ytst))
        Yrnd = Ytr[I,]
        rnd_objs.append( diff_measure(Ytst, Yrnd) )
    
    rnd_obj = np.mean(rnd_objs)
    
    return rnd_obj / pr_obj

def select_best_from(params):

    pool = Pool(12)
    results = pool.map(fx.train_evaluate, params)
    pool.close()

    val, tst, spc = max(results, key=lambda r: r[0])

    return val, tst, spc


def Relation_Generalization(x,y, approximator):
    # establishes how well relation between inputs and outputs can generalize
    # x : input matrix
    # y : output vector
    
    
    # train 
    params = []

    if approximator == fx.ANN_approximator:
        for neurons in 2 ** np.arange(1,10):
            for layers in [1,2,3,4,5]:
                for i in range(3):
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': improvement_over_guessing,
                        'params': {'neurons': neurons, 'layers': layers}
                    })
    else: # select parameters for every output separately, and then evalueate them all together

        param = {
                    'class':approximator,
                    'x':x,
                    'y':y,
                    'performance measure': improvement_over_guessing,
                    'params': {'n_estimators': 2 ** pw, 'learning_rate': lr}
                })

    elif approximator == fx.KNN_approximator:
        for k in range(1, len(x) // 2, 5):
            for msr in ['minkowski']:
                for weights in ['uniform', 'distance']:
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': improvement_over_guessing,
                        'params': {'n_neighbors': k, 'metric': msr, 'weights': weights}
                    })
    else:
        raise BaseException('approximator type not understood')
    
    pool = Pool()
    results = pool.map(fx.train_evaluate, params)
    pool.close()
    
    best_val = 0.0
    best_tst = 0

    for val, tst, spcs in results:
        # spcs is necessary for plots if any
        if val > best_val:
            best_val = val
            best_tst = tst
    
    
    return best_tst


def Relation_Generalization_WRP(X, Y, procnum, return_dict):
    w = Relation_Generalization(X, Y)
    return_dict[procnum] = w


def Extract_1_to_1_Relations(concepts, approximator):
    # return arrays of size 3 of the form colx, coly, relation strength
    
    result = {}

    N = len(concepts) ** 2
    avg_time = None

    # given A, predict B
    for A in concepts.keys(): # relation from A ...

        result[A] = {}

        for B in concepts.keys(): # to B
                        
            if A == B:
                result[A][B] = None
                continue
            
            start_time = time.time()
            
            X = concepts[A]
            Y = concepts[B]
            W = Relation_Generalization(X, Y, approximator)
            result[A][B] = W
        
            est_time = (time.time() - start_time)
            avg_time = est_time if avg_time is None else avg_time*0.8 + 0.2*est_time
            N = N - 1
            
            print "relation",A,"->",B,":",W," est. time:", avg_time*N

    return result
