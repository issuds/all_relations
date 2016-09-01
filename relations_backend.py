'''
Contains functions for processing of all relations
@author: Euler
'''
import csv
import numpy as np
from sklearn.svm import SVR
import fit_approximator as fx

import networkx as nx
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool



def plot_relations(relations, thr):
    
    
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

def Relation_Generalization(x,y, approximator):
    # establishes how well relation between inputs and outputs can generalize
    # x : input matrix
    # y : output vector
    
    
    # train 
    params = []
    results = []

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
    elif approximator == fx.SVR_approximator:
        for C in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
            for gamma in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
                for eps in 2.0 ** np.array([-10,-8,-6,-4,-2,0]):
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': improvement_over_guessing,
                        'params': {'C': C, 'gamma': gamma, 'epsilon': eps}
                    })
    elif approximator == fx.AdaBoost_approximator:
        for pw in [2,3,4,5,6,7,8,9,10]:
            for lr in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
                params.append({
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
    
    pool = Pool(12)
    results = pool.map(fx.train_evaluate, params)
    pool.close()
    
    best_val = 0.0;
    best_tst = 0;

    """
    avg = { }
    for val, tst, spcs in results:
        key = spcs['n_neighbors']
        if key not in avg:
            avg[key] = []
        if spcs['weights'] == 'uniform':
            avg[key].append(val)

    x, y = [], []

    for key in avg:
        x.append(key)
        y.append(np.max(avg[key]))

    from matplotlib import pyplot as plt

    plt.figure(figsize=(4,3))
    plt.scatter(x, y)
    plt.xlabel("Number of neighbors")
    plt.ylabel("IRG")

    plt.axis([-10, 310, 1.1, 1.6])

    plt.grid()
    plt.show()
    """

    for val, tst, spcs in results:
        # spcs is necessary for plots if any
        if val > best_val:
            best_val = val;
            best_tst = tst;
    
    
    return best_tst;

def Relation_Generalization_WRP(X, Y, procnum, return_dict):
    w = Relation_Generalization(X, Y)
    return_dict[procnum] = w;

def Extract_1_to_1_Relations(concepts, approximator, prefix = None):
    # return arrays of size 3 of the form colx, coly, relation strength
    
    result = [];
    
    idx = 0;
    N = len(concepts) ** 2;
    avg_time = None
    
    for A in concepts.keys():
        
        for B in concepts.keys():
                        
            if A == B:
                result.append([A,B, 1000.0 ]);
                continue
            
            start_time = time.time()
            
            X = concepts[A];
            Y = concepts[B];
            W = Relation_Generalization(X, Y, approximator)
            result.append([A, B, W]);
        
            est_time = (time.time() - start_time)
            avg_time = est_time if avg_time is None else avg_time*0.8 + 0.2*est_time
            N = N - 1
            
            print "relation",A,"->",B,":",W,"; est. time:", avg_time*N
                    
    
    return result
