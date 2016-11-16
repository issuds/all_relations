'''
Contains functions for processing of all relations
@author: Euler
'''
import csv
import numpy as np
import time
from multiprocessing import Pool
from sklearn.metrics import r2_score
import os

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

def Rsq(Ytr, Ytst, Ypr):
    """
    Computes improvement over random guess evaluation metric for model predictions

    :param Ytr: outputs on which the model was trained - is used for sampling here
    :param Ytst: ground truth - test outputs
    :param Ypr: predictions of the model
    :return: irg of predictions
    """
    return r2_score(Ytst, Ypr, multioutput='uniform_average')

def IRG(Ytr, Ytst, Ypr):
    """
    Computes R^2 error of model predictions

    :param Ytr: outputs on which the model was trained - is NOT used here
    :param Ytst: ground truth - test outputs
    :param Ypr: predictions of the model
    :return: R^2 of predictions
    """
    pr_obj = diff_measure(Ytst, Ypr)

    rnd_objs = []
    for i in range(100):
        I = np.random.choice(len(Ytr), len(Ytst))
        Yrnd = Ytr[I,]
        rnd_objs.append(diff_measure(Ytst, Yrnd))

    rnd_obj = np.mean(rnd_objs)

    return rnd_obj / pr_obj

# names
ANN_approximator = "ANN"
SVR_approximator = "SVR"
AdaBoost_approximator = "AdaBoost"
KNN_approximator = "KNN"
Linear_approximator = "Lasso"
Tree_approximator = "Tree"

def split_matrix(X, tr = 0.5, vl = 0.25):
    """ splits data into training, validation and testing parts
    """
    X, Xv, Xt = X[:int(len(X)*tr)], X[int(len(X)*tr):int(len(X)*(tr + vl))], X[int(len(X)*(tr + vl)):]
    return X, Xv, Xt

def prepare_data(x, y):
    """
    Does normalization of inputs and outputs
    :param x: inputs
    :param y: outputs
    :return:
    """
    X, Xv, Xt = split_matrix(x)
    Y, Yv, Yt = split_matrix(y)

    # normalize
    xm = np.mean(X, axis=0)
    ym = np.mean(Y, axis=0)

    X = X - xm
    Y = Y - ym

    Xv = Xv - xm
    Yv = Yv - ym

    Xt = Xt - xm
    Yt = Yt - ym

    xd = np.std(X, axis=0)
    yd = np.std(Y, axis=0)

    X = X / xd
    Y = Y / yd

    Xv = Xv / xd
    Yv = Yv / yd

    Xt = Xt / xd
    Yt = Yt / yd

    return X, Y, Xv, Yv, Xt, Yt

def fit_report_ANN(params):
    """
    Reports performance with ANN.

    :param params: specification of neural network
    :return: performance estimate
    """
    # all imports are done here so that if TF is not installed rest of code can work

    import tensorflow as tf

    class ffnn():

        def __init__(self, ipt):
            self.ipt = ipt
            self.output_size = ipt.get_shape()[1].value
            self.output = self.ipt

        def add_dense(self, size):
            W = tf.Variable(tf.truncated_normal([self.output_size, size], stddev=0.05))
            b = tf.Variable(tf.truncated_normal([size], stddev=0.05))

            self.output = tf.matmul(self.output, W) + b
            self.output_size = self.output.get_shape()[1].value

        def add_activation(self, activation):
            if activation == "relu":
                self.output = tf.maximum(self.output, 0.0)

        def get_output(self):
            return self.output

    X, Y, Xv, Yv, Xt, Yt = prepare_data(params['x'], params['y'])
    measure = params['performance measure']

    sess = tf.InteractiveSession()
    ipt = tf.placeholder("float", shape=[None, X.shape[-1]])

    ff = ffnn(ipt)

    specs = params['params']

    for l in range(specs['layers']):
        ff.add_dense(specs['neurons'])
        ff.add_activation("relu")

    ff.add_dense(Y.shape[-1])
    otp = ff.get_output()
    gtr = tf.placeholder("float", shape=[None, Y.shape[-1]])

    loss = tf.reduce_mean(((otp - gtr) ** 2))
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    sess.run(tf.initialize_all_variables())

    # saver = tf.train.Saver()
    # stopping criterion: improvement on validation set is no more

    finish_checks = 30
    # best_model_name = "best_model.temp"

    finished = finish_checks
    best_acc = 0.0
    test_acc = 0.0

    while finished > 0:
        inputs = {ipt: X, gtr: Y}
        train_step.run(feed_dict=inputs)

        inputs_val = {ipt: Xv, gtr: Yv}
        Ypr = otp.eval(feed_dict=inputs_val)
        local_acc = measure(Y, Yv, Ypr)

        if local_acc > best_acc:
            finished = finish_checks
            best_acc = local_acc

            inputs_tst = {ipt: Xt, gtr: Yt}
            Ypr = otp.eval(feed_dict=inputs_tst)
            test_acc = measure(Y, Yt, Ypr)

            # saver.save(sess, best_model_name)
        else:
            finished = finished - 1

    # load the best nn model

    sess.close()

    return best_acc, test_acc, specs

def render_trees(specs, X, Y, A, B, folder):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import export_graphviz
    from cStringIO import StringIO

    trees_folder = os.path.join(folder, "trees")

    if not os.path.exists(trees_folder):
        os.mkdir(trees_folder)

    idx = 0
    renderer_html = os.path.join(os.path.dirname(__file__), "render_pydot.html")
    renderer = open(renderer_html,'r').read()

    for column in range(Y.shape[1]):
        idx += 1
        regr = DecisionTreeRegressor(**specs)
        regr.fit(X, Y[:, column])

        dot_data = StringIO()
        export_graphviz(regr, out_file=dot_data)
        graph = dot_data.getvalue()

        name = A + "-to-" + B + "-" + str(idx)

        tree = renderer.replace("Relation", name).replace('"digraph { a -> b; }"', "`" + graph + "`")
        fname = os.path.join(trees_folder, name + ".html")

        with open(fname, 'w') as f:
            f.write(tree)


def fit_report_sklearn(params, apx):

    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Lasso
    from sklearn.tree import DecisionTreeRegressor

    accepted_sklearn_classes = {}
    accepted_sklearn_classes[SVR_approximator] = SVR
    accepted_sklearn_classes[AdaBoost_approximator] = AdaBoostRegressor
    accepted_sklearn_classes[KNN_approximator] = KNeighborsRegressor
    accepted_sklearn_classes[Linear_approximator] = Lasso
    accepted_sklearn_classes[Tree_approximator] = DecisionTreeRegressor

    X, Y, Xv, Yv, Xt, Yt = prepare_data(params['x'], params['y'])
    measure = params['performance measure']
    specs = params['params']

    # for every column in Y, train the SVR, evaluate its performance.
    # return the average of performances.

    vals = []
    tsts = []

    if not apx in accepted_sklearn_classes:
        raise BaseException("This sklearn model is not added to accepted sklearn models. Please add it. ")

    for column in range(Y.shape[1]):
        regr = accepted_sklearn_classes[apx](**specs)
        regr.fit(X, Y[:, column])

        Yp = regr.predict(Xv)
        vals.append(Yp)

        Yp = regr.predict(Xt)
        tsts.append(Yp)

    Yp = np.column_stack(vals)
    val_measure = measure(Y, Yv, Yp)

    Yp = np.column_stack(tsts)
    tst_measure = measure(Y, Yt, Yp)

    return val_measure, tst_measure, specs

def train_evaluate((params)):
    """
    Trains and evaluates particular params['class'] of models on params['x'] inputs
    and params['y'] outputs, with particular params['params'] configuration of hyperparameters

    The brackets in argument (params) are necessary so that it can be applied to the pool.map function.

    :return: a scalar, which is an estimate of performance of model class with given hyperparameters
    """

    # train and evaluate 2 layer nn
    approximator = params['class']

    if approximator == ANN_approximator:
        return fit_report_ANN(params)
    else:
        return fit_report_sklearn(params, approximator)

def Relation_Generalization(x,y, measure, approximator, A=None, B=None, results_folder=None):
    """
    Establishes how well relation between inputs and outputs can generalize.

    :param x: Input matrix
    :param y: Output vector
    :param measure: evaluation metric for performance of model (eg IRG or R^2)
    :param approximator: class of models to use
    :param A, B : names of input and output concept, this is necessary for trees only
    :return:
    """
    #
    # x : input matrix
    # y : output vector
    
    
    # train 
    params = []
    results = []

    if approximator == ANN_approximator:
        for neurons in 2 ** np.arange(1,10):
            for layers in [1,2,3,4,5]:
                for i in range(3):
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': measure,
                        'params': {'neurons': neurons, 'layers': layers}
                    })
    elif approximator == SVR_approximator:
        for C in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
            for gamma in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
                for eps in 2.0 ** np.array([-10,-8,-6,-4,-2,0]):
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': measure,
                        'params': {'C': C, 'gamma': gamma, 'epsilon': eps}
                    })
    elif approximator == AdaBoost_approximator:
        for pw in [2,3,4,5,6,7,8,9,10]:
            for lr in 2.0 ** np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]):
                params.append({
                    'class':approximator,
                    'x':x,
                    'y':y,
                    'performance measure': measure,
                    'params': {'n_estimators': 2 ** pw, 'learning_rate': lr}
                })

    elif approximator == KNN_approximator:
        for k in range(1, len(x) // 2, 5):
            for msr in ['minkowski']:
                for weights in ['uniform', 'distance']:
                    params.append({
                        'class':approximator,
                        'x':x,
                        'y':y,
                        'performance measure': measure,
                        'params': {'n_neighbors': k, 'metric': msr, 'weights': weights}
                    })
    elif approximator == Linear_approximator:
        for alpha in np.logspace(-4,4):
            params.append({
                'class':approximator,
                'x':x,
                'y':y,
                'performance measure': measure,
                'params': {'alpha': alpha}
            })
    elif approximator == Tree_approximator:
        for depth in range(1,8):
            params.append({
                'class':approximator,
                'x':x,
                'y':y,
                'performance measure': measure,
                'params': {'max_depth': depth}
            })
    else:
        raise BaseException('approximator type not understood')
    
    pool = Pool()
    results = pool.map(train_evaluate, params)
    pool.close()
    
    best_val = -(10.0 ** 10.0)
    best_tst = 0
    best_parameters = None

    for val, tst, spcs in results:
        # spcs is necessary for plots if any
        if val > best_val:
            best_val = val
            best_tst = tst
            best_parameters = spcs

    #if approximator == Tree_approximator and not results_folder is None:
    #    render_trees(best_parameters, x, y, A, B, results_folder)
    
    return best_tst

def Extract_1_to_1_Relations(concepts, evaluation_metric, approximator, results_folder=None):
    """

    :param concepts: set of matricies with same number of rows, where every matrix corresponds to a concept
    :param evaluation_metric: how to measure quality of approximator
    :param approximator: class of models to consider
    :param results_folder: this is only necessary if you want to use decision trees
    :return:
    """
    
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
            W = Relation_Generalization(X, Y, evaluation_metric, approximator, A, B, results_folder)
            result[A][B] = W
        
            est_time = (time.time() - start_time)
            avg_time = est_time if avg_time is None else avg_time*0.8 + 0.2*est_time
            N = N - 1
            
            print "relation",A,"->",B,":",W," est. time:", avg_time*N

    return result
