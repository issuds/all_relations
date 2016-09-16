'''
This does fittingo of the
'''

import numpy as np

# names
ANN_approximator = "ANN"
SVR_approximator = "SVR"
AdaBoost_approximator = "AdaBoost"
KNN_approximator = "KNN"

def split_matrix(X, tr = 0.5, vl = 0.25):
    """ splits data into training, validation and testing parts
    """
    X, Xv, Xt = X[:int(len(X)*tr)], X[int(len(X)*tr):int(len(X)*(tr + vl))], X[int(len(X)*(tr + vl)):]
    return X, Xv, Xt

def prepare_data(x, y):
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

    from feedforward import ffnn
    import tensorflow as tf

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



def fit_report_sklearn(params, apx):


    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor

    accepted_sklearn_classes = {}
    accepted_sklearn_classes[SVR_approximator] = SVR
    accepted_sklearn_classes[AdaBoost_approximator] = AdaBoostRegressor
    accepted_sklearn_classes[KNN_approximator] = KNeighborsRegressor


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
    # train and evaluate 2 layer nn
    approximator = params['class']

    if approximator == ANN_approximator:
        return fit_report_ANN(params)
    else:
        return fit_report_sklearn(params, approximator)
    
    
    
    