'''
Created on Mar 11, 2016

@author: iaroslav

methods to train ffnn 
'''

from feedforward import ffnn
import numpy as np
import tensorflow as tf

def train_evaluate(x,y, measure, params):
    # train and evaluate 2 layer nn
    
    sess = tf.InteractiveSession()
    
    X, Xvt = np.array_split(x, 2)
    Y, Yvt = np.array_split(y, 2)
    
    Xv, Xt = np.array_split(Xvt, 2)
    Yv, Yt = np.array_split(Yvt, 2)  
    
    # normalize
    xm = np.mean(X, axis = 0)
    ym = np.mean(Y, axis = 0)
    
    X = X - xm;
    Y = Y - ym;
    
    Xv = Xv - xm;
    Yv = Yv - ym;
    
    Xt = Xt - xm;
    Yt = Yt - ym;
    
    xd = np.std(X, axis = 0)
    yd = np.std(Y, axis = 0)
    
    X = X / xd;
    Y = Y / yd;
    
    Xv = Xv / xd;
    Yv = Yv / yd;
    
    Xt = Xt / xd;
    Yt = Yt / yd;        
    
    ipt = tf.placeholder("float", shape=[None, X.shape[-1]])

    ff = ffnn(ipt)
    
    for l in range(params[1]):
        ff.add_dense(params[0])
        ff.add_activation("relu")
        
    ff.add_dense(Y.shape[-1])
    
    otp = ff.get_output()

    gtr = tf.placeholder("float", shape=[None, Y.shape[-1]])
        
    loss = tf.reduce_mean( ((otp - gtr) ** 2) )
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    
    sess.run(tf.initialize_all_variables())
    
    #saver = tf.train.Saver()
    # stopping criterion: improvement on validation set is no more 
    
    finish_checks = 30
    #best_model_name = "best_model.temp"
    
    finished = finish_checks
    best_acc = 0.0
    test_acc = 0.0
    
    while finished > 0:
        inputs = {ipt: X, gtr: Y}
        train_step.run(feed_dict=inputs)
        
        inputs_val = {ipt: Xv, gtr: Yv}
        Ypr = otp.eval(feed_dict = inputs_val);
        local_acc = measure(Y, Yv, Ypr)
        
        if local_acc > best_acc:
            finished = finish_checks
            best_acc = local_acc
            
            inputs_tst = {ipt: Xt, gtr: Yt}
            Ypr = otp.eval(feed_dict = inputs_tst);
            test_acc = measure(Y, Yt, Ypr)
            
            #saver.save(sess, best_model_name)
        else:
            finished = finished - 1
    
    # load the best nn model
    """saver.restore(sess, best_model_name)
    
    inputs_tst = {ipt: Xt, gtr: Yt}
    Ypr = otp.eval(feed_dict = inputs_tst);
    test_acc = measure(Y, Yt, Ypr)"""
    
    sess.close()
    
    return best_acc, test_acc
    
    
    
    