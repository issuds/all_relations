
import tensorflow as tf

class ffnn():
    
    def __init__(self, ipt):
        self.ipt = ipt
        self.output_size = ipt.get_shape()[1].value;
        self.output = self.ipt
    
    def add_dense(self, size):
        W = tf.Variable( tf.truncated_normal([ self.output_size, size], stddev = 0.05) )
        b = tf.Variable( tf.truncated_normal([ size], stddev = 0.05) )
        
        self.output = tf.matmul(self.output,W) + b
        self.output_size = self.output.get_shape()[1].value;
    
    def add_activation(self, activation):
        if activation == "relu":
            self.output = tf.maximum(self.output, 0.0)
    
    def get_output(self):
        return self.output
        
       