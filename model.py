import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np
class build_model(object):
    def __init__(self,args):
        self.args=args
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError
        self.b_model(args)
        ########################ACTIVATION FUNCTIONS#########################

    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

        ############################LOSS FUNCTIONS######################

    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def b_model(self,args):
        self.X = tf.placeholder(tf.int32, [self.args.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.args.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.args.batch_size, self.args.rnn_size], name='rnn_state') for _ in
                      xrange(self.args.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = self.args.sigma if self.args.sigma != 0 else np.sqrt(6.0 / (self.args.n_items + self.args.rnn_size))
            if self.args.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.args.n_items, self.args.rnn_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.args.n_items, self.args.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.args.n_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(self.args.rnn_size, activation=self.args.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.args.dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.args.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            print inputs
            print "**************8888"
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        if self.args.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            print 1
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)

            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)


            # logits_exploit = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            #
            # with tf.variable_scope('bandit'):
            #     epsilon = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
            #     logits_explore = tf.random_uniform([self.batch_size,self.batch_size], minval=0, maxval=1,
            #                                             dtype=tf.float32)
            #     logits_explore_norm = tf.nn.l2_normalize(logits_explore, dim=0)
            #     logits = tf.cond(epsilon > tf.constant(0.5), lambda: logits_exploit,
            #                           lambda: logits_explore_norm)
            #     self.yhat = self.final_activation(logits)
            #     self.cost = self.loss_function(self.yhat)

        else:

            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat =self.args.final_activation(logits)

            # logits_exploit = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            #
            # with tf.variable_scope('bandit'):
            #     epsilon = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
            #     logits_explore = tf.random_uniform([self.batch_size, self.n_items], minval=0, maxval=1,
            #                                             dtype=tf.float32)
            #     logits_explore_norm = tf.nn.l2_normalize(logits_explore, dim=0)
            #     logits = tf.cond(epsilon > tf.constant(0.5), lambda: logits_exploit,
            #                           lambda: logits_explore_norm)
            #     self.yhat = self.final_activation(logits)

        if not self.args.is_training:
            return

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.args.learning_rate, self.args.global_step, self.args.decay_steps,
                                                              self.args.decay, staircase=True))

        # with tf.name_scope("optimize"):
        #     self.depth=self.batch_size
        #     self.targets_onehot=tf.one_hot(self.Y,self.depth)
        #     self.Y_ = tf.cast(self.targets_onehot, tf.float32)
        #     b=tf.multiply(logits,self.Y_)
        #     a=tf.ones([self.batch_size,self.batch_size],dtype = tf.float32)+logits-tf.constant(2.0)*b
        #     compare=tf.maximum(a,0)
        #     self.prediction_cost =tf.reduce_sum(compare,axis=1)
        #     self.cross_entropy=tf.reduce_mean(self.prediction_cost)
        #     self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cross_entropy)  # Adam Optimizer


        '''
        Try different optimizers.
        '''
        # optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.lr)
        # optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.args.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.args.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        tf.summary.scalar("loss", self.cost)
        self.summary_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter('/home/zoe/PycharmProjects/RecSys-master/checkpoint',
                                            tf.get_default_graph())