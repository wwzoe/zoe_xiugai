# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from utils import DataHelper
import train
import evaluation

PATH_TO_TRAIN = './save/rsc15_train_full.txt'
PATH_TO_TEST = './save/rsc15_test.txt'


def main():

    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--is_training', default=1,type=int)
    parser.add_argument('--batch_size', default=5,type=int)
    parser.add_argument('--dropout_p_hidden', default=1,type=int)
    parser.add_argument('--decay', default=0.96,type=float)
    parser.add_argument('--decay_steps', default=1e4,type=float)
    parser.add_argument('--sigma', default=0,type=int)
    parser.add_argument('--session_key', default="SessionId", type=str)
    parser.add_argument('--item_key', default="ItemId", type=str)
    parser.add_argument('--time_key', default="Time", type=str)
    parser.add_argument('--grad_cap', default=0, type=int)
    parser.add_argument('--checkpoint_dir', default="./checkpoint", type=str)
    parser.add_argument('--reset_after_session', default=True, type=bool)
    parser.add_argument('--n_items', default=-1, type=int)
    parser.add_argument('--init_as_normal', default=False, type=bool)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--rnn_size', default=1000, type=int)
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--test_model', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--save_dir', default='./save/', type=str)
    parser.add_argument('--data_path', default='./data/', type=str)
    args = parser.parse_args()
    run(args)



def run(args):
    print 1

    data_loader = DataHelper(args, args.data_path, args.save_dir)
    data_loader.choose_data()

    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    args.n_items = len(data['ItemId'].unique())

    args.dropout_p_hidden = 1.0 if args.is_training == 0 else args.dropout

    print(args.n_epochs)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = train.GRU4Rec(sess, args)
        if args.is_training==1:
            gru.fit(data)

        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))


if __name__ == '__main__':
    main()