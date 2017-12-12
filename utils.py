import pickle
import numpy as np
import pandas as pd
import argparse
import os
import datetime as dt
import time
data_path='/home/zoe/PycharmProjects/RecSys-master/data/'
save_dir='/home/zoe/PycharmProjects/RecSys-master/save/'
seq_length=1
class DataHelper(object):
    def __init__(self,args,data_path,save_dir):
        self.data_path = data_path
        self.save_dir = save_dir
        self.args = args
        self.res = None
        self.session_key = 'SessionId'
        self.item_key = 'ItemId'
        self.time_key = 'Time'
        self. reset_after_session = True
        self.train_random_order = False
        self.time_sort = True
        self.length=0

    # def __init__(self,data_path,save_dir):
    #     self.data_path = data_path
    #     self.save_dir = save_dir
    #     self.res = None
    #     self.session_key = 'SessionId'
    #     self.item_key = 'ItemId'
    #     self.time_key = 'Time'
    #     self. reset_after_session = True
    #     self.train_random_order = False
    #     self.time_sort = True
    #     self.batch_size=5



    def data_convert(self):
        if os.path.exists(self.save_dir + 'data.pkl'):
            with open(self.save_dir + 'data.pkl', 'rb') as f:
                data_ = pickle.load(f)
        else:
            data_ = pd.read_csv(self.data_path + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0, 1, 2],
                                dtype={0: np.int32, 1: str, 2: np.int64})
            data_.columns = ['SessionId', 'TimeStr', 'ItemId']
            data_['Time'] = data_.TimeStr.apply(lambda x: time.mktime(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timetuple()))
            del (data_['TimeStr'])
            with open(self.save_dir + 'data.pkl', 'wb') as f:
                pickle.dump(data_, f, pickle.HIGHEST_PROTOCOL)
        return data_

    def choose_data(self):
        data=self.data_convert()[0:1000]
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
        item_supports = data.groupby('ItemId').size()
        data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax - 86400].index
        session_test = session_max_times[session_max_times >= tmax - 86400].index
        train = data[np.in1d(data.SessionId, session_train)]
        test = data[np.in1d(data.SessionId, session_test)]
        test = test[np.in1d(test.ItemId, train.ItemId)]
        tslength = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
        print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                                 train.ItemId.nunique()))
        train.to_csv(self.save_dir + 'rsc15_train_full.txt', sep='\t', index=False)
        print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                           test.ItemId.nunique()))
        test.to_csv(self.save_dir + 'rsc15_test.txt', sep='\t', index=False)

        tmax = train.Time.max()
        session_max_times = train.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax - 86400].index
        session_valid = session_max_times[session_max_times >= tmax - 86400].index
        train_tr = train[np.in1d(train.SessionId, session_train)]
        valid = train[np.in1d(train.SessionId, session_valid)]
        valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
        tslength = valid.groupby('SessionId').size()
        valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
        print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                            train_tr.ItemId.nunique()))
        train_tr.to_csv(self.save_dir + 'rsc15_train_tr.txt', sep='\t', index=False)
        print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                                 valid.ItemId.nunique()))
        valid.to_csv(self.save_dir + 'rsc15_train_valid.txt', sep='\t', index=False)


    # def get_input_train(self,retrain=False):
    #
    #     data = pd.read_csv(self.save_dir+"rsc15_train_tr.txt", sep='\t', dtype={'ItemId': np.int64})
    #     self.predict = None
    #     self.error_during_train = False
    #     itemids = data[self.item_key].unique() #all unique item ids
    #
    #     if not retrain:
    #         self.n_items = len(itemids)  # the lengtg if akk unique iten uds
    #         self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)  # g
    #         data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
    #                         on=self.item_key, how='inner')
    #         self.length = data.ItemIdx.values.max() + 1
        #     data.sort_values([self.session_key, self.time_key], inplace=True)
        #     offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        #     offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        # else:
        #     new_item_mask = ~np.in1d(itemids, self.itemidmap.index)
        #     n_new_items = new_item_mask.sum()
        #     if n_new_items:
        #         self.itemidmap = self.itemidmap.append(
        #             pd.Series(index=itemids[new_item_mask], data=np.arange(n_new_items) + len(self.itemidmap)))
        #         for W in [self.E if self.embedding else self.Wx[0], self.Wy]:
        #             self.extend_weights(W, n_new_items)
        #         self.By.set_value(
        #             np.vstack([self.By.get_value(), np.zeros((n_new_items, 1))]))
        #         self.n_items += n_new_items
        #         print('Added {} new items. Number of items is {}.'.format(n_new_items, self.n_items))
        #     data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
        #                     on=self.item_key, how='inner')
        #     data.sort_values([self.session_key, self.time_key], inplace=True)
        #     offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        #     offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        #
        # self.base_order = np.argsort(
        #     data.groupby(self.session_key)[self.time_key].min().values) if self.time_sort else np.arange(
        #     len(offset_sessions) - 1)
        # data_items = data.ItemIdx.values

        # return self.length
    #
    #
    # # def get_seq(self):
    #
    # #

    def get_length(self):
        data1 = pd.read_csv(self.save_dir+"rsc15_train_tr.txt", sep='\t', dtype={'ItemId': np.int64})
        itemids = data1[self.item_key].unique() #all unique item ids
        length=len(itemids)

        return length

    def get_input_test(self,retrain=False):

        data = pd.read_csv(self.save_dir+"rsc15_test.txt", sep='\t', dtype={'ItemId': np.int64})
        self.predict = None
        self.error_during_train = False
        itemids = data[self.item_key].unique() #all unique item ids

        self.n_items = len(itemids)  # the lengtg if akk unique iten uds
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)  # g
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()

        input=[]
        output=[]
        base_order = np.argsort(
            data.groupby(self.session_key)[self.time_key].min().values) if self.time_sort else np.arange(
            len(offset_sessions) - 1)
        data_items = data.ItemIdx.values

        session_idx_arr = np.random.permutation(len(offset_sessions)) if self.train_random_order else base_order
        # print session_idx_arr
        iters = np.arange(self.args.batch_size)
        # print iters
        maxiter = iters.max()
        start = offset_sessions[session_idx_arr[iters]]
        end = offset_sessions[session_idx_arr[iters]+1]
        finished = False
        while not finished:
            minlen = (end - start).min()
            out_idx = data_items[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = data_items[start + i + 1]
                input.append(in_idx)
                output.append(out_idx)
            start = start + minlen - 1
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(offset_sessions) - 1:
                    finished = True
                    break
                iters[idx] = maxiter
                start[idx] = offset_sessions[session_idx_arr[maxiter]]
                end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]

            # one_hot = np.zeros(shape=[self.args.batch_size, data.ItemIdx.values.max()+1])
            # for i in range(self.args.batch_size):
            #     one_hot[i][in_idx[i]]=1

            # input.append(one_hot)
            # output.append(out_idx)
        # print input
        # print np.shape(input)
        # print np.shape(output)
        # print len(input)

        return input,output
#
# if __name__ == '__main__':
#     DH=DataHelper(data_path,save_dir)
#     DH.get_input_train()