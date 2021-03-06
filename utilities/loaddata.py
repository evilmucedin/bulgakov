import numpy as np
import theano

from utilities.textreader import get_one_hot, get_one_hot_vocab_list

__author__ = 'evilmucedin'

BORROW = True


def load_data(data, vocab, vocab_encoded, one_hot=True):
    print('load_data(..)')
    train_set = data
    print('[Train] # of rows: %i' % (len(train_set[0])))

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        vocab_one_hot = []
        # get one-hot representation for every word
        if one_hot:
            t_x = []
            for i in data_x:
                t_x.append([get_one_hot(x, len(vocab)) for x in i])
            data_x = t_x
            t_y = []
            for j in data_y:
                t_y.append([get_one_hot(y, len(vocab)) for y in j])
            data_y = t_y
            vocab_one_hot = get_one_hot_vocab_list(vocab_encoded, len(vocab))

        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX), borrow=BORROW)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX), borrow=BORROW)
        shared_v = theano.shared(np.asarray(vocab_one_hot,
                                            dtype=theano.config.floatX), borrow=BORROW)

        return shared_x, shared_y, shared_v

    print('... transferring data to the %s' % theano.config.device)
    train_set_x, train_set_y, voc = shared_dataset(train_set)
    return train_set_x, train_set_y, voc


def unload_data(train_set_x, train_set_y, voc):
    train_set_x.set_value([[[]]])
    train_set_y.set_value([[[]]])
    voc.set_value([[]])
