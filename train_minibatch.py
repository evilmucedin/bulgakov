#!/usr/bin/env python3

from random import randint
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
import timeit
import os.path

import theano
import theano.tensor as T

from model.rnn import Rnn, BiRnn
from model.gru import Gru, BiGru
from model.lstm import Lstm, BiLstm

from utilities.optimizers import get_optimizer
from utilities.loaddata import load_data
from utilities.textreader import read_word_data, read_char_data

__author__ = 'evilmucedin'


def train(dataset, vocabulary, b_path, rec_model='gru',
          n_h=100, use_existing_model=False, optimizer='rmsprop',
          learning_rate=0.001, n_epochs=100, sample_length=200,
          batch_size=30):
    print('train(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)
    n_train_batches = int(train_set_x.get_value(
        borrow=True).shape[0] / batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.ftensor3('x')
    y = T.ftensor3('y')
    print('... building the model')
    # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_x = len(vocab)
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + rec_model + '-best_model_' + str(batch_size) + '.pkl'

    rec_params = None
    if use_existing_model:
        if rec_model == 'birnn' or rec_model == 'bigru' or rec_model == 'bilstm':
            print('Loading parameters for bidirectional models is not supported')
            raise NotImplementedError
        else:
            if os.path.isfile(m_path):
                with open(m_path, 'rb') as f:
                    rec_params = pkl.load(f)
            else:
                print(
                    'Unable to load existing model %s , initializing model with random weights' % m_path)

    if rec_model == 'rnn':
        model = Rnn(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                    params=rec_params, mini_batch=True)
    elif rec_model == 'gru':
        model = Gru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                    params=rec_params, mini_batch=True)
    elif rec_model == 'lstm':
        model = Lstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                     params=rec_params, mini_batch=True)
    elif rec_model == 'birnn':
        model = BiRnn(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                      params=rec_params, mini_batch=True)
    elif rec_model == 'bigru':
        model = BiGru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                      params=rec_params, mini_batch=True)
    elif rec_model == 'bilstm':
        model = BiLstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                       params=rec_params, mini_batch=True)
    else:
        print('Only supported options for recurrent models are:\n'
              'rnn, gru, lstm, birnn, bigru, bilstm')
        raise NotImplementedError

    cost = model.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, model.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('model -- %s' % rec_model)
    print('... training')
    logging_freq = batch_size / 10
    sampling_freq = 10  # sampling is computationally expensive, therefore, need to be adjusted
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in range(n_train_batches):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model
            if train_cost < best_train_error:
                best_train_error = train_cost
                with open(m_path, 'wb') as f:
                    pkl.dump(model.params, f, pkl.HIGHEST_PROTOCOL)

            if i % logging_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, minibatch: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_batches, train_cost / (i + 1),
                       (iter_end_time - iter_start_time) / 60.))

        # sample from the model now and then
        if epoch % sampling_freq == 0:
            seed = randint(0, len(vocab) - 1)
            idxes = model.generative_sampling(
                seed, emb_data=voc, sample_length=sample_length)
            sample = ''.join(ix_to_words[ix] for ix in idxes)
            print(sample)

        train_cost /= n_train_batches
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('%s [batch size: %i]' % (rec_model, batch_size))
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + rec_model + '-error-plot_ ' +
                str(batch_size) + '.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data, vocabulary = read_char_data(
        'data/Zolotoi Tielienok - Ievghienii Pietrovich Pietrov.txt', seq_length=100)
    train(data, vocabulary, b_path='data/models/', rec_model='gru',
          n_h=100, optimizer='rmsprop', use_existing_model=True,
          n_epochs=600, batch_size=1000)
    print('... done')
