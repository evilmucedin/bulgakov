#!/usr/bin/env python3

import argparse

from random import randint, shuffle
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

IDS = [
    "1001",
    "12",
    "aelita",
    "beg",
    "garin",
    "gvardiya",
    "master",
    "nerv",
    "staruha",
    "telenok",
]

SEQ_LENGTH = 100
BATCH_SIZE = 100
REC_MODEL = 'gru'
N_H = 256
OPTIMIZER = 'rmsprop'
LEARNING_RATE = 0.0002


def build_model(dataset, vocabulary, m_path, batch_size, use_existing_model, rec_model, n_h, optimizer, learning_rate):
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)
    n_train_batches = int(train_set_x.get_value(
        borrow=True).shape[0] / batch_size) + 1

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

    rec_params = None
    if use_existing_model:
        if rec_model == 'birnn' or rec_model == 'bigru' or rec_model == 'bilstm':
            raise NotImplementedError(
                'Loading parameters for bidirectional models is not supported')
        else:
            if os.path.isfile(m_path):
                with open(m_path, 'rb') as f:
                    rec_params, lvocabulary, lvoc = pkl.load(f)
                    assert(lvocabulary == vocabulary)
            else:
                print(
                    'Unable to load existing model %s, initializing model with random weights' % m_path)

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
        raise NotImplementedError('Only supported options for recurrent models are:\n'
                                  'rnn, gru, lstm, birnn, bigru, bilstm')

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
    eval_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    return model, train_model, voc, eval_model, n_train_batches


def train(dataset, vocabulary, b_path, rec_model='gru',
          n_h=100, use_existing_model=False, optimizer='rmsprop',
          learning_rate=LEARNING_RATE, n_epochs=100, sample_length=SEQ_LENGTH,
          batch_size=30, id="none"):
    print('train(..)')
    m_path = b_path + rec_model + '-best_model_' + \
        str(batch_size) + "-" + id + '.pkl'
    model, train_model, voc, _, n_train_batches = build_model(
        dataset, vocabulary, m_path, batch_size, use_existing_model, rec_model, n_h, optimizer, learning_rate)
    vocab, ix_to_words, words_to_ix = vocabulary

    ###############
    # TRAIN MODEL #
    ###############
    print('model -- %s training' % rec_model)
    logging_freq = 5
    sampling_freq = 10  # sampling is computationally expensive, therefore, need to be adjusted
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    train_cost = 0.
    index = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        index += 1
        indices = list(range(n_train_batches))
        shuffle(indices)

        for i in indices:
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

        if epoch % logging_freq == 0:
            iter_end_time = timeit.default_timer()
            print('epoch: %i/%i, cost: %0.8f, sample: %.4fm' %
                  (epoch, n_epochs, train_cost / index, (iter_end_time - iter_start_time) / 60.))

            # save the current best model
            if train_cost < best_train_error:
                print("save new best model %f" % (best_train_error - train_cost))
                best_train_error = train_cost
                with open(m_path, 'wb') as f:
                    pkl.dump((model.params, vocabulary, voc),
                             f, pkl.HIGHEST_PROTOCOL)

            train_cost = 0.
            index = 0

        # sample from the model now and then
        if epoch % sampling_freq == 0:
            seed = randint(0, len(vocab) - 1)
            idxes = model.generative_sampling(
                seed, emb_data=voc, sample_length=2*sample_length)
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
    plt.savefig(b_path + rec_model + '-error-plot_ ' + id + "_" +
                str(batch_size) + '.png')
    # plt.show()
    plt.close()


def trainAll():
    modelIds = IDS[:]
    shuffle(modelIds)
    for id in modelIds:
        data, vocabulary = read_char_data(
            "data/" + id + ".txt", seq_length=SEQ_LENGTH)
        train(data, vocabulary, b_path='data/models/', rec_model=REC_MODEL,
              n_h=N_H, optimizer='rmsprop', use_existing_model=True,
              n_epochs=600, batch_size=BATCH_SIZE, id=id)
        print('... done')


def predict(model, txt):
    with open(model, 'rb') as f:
        _, vocabulary, _ = pkl.load(f)
    dataset, vocabulary = read_char_data(
        txt, seq_length=SEQ_LENGTH, vocabulary=vocabulary)
    model, _, _, eval_model, n_train_batches = build_model(
        dataset, vocabulary, model, BATCH_SIZE, True, REC_MODEL, N_H, OPTIMIZER, LEARNING_RATE)
    cost = 0.
    for i in range(n_train_batches):
        cost += eval_model(i)
    print('Cost: %f' % (cost / n_train_batches))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Character-level LSTM model for texts")
    parser.add_argument('-mode', type=str)
    parser.add_argument('-model', type=str, default="")
    parser.add_argument('-txt', type=str, default="")
    args = parser.parse_args()
    if args.mode == "predict":
        predict(args.model, args.txt)
    elif args.mode == "train":
        trainAll()
