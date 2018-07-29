import random

import cPickle as pkl
import os.path

import theano.tensor as T

from model.rnn import Rnn
from model.gru import Gru
from model.lstm import Lstm

from utilities.loaddata import load_data
from utilities.textreader import read_word_data, read_char_data

__author__ = 'uyaseen'


def sample(dataset, vocabulary, m_path, n_h, rec_model, sample_count, sample_length):
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_y = len(vocab)  # dimension of output classes
    if rec_model == 'birnn' or rec_model == 'bigru' or rec_model == 'bilstm':
        print('Loading parameters for bidirectional models is not supported')
        raise NotImplementedError
    else:
        if os.path.isfile(m_path):
            with open(m_path, 'r') as f:
                rec_params = pkl.load(f)
        else:
            print('Unable to load model: %s, please make sure model path is correct.' % m_path)
            return IOError

    x = T.fmatrix('x')  # for sampling, type of 'x' does not matter
    if rec_model == 'rnn':
        model = Rnn(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                    params=rec_params)
    elif rec_model == 'gru':
        model = Gru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                    params=rec_params)
    elif rec_model == 'lstm':
        model = Lstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                     params=rec_params)
    else:
        print('Sampling is only supported for:\n'
              'rnn, gru, lstm')
        raise NotImplementedError

    random.seed(36)
    for i in xrange(0, sample_count):
        seed = random.randint(0, len(vocab)-1)
        idxes = model.generative_sampling(seed, emb_data=voc, sample_length=sample_length)
        sample = ''.join(ix_to_words[ix] for ix in idxes)
        print(str(i+1) + '\n' + sample)


if __name__ == '__main__':
    data, vocabulary = read_char_data('data/input.txt', seq_length=50)
    sample(data, vocabulary, m_path='data/models/gru-best_model.pkl', n_h=100,
           rec_model='gru', sample_count=10, sample_length=200)
