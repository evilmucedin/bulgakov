import theano
import theano.tensor as T

from utilities.initializations import get

__author__ = 'uyaseen'


class Lstm(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim, init='uniform',
                 inner_init='orthonormal', inner_activation=T.nnet.hard_sigmoid,
                 activation=T.tanh, mini_batch=False, params=None):
        self.inner_activation = inner_activation
        self.activation = activation
        self.mini_batch = mini_batch
        if mini_batch:
            input = input.dimshuffle(1, 0, 2)
        if params is None:
            # input gate
            self.W_i = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                     name='W_i',
                                     borrow=True)
            self.U_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_i',
                                     borrow=True)
            self.b_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_i',
                                     borrow=True)
            # forget gate
            self.W_f = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                     name='W_f',
                                     borrow=True)
            self.U_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_f',
                                     borrow=True)
            self.b_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                     name='b_f',
                                     borrow=True)
            # memory
            self.W_c = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                     name='W_c',
                                     borrow=True)
            self.U_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_c',
                                     borrow=True)
            self.b_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_c',
                                     borrow=True)
            # output gate
            self.W_o = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                     name='W_o',
                                     borrow=True)
            self.U_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_o',
                                     borrow=True)
            self.b_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_o',
                                     borrow=True)
            # weights pertaining to output neuron
            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)),
                                   name='V',
                                   borrow=True)
            self.b_y = theano.shared(value=get(identifier='zero', shape=(output_dim,)),
                                     name='b_y',
                                     borrow=True)

        else:
            self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, self.b_f, \
                self.W_c, self.U_c, self.b_c, self.W_o, self.U_o, self.b_o, self.V, self.b_y = params

        self.c0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='c0', borrow=True)
        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.b_c,
                       self.W_o, self.U_o, self.b_o,
                       self.V, self.b_y]

        if mini_batch:
            def recurrence(x_t, c_tm_prev, h_tm_prev):
                x_i = T.dot(x_t, self.W_i) + self.b_i
                x_f = T.dot(x_t, self.W_f) + self.b_f
                x_c = T.dot(x_t, self.W_c) + self.b_c
                x_o = T.dot(x_t, self.W_o) + self.b_o

                i_t = inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
                f_t = inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
                c_t = f_t * c_tm_prev + i_t * activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
                o_t = inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
                h_t = o_t * activation(c_t)  # actual hidden state

                y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b_y)

                return c_t, h_t, y_t

            [_, self.h_t, self.y_t], _ = theano.scan(
                recurrence,
                sequences=input,
                outputs_info=[T.alloc(self.c0, input.shape[1], hidden_dim),
                              T.alloc(self.h0, input.shape[1], hidden_dim),
                              None]
            )
            self.h_t = self.h_t.dimshuffle(1, 0, 2)
            self.y_t = self.y_t.dimshuffle(1, 0, 2)
            self.y = T.argmax(self.y_t, axis=2)
        else:
            def recurrence(x_t, c_tm_prev, h_tm_prev):
                x_i = T.dot(x_t, self.W_i) + self.b_i
                x_f = T.dot(x_t, self.W_f) + self.b_f
                x_c = T.dot(x_t, self.W_c) + self.b_c
                x_o = T.dot(x_t, self.W_o) + self.b_o

                i_t = inner_activation(x_i + T.dot(h_tm_prev, self.U_i))
                f_t = inner_activation(x_f + T.dot(h_tm_prev, self.U_f))
                c_t = f_t * c_tm_prev + i_t * activation(x_c + T.dot(h_tm_prev, self.U_c))  # internal memory
                o_t = inner_activation(x_o + T.dot(h_tm_prev, self.U_o))
                h_t = o_t * activation(c_t)  # actual hidden state

                y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b_y)

                return c_t, h_t, y_t[0]

            [_, self.h_t, self.y_t], _ = theano.scan(
                recurrence,
                sequences=input,
                outputs_info=[self.c0, self.h0, None]
            )
            self.y = T.argmax(self.y_t, axis=1)

    def cross_entropy(self, y):
        if self.mini_batch:
            return T.mean(T.sum(T.nnet.categorical_crossentropy(self.y_t, y), axis=1))  # naive batch-normalization
        else:
            return T.sum(T.nnet.categorical_crossentropy(self.y_t, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_t)[:, y])

    def errors(self, y):
        return T.mean(T.neq(self.y, y))

    # TODO: Find a better way of sampling.
    def generative_sampling(self, seed, emb_data, sample_length):
        fruit = theano.shared(value=seed)

        def step(c_tm, h_tm, y_tm):

            x_i = T.dot(emb_data[y_tm], self.W_i) + self.b_i
            x_f = T.dot(emb_data[y_tm], self.W_f) + self.b_f
            x_c = T.dot(emb_data[y_tm], self.W_c) + self.b_c
            x_o = T.dot(emb_data[y_tm], self.W_o) + self.b_o

            i_t = self.inner_activation(x_i + T.dot(h_tm, self.U_i))
            f_t = self.inner_activation(x_f + T.dot(h_tm, self.U_f))
            c_t = f_t * c_tm + i_t * self.activation(x_c + T.dot(h_tm, self.U_c))  # internal memory
            o_t = self.inner_activation(x_o + T.dot(h_tm, self.U_o))
            h_t = o_t * self.activation(c_t)  # actual hidden state

            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b_y)
            y = T.argmax(y_t, axis=1)

            return c_t, h_t, y[0]

        [_, _, samples], _ = theano.scan(fn=step,
                                         outputs_info=[self.c0, self.h0, fruit],
                                         n_steps=sample_length)

        get_samples = theano.function(inputs=[],
                                      outputs=samples)

        return get_samples()


class BiLstm(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim,
                 mini_batch=False, params=None):
        self.mini_batch = mini_batch
        input_f = input
        if mini_batch:
            input_b = input[::, ::-1]
        else:
            input_b = input[::-1]
        if params is None:
            self.fwd_lstm = Lstm(input=input_f, input_dim=input_dim, hidden_dim=hidden_dim,
                                 output_dim=output_dim, mini_batch=mini_batch)
            self.bwd_lstm = Lstm(input=input_b, input_dim=input_dim, hidden_dim=hidden_dim,
                                 output_dim=output_dim, mini_batch=mini_batch)
            self.V_f = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_f',
                borrow=True
            )
            self.V_b = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_b',
                borrow=True
            )
            self.by = theano.shared(
                value=get('zero', shape=(output_dim,)),
                name='by',
                borrow=True)

        else:
            # To support loading from persistent storage, the current implementation of Lstm() will require a
            # change and is therefore not supported.
            # An elegant way would be to implement BiLstm() without using Lstm() [is a trivial thing to do].
            raise NotImplementedError

        # since now bilstm is doing the actual classification ; we don't need 'Lstm().V & Lstm().by' as they
        # are not part of computational graph (separate logistic-regression unit/layer is probably the best way to
        # handle this). Here's the ugly workaround -_-
        self.params = [self.fwd_lstm.W_i, self.fwd_lstm.U_i, self.fwd_lstm.b_i,
                       self.fwd_lstm.W_f, self.fwd_lstm.U_f, self.fwd_lstm.b_f,
                       self.fwd_lstm.W_c, self.fwd_lstm.U_c, self.fwd_lstm.b_c,
                       self.fwd_lstm.W_o, self.fwd_lstm.U_o, self.fwd_lstm.b_o,

                       self.bwd_lstm.W_i, self.bwd_lstm.U_i, self.bwd_lstm.b_i,
                       self.bwd_lstm.W_f, self.bwd_lstm.U_f, self.bwd_lstm.b_f,
                       self.bwd_lstm.W_c, self.bwd_lstm.U_c, self.bwd_lstm.b_c,
                       self.bwd_lstm.W_o, self.bwd_lstm.U_o, self.bwd_lstm.b_o,

                       self.V_f, self.V_b, self.by]

        self.bwd_lstm.h_t = self.bwd_lstm.h_t[::-1]
        # Take the weighted sum of forward & backward lstm's hidden representation
        self.h_t = T.dot(self.fwd_lstm.h_t, self.V_f) + T.dot(self.bwd_lstm.h_t, self.V_b)

        if mini_batch:
            # T.nnet.softmax cannot operate on tensor3, here's a simple reshape trick to make it work.
            h_t = self.h_t + self.by
            h_t_t = T.reshape(h_t, (h_t.shape[0] * h_t.shape[1], -1))
            y_t = T.nnet.softmax(h_t_t)
            self.y_t = T.reshape(y_t, h_t.shape)
            self.y = T.argmax(self.y_t, axis=2)
        else:
            self.y_t = T.nnet.softmax(self.h_t + self.by)
            self.y = T.argmax(self.y_t, axis=1)

    def cross_entropy(self, y):
        if self.mini_batch:
            return T.mean(T.sum(T.nnet.categorical_crossentropy(self.y_t, y), axis=1))  # naive batch-normalization
        else:
            return T.sum(T.nnet.categorical_crossentropy(self.y_t, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_t)[:, y])

    def errors(self, y):
        return T.mean(T.neq(self.y, y))

    # TODO: Find a way of sampling (running forward + backward lstm manually is really ugly and therefore, avoided).
    def generative_sampling(self, seed, emb_data, sample_length):
        return NotImplementedError
