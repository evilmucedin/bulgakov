import theano
import theano.tensor as T

from utilities.initializations import get


__author__ = 'uyaseen'


class Rnn(object):

    def __init__(self, input, input_dim, hidden_dim, output_dim,
                 activation=T.tanh, init='uniform', inner_init='orthonormal',
                 mini_batch=False, params=None):
        self.activation = activation
        self.mini_batch = mini_batch
        if mini_batch:
            input = input.dimshuffle(1, 0, 2)
        if params is None:
            self.W = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim)),
                                   name='W',
                                   borrow=True
                                   )
            self.U = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                   name='U',
                                   borrow=True
                                   )
            self.V = theano.shared(value=get(identifier=init, shape=(hidden_dim, output_dim)),
                                   name='V',
                                   borrow=True
                                   )
            self.bh = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                    name='bh',
                                    borrow=True)
            self.by = theano.shared(value=get(identifier='zero', shape=(output_dim, )),
                                    name='by',
                                    borrow=True)
        else:
            self.W, self.U, self.V, self.bh, self.by = params

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.params = [self.W, self.U, self.V, self.bh, self.by]

        if mini_batch:
            def recurrence(x_t, h_tm_prev):
                h_t = activation(T.dot(x_t, self.W) +
                                 T.dot(h_tm_prev, self.U) + self.bh)
                y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.by)
                return h_t, y_t

            [self.h_t, self.y_t], _ = theano.scan(
                recurrence,
                sequences=input,
                outputs_info=[T.alloc(self.h0, input.shape[1], hidden_dim), None]
            )
            self.h_t = self.h_t.dimshuffle(1, 0, 2)
            self.y_t = self.y_t.dimshuffle(1, 0, 2)
            self.y = T.argmax(self.y_t, axis=2)
        else:
            def recurrence(x_t, h_tm_prev):
                h_t = activation(T.dot(x_t, self.W) +
                                 T.dot(h_tm_prev, self.U) + self.bh)
                y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.by)
                return h_t, y_t[0]

            [self.h_t, self.y_t], _ = theano.scan(
                recurrence,
                sequences=input,
                outputs_info=[self.h0, None]
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

    '''
    This method is required because recurrence is achieve by 'scan' in theano; and generative sampling requires
    prediction at each time-step to be the input for the next time steps; and using scan for achieving this doesn't
    makes sense as each prediction will require a scan to run only once and in each of those runs, it will use the
    default value of '0' of previous (hidden) time-steps (I am referring to 'h0' in Rnn) which means that on
    getting each sample we are throwing away the context information from previous time-steps (which will be very wrong).
    * seed: initial seed to start sampling from
    * emb_data: required fetch the original data 'row'
    TODO: Find a better way of sampling.
    '''
    def generative_sampling(self, seed, emb_data, sample_length):
        fruit = theano.shared(value=seed)

        def step(h_tm, y_tm):
            h_t = self.activation(T.dot(emb_data[y_tm], self.W) +
                                  T.dot(h_tm, self.U) + self.bh)
            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.by)
            y = T.argmax(y_t, axis=1)

            return h_t, y[0]

        [_, samples], _ = theano.scan(fn=step,
                                      outputs_info=[self.h0, fruit],
                                      n_steps=sample_length)

        get_samples = theano.function(inputs=[],
                                      outputs=samples)

        return get_samples()


class BiRnn(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim,
                 mini_batch=False, params=None):
        self.mini_batch = mini_batch
        input_f = input
        if mini_batch:
            input_b = input[::, ::-1]
        else:
            input_b = input[::-1]
        if params is None:
            self.fwd_rnn = Rnn(input=input_f, input_dim=input_dim, hidden_dim=hidden_dim,
                               output_dim=output_dim, mini_batch=mini_batch)
            self.bwd_rnn = Rnn(input=input_b, input_dim=input_dim, hidden_dim=hidden_dim,
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
            # To support loading from persistent storage, the current implementation of Rnn() will require a
            # change and is therefore not supported.
            # An elegant way would be to implement BiRnn() without using Rnn() [is a trivial thing to do].
            raise NotImplementedError

        # since now birnn is doing the actual classification ; we don't need 'Rnn().V & Rnn().by' as they
        # are not part of computational graph (separate logistic-regression unit/layer is probably the best way to
        # handle this). Here's the ugly workaround -_-
        self.params = [self.fwd_rnn.W, self.fwd_rnn.U, self.fwd_rnn.bh,
                       self.bwd_rnn.W, self.bwd_rnn.U, self.bwd_rnn.bh,
                       self.V_f, self.V_b, self.by]

        self.bwd_rnn.h_t = self.bwd_rnn.h_t[::-1]
        # Take the weighted sum of forward & backward rnn's hidden representation
        self.h_t = T.dot(self.fwd_rnn.h_t, self.V_f) + T.dot(self.bwd_rnn.h_t, self.V_b)

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

    # TODO: Find a way of sampling (running forward + backward rnn manually is really ugly and therefore, avoided).
    def generative_sampling(self, seed, emb_data, sample_length):
        return NotImplementedError
