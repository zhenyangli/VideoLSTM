__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
from theano.gradient import grad_clip

from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)

'''
    (simplified) Convolutional LSTM.
    Cell is not connected with any gate, i.e. input gate, output gate, or forget gate.
'''
class ConvLSTMLayer(Layer):
    def __init__(self, layer_param):
        super(ConvLSTMLayer, self).__init__(layer_param)
        assert 5 == self.input.ndim
        #assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)

        self.input_receptive_field = layer_param['input_receptive_field']
        self.transition_receptive_field = layer_param['transition_receptive_field']

        self.gate_activation = layer_param.get('gate_activation', 'sigmoid')
        self.modular_activation = layer_param.get('modular_activation', 'tanh')
        self.hidden_activation = layer_param.get('hidden_activation', 'tanh')

        self.init_hidden_state = layer_param.get("init_hidden_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_cell_state = layer_param.get("init_cell_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_hidden_state = TT.unbroadcast(self.init_hidden_state, *range(self.init_hidden_state.ndim))
        self.init_cell_state = TT.unbroadcast(self.init_cell_state, *range(self.init_cell_state.ndim))
        self.learn_padding = layer_param.get('learn_padding', False)
        self.input_padding = layer_param.get('input_padding', None)
        if 'n_steps' in layer_param:
            self.n_steps = layer_param['n_steps']
        else:
            self.n_steps = layer_param.get('n_steps', self.input.shape[0])
        self.kernel_size = (self.feature_out, self.feature_in,
                            self.input_receptive_field[0], self.input_receptive_field[1])
        self.transition_mat_size = (self.feature_out, self.feature_out,
                                    self.transition_receptive_field[0], self.transition_receptive_field[1])

        # input to LSTM
        self.W_xi = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xi"))
        self.W_xf = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xf"))
        self.W_xo = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xo"))
        self.W_xc = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xc"))

        # LSTM to LSTM
        self.W_hi = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hi"))
        self.W_hf = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hf"))
        self.W_ho = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_ho"))
        self.W_hc = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hc"))
        if self.learn_padding:
            self.hidden_padding = quick_zero((self.feature_out, ), self._s("hidden_padding"))
        else:
            self.hidden_padding = None

        # bias to LSTM
        self.b_i = quick_zero((self.feature_out, ), self._s("b_i"))
        self.b_f = quick_zero((self.feature_out, ), self._s("b_f"))
        self.b_o = quick_zero((self.feature_out, ), self._s("b_o"))
        self.b_c = quick_zero((self.feature_out, ), self._s("b_c"))

        # collect all parameters
        self.param = [self.W_xi, self.W_hi, self.b_i,
                      self.W_xf, self.W_hf, self.b_f,
                      self.W_xo, self.W_ho, self.b_o,
                      self.W_xc, self.W_hc, self.b_c]
        if self.learn_padding:
            self.param.append(self.hidden_padding)

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "ConvLSTMLayer-" + str(self.id)

    def step_fprop(self, x_t, h_tm1, c_tm1, *args):
        input_gate = quick_activation(conv2d_same(x_t, self.W_xi, (None, ) + self.dim_in,
                                                  self.kernel_size, self.input_padding)
                                      + conv2d_same(h_tm1, self.W_hi, (None, ) + self.dim_out,
                                                    self.transition_mat_size, self.hidden_padding)
                                      #+ c_tm1 * self.W_ci.dimshuffle('x', 0, 'x', 'x')
                                      + self.b_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        forget_gate = quick_activation(conv2d_same(x_t, self.W_xf, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                       + conv2d_same(h_tm1, self.W_hf, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.hidden_padding)
                                       #+ c_tm1 * self.W_cf.dimshuffle('x', 0, 'x', 'x')
                                       + self.b_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        c_t = forget_gate * c_tm1 \
              + input_gate * quick_activation(conv2d_same(x_t, self.W_xc, (None, ) + self.dim_in,
                                                          self.kernel_size, self.input_padding)
                                              + conv2d_same(h_tm1, self.W_hc, (None, ) + self.dim_out,
                                                            self.transition_mat_size, self.hidden_padding)
                                              + self.b_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
        output_gate = quick_activation(conv2d_same(x_t, self.W_xo, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                       + conv2d_same(h_tm1, self.W_ho, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.hidden_padding)
                                       #+ c_t * self.W_co.dimshuffle('x', 0, 'x', 'x')
                                       + self.b_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        h_t = output_gate * quick_activation(c_t, "tanh")

        return [h_t, c_t]

    def step_masked_fprop(self, x_t, mask_t, h_tm1, c_tm1, *args):
        h_t, c_t = self.step_fprop(x_t, h_tm1, c_tm1, *args)

        h_t = TT.switch(mask_t, h_t, h_tm1)
        c_t = TT.switch(mask_t, c_t, c_tm1)

        return [h_t, c_t]

    def init_states(self):
        return self.init_hidden_state, self.init_cell_state

    def fprop(self):

        # The dimension of self.mask is (Timestep, Minibatch).
        # We need to pad it to (Timestep, Minibatch, FeatureDim, Row, Col)
        # and keep the last three added dimensions broadcastable. TT.shape_padright
        # function is thus a good choice

        if self.mask is None:
            scan_input = [self.input]
            scan_fn = self.step_fprop
        else:
            scan_input = [self.input, TT.shape_padright(self.mask, 3)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        [self.output, self.cell_output], self.output_update = quick_unroll_scan(fn=scan_fn,
        #[self.output, self.cell_output], self.output_update = quick_scan(fn=scan_fn,
                                                                          outputs_info=[self.init_hidden_state,
                                                                                        self.init_cell_state],
                                                                          sequences=scan_input,
                                                                          non_sequences=non_seqs,
                                                                          n_steps=self.n_steps
                                                                          )