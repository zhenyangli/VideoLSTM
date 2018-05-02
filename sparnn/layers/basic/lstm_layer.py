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
simplified LSTM (Cell is not connected with any gate, 
                 i.e. input gate, output gate, or forget gate)
'''
class LSTMLayer(Layer):
    def __init__(self, layer_param):
        super(LSTMLayer, self).__init__(layer_param)
        assert 3 == self.input.ndim
        #assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)
        
        self.gate_activation = layer_param.get('gate_activation', 'sigmoid')
        self.modular_activation = layer_param.get('modular_activation', 'tanh')
        self.hidden_activation = layer_param.get('hidden_activation', 'tanh')

        self.init_hidden_state = layer_param.get("init_hidden_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_cell_state = layer_param.get("init_cell_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_hidden_state = TT.unbroadcast(self.init_hidden_state, *range(self.init_hidden_state.ndim))
        self.init_cell_state = TT.unbroadcast(self.init_cell_state, *range(self.init_cell_state.ndim))
        if 'n_steps' in layer_param:
            self.n_steps = layer_param['n_steps']
        else:
            self.n_steps = layer_param.get('n_steps', self.input.shape[0])
        self.input_mat_size = (self.feature_in, self.feature_out)
        self.transition_mat_size = (self.feature_out, self.feature_out)

        # input to LSTM
        self.W_xi = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xi"), scale=0.1)
        self.W_xf = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xf"), scale=0.1)
        self.W_xo = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xo"), scale=0.1)
        self.W_xc = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xc"), scale=0.1)

        # LSTM to LSTM
        self.W_hi = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hi"), scale=0.1)
        self.W_hf = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hf"), scale=0.1)
        self.W_ho = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_ho"), scale=0.1)
        self.W_hc = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hc"), scale=0.1)

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

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "LSTMLayer-" + str(self.id)

    def step_fprop(self, x_t, h_tm1, c_tm1, *args):
        input_gate = quick_activation(TT.dot(x_t, self.W_xi)
                                      + TT.dot(h_tm1, self.W_hi)
                                      + self.b_i, "sigmoid")
        forget_gate = quick_activation(TT.dot(x_t, self.W_xf)
                                       + TT.dot(h_tm1, self.W_hf)
                                       + self.b_f, "sigmoid")
        c_t = forget_gate * c_tm1 \
              + input_gate * quick_activation(TT.dot(x_t, self.W_xc)
                                              + TT.dot(h_tm1, self.W_hc)
                                              + self.b_c, "tanh")
        output_gate = quick_activation(TT.dot(x_t, self.W_xo)
                                       + TT.dot(h_tm1, self.W_ho)
                                       + self.b_o, "sigmoid")
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
        # We need to pad it to (Timestep, Minibatch, FeatureDim)
        # and keep the last one added dimensions broadcastable. TT.shape_padright
        # function is thus a good choice

        if self.mask is None:
            scan_input = [self.input]
            scan_fn = self.step_fprop
        else:
            scan_input = [self.input, TT.shape_padright(self.mask, 1)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        [self.output, self.cell_output], self.output_update = quick_unroll_scan(fn=scan_fn,
        #[self.output, self.cell_output], self.output_update = theano.scan(fn=scan_fn,
                                                                          outputs_info=[self.init_hidden_state,
                                                                                        self.init_cell_state],
                                                                          sequences=scan_input,
                                                                          non_sequences=non_seqs,
                                                                          n_steps=self.n_steps
                                                                          )