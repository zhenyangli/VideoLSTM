__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)


# TODO [IMPORTANT]The current ConvRNNLayer hasn't been tested. We should test it in the future version when Hessian Free Optimization is added!

class ConvRNNLayer(Layer):
    def __init__(self, layer_param):
        super(ConvRNNLayer, self).__init__(layer_param)
        assert self.input.ndim == 5
        self.input_receptive_field = layer_param['input_receptive_field']
        self.transition_receptive_field = layer_param['transition_receptive_field']
        self.activation = layer_param['activation']
        self.init_hidden_state = layer_param.get("init_hidden_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.n_steps = layer_param.get('n_steps', self.input.shape[0])

        self.kernel_size = (self.feature_out, self.feature_in,
                            self.input_receptive_field[0], self.input_receptive_field[1])
        self.transition_mat_size = (self.feature_out, self.feature_out,
                                    self.transition_receptive_field[0], self.transition_receptive_field[1])

        self.W_hh = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hh"))
        self.W_xh = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xh"))
        self.b_h = quick_zero((self.feature_out,), self._s("b_h"))

        self.param = [self.W_hh, self.W_xh, self.b_h]
        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "ConvRNNLayer-" + str(self.id)

    def step_fprop(self, x_t, m_t, h_tm1):
        h_t = conv2d_same(x_t, self.W_xh, (None,) + self.dim_in, self.kernel_size) \
              + conv2d_same(h_tm1, self.W_hh, (None,) + self.dim_out, self.transition_mat_size) \
              + self.b_h.dimshuffle('x', 0, 'x', 'x')
        h_t = quick_activation(h_t, self.activation)
        if m_t is not None:
            h_t = m_t * h_t + (1 - m_t) * h_tm1
        return h_t

    def init_states(self):
        return self.init_hidden_state,

    def fprop(self):
        # Here the masking strategy is similar to ConvLSTMLayer, check the fprop() function in conv_lstm_layer.py
        if self.mask is None:
            scan_input = [self.input]
            scan_fn = lambda x_t, h_tm1: self.step_fprop(x_t, None, h_tm1)
        else:
            scan_input = [self.input, TT.shape_padright(self.mask, 3)]
            scan_fn = lambda x_t, mask_t, h_tm1: self.step_fprop(x_t, mask_t, h_tm1)

        self.output, self.output_update = quick_scan(fn=scan_fn,
                                                     outputs_info=[self.init_hidden_state],
                                                     sequences=scan_input,
                                                     name=self._s("recurrent_output_func"),
                                                     n_steps=self.input.shape[0]
                                                     )
