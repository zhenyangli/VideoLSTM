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
    Conditional (simplified) Convolutional LSTM with Attention.
    Cell is not connected with any gate, i.e. input gate, output gate, or forget gate.
'''
class CondConvLSTMSigLayer(Layer):
    def __init__(self, layer_param):
        super(CondConvLSTMSigLayer, self).__init__(layer_param)
        assert 5 == self.input.ndim
        assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)

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

        self.temperature_inverse = numpy_floatX(layer_param.get('temperature_inverse', 1.))
        self.fmap_size = (self.dim_in[1], self.dim_in[2])
        self.alpha = None

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
        
        # attention: LSTM -> hidden
        self.Wd_att = quick_init_xavier(self.rng, (self.feature_out, self.feature_in), self._s("Wd_att"))
        # attention: input -> hidden
        self.Wc_att = quick_init_xavier(self.rng, (self.feature_in, self.feature_in), self._s("Wc_att"))
        # attention: hidden bias
        self.b_att = quick_zero((self.feature_in, ), self._s("b_att"))
        # attention:
        self.U_att = quick_init_xavier(self.rng, (self.feature_in, 1), self._s("U_att"))
        self.c_att = quick_zero((1, ), self._s("c_att"))
        
        # collect all parameters
        self.param = [self.W_xi, self.W_hi, self.b_i,
                      self.W_xf, self.W_hf, self.b_f,
                      self.W_xo, self.W_ho, self.b_o,
                      self.W_xc, self.W_hc, self.b_c,
                      self.Wd_att, self.Wc_att, self.b_att,
                      self.U_att, self.c_att]
        if self.learn_padding:
            self.param.append(self.hidden_padding)

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "CondConvLSTMSigLayer-" + str(self.id)

    def step_fprop(self, x_t, h_tm1, c_tm1, alpha_, *args):
        # x_t input @ t (BS, IN, H, W)
        # h_tm1 lstm hidden state @ t-1 (BS, OUT, H, W)
        # c_tm1 lstm cell state @ t-1 (BS, OUT, H, W)

        pstate = TT.tensordot(self.Wd_att, h_tm1, axes=[[0], [1]]) # IN x BS x H x W
        pattend = TT.tensordot(self.Wc_att, x_t, axes=[[0], [1]]) + self.b_att.dimshuffle(0, 'x', 'x', 'x') # IN x BS x H x W
        pattend = quick_activation(pattend + pstate, 'tanh')
        
        alpha = TT.tensordot(self.U_att, pattend, axes=[[0], [0]]) + self.c_att.dimshuffle(0, 'x', 'x', 'x')  # 1 x BS x H x W
        alpha_shp = alpha.shape
        alpha = quick_activation(alpha.reshape((alpha_shp[1],alpha_shp[2],alpha_shp[3])), 'sigmoid') # BS x H x W
        attend = x_t * alpha.dimshuffle(0, 'x', 1, 2) # BS x IN X H x W
        # print '\n\ncheck\n\n'

        input_gate = quick_activation(conv2d_same(attend, self.W_xi, (None, ) + self.dim_in,
                                                  self.kernel_size, self.input_padding)
                                      + conv2d_same(h_tm1, self.W_hi, (None, ) + self.dim_out,
                                                    self.transition_mat_size, self.hidden_padding)
                                      #+ c_tm1 * self.W_ci.dimshuffle('x', 0, 'x', 'x')
                                      + self.b_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        forget_gate = quick_activation(conv2d_same(attend, self.W_xf, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                       + conv2d_same(h_tm1, self.W_hf, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.hidden_padding)
                                       #+ c_tm1 * self.W_cf.dimshuffle('x', 0, 'x', 'x')
                                       + self.b_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        c_t = forget_gate * c_tm1 \
              + input_gate * quick_activation(conv2d_same(attend, self.W_xc, (None, ) + self.dim_in,
                                                          self.kernel_size, self.input_padding)
                                              + conv2d_same(h_tm1, self.W_hc, (None, ) + self.dim_out,
                                                            self.transition_mat_size, self.hidden_padding)
                                              + self.b_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
        output_gate = quick_activation(conv2d_same(attend, self.W_xo, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                       + conv2d_same(h_tm1, self.W_ho, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.hidden_padding)
                                       #+ c_t * self.W_co.dimshuffle('x', 0, 'x', 'x')
                                       + self.b_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        h_t = output_gate * quick_activation(c_t, "tanh")

        return [h_t, c_t, alpha]

    def step_masked_fprop(self, x_t, mask_t, h_tm1, c_tm1, alpha_, *args):
        h_t, c_t, alpha = self.step_fprop(x_t, h_tm1, c_tm1, alpha_, *args)

        h_t = TT.switch(mask_t, h_t, h_tm1)
        c_t = TT.switch(mask_t, c_t, c_tm1)

        return [h_t, c_t, alpha]

    def init_states(self):
        return self.init_hidden_state, self.init_cell_state

    def fprop(self):

        # The dimension of self.mask is (Timestep, Minibatch).
        # We need to pad it to (Timestep, Minibatch, FeatureDim)
        # and keep the last one added dimensions broadcastable. TT.shape_padright
        # function is thus a good choice
        # input should be (Timestep, Minibatch, FeatureDim, Region)
        # however, x should be (Timestep, Minibatch, Region, FeatureDim) to scan
        # if it's in from of (TS, BS, DIN, RS), transform it to (TS, BS, RS, DIN)
        # self.input = self.input.dimshuffle((0, 1, 3, 2))

        #self.input = self.input.dimshuffle((0, 1, 3, 2))
        if self.mask is None:
            scan_input = [self.input]
            scan_fn = self.step_fprop
        else:
            scan_input = [self.input, TT.shape_padright(self.mask, 3)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        [self.output, self.cell_output, self.alpha], self.output_update = quick_unroll_scan(fn=scan_fn,
        #[self.output, self.cell_output, self.alpha], self.output_update = theano.scan(fn=scan_fn,
                                                                        outputs_info=[self.init_hidden_state,
                                                                                      self.init_cell_state,
                                                                                      quick_theano_zero(((self.minibatch_size,) + self.fmap_size))],
                                                                        sequences=scan_input,
                                                                        non_sequences=non_seqs,
                                                                        n_steps=self.n_steps
                                                                        )