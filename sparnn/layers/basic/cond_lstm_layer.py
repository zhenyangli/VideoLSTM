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
    Conditional (simplified) LSTM with Attention.
    Cell is not connected with any gate, i.e. input gate, output gate, or forget gate.
'''
class CondLSTMLayer(Layer):
    def __init__(self, layer_param):
        super(CondLSTMLayer, self).__init__(layer_param)
        assert 4 == self.input.ndim
        assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)

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
        
        self.temperature_inverse = numpy_floatX(layer_param.get('temperature_inverse', 1.))
        self.alpha = None
        self.attend = None
        self.region_size = self.dim_in[1]
        
        # input to LSTM
        self.W_xi = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xi"), scale=0.1)
        self.W_xf = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xf"), scale=0.1)
        self.W_xo = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xo"), scale=0.1)
        self.W_xc = quick_init_norm(self.rng, self.input_mat_size, self._s("W_xc"), scale=0.1)

        # LSTM to LSTM ( orthogonal weights do not work well?! )
        # self.W_hi = quick_init_ortho(self.rng, self.transition_mat_size, self._s("W_hi"))
        # self.W_hf = quick_init_ortho(self.rng, self.transition_mat_size, self._s("W_hf"))
        # self.W_ho = quick_init_ortho(self.rng, self.transition_mat_size, self._s("W_ho"))
        # self.W_hc = quick_init_ortho(self.rng, self.transition_mat_size, self._s("W_hc"))
        self.W_hi = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hi"), scale=0.1)
        self.W_hf = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hf"), scale=0.1)
        self.W_ho = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_ho"), scale=0.1)
        self.W_hc = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_hc"), scale=0.1)

        # bias to LSTM
        self.b_i = quick_zero((self.feature_out, ), self._s("b_i"))
        self.b_f = quick_zero((self.feature_out, ), self._s("b_f"))
        self.b_o = quick_zero((self.feature_out, ), self._s("b_o"))
        self.b_c = quick_zero((self.feature_out, ), self._s("b_c"))
        
        # attention: input -> hidden
        self.Wc_att = quick_init_norm(self.rng, (self.feature_in, self.feature_in), self._s("Wc_att"), scale=0.1)
        # attention: LSTM -> hidden
        self.Wd_att = quick_init_norm(self.rng, (self.feature_out, self.feature_in), self._s("Wd_att"), scale=0.1)
        # attention: hidden bias
        self.b_att = quick_zero((self.feature_in, ), self._s("b_att"))
        # optional "deep" attention
        # if options['n_layers_att'] > 1:
        #     for lidx in xrange(1, options['n_layers_att']):
        # attention:
        self.U_att = quick_init_norm(self.rng, (self.feature_in, 1), self._s("U_att"), scale=0.1)
        self.c_att = quick_zero((1, ), self._s("c_att"))

        # collect all parameters
        self.param = [self.W_xi, self.W_hi, self.b_i,
                      self.W_xf, self.W_hf, self.b_f,
                      self.W_xo, self.W_ho, self.b_o,
                      self.W_xc, self.W_hc, self.b_c,
                      self.Wc_att, self.Wd_att, self.b_att,
                      self.U_att, self.c_att]

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "CondLSTMLayer-" + str(self.id)

    def step_fprop(self, x_t, h_tm1, c_tm1, alpha_, attend_, *args):
        # x_t input @ t
        # h_tm1 lstm hidden state @ t-1
        # c_tm1 lstm cell state @ t-1
        pstate = TT.dot(h_tm1, self.Wd_att) # BS x DIN
        pattend = TT.dot(x_t, self.Wc_att) + self.b_att # BS x RS x DIN
        pattend = pattend + pstate.dimshuffle(0, 'x', 1) #pstate[:,None,:]
        #pattend_list = []
        #pattend_list.append(pattend)
        pattend = quick_activation(pattend, 'tanh')

        alpha = TT.dot(pattend, self.U_att) + self.c_att
        #alpha_pre = alpha
        alpha_shp = alpha.shape
        alpha = quick_activation(self.temperature_inverse*alpha.reshape((alpha_shp[0],alpha_shp[1])), 'softmax')
        attend = (x_t * alpha.dimshuffle(0, 1, 'x')).sum(1) #alpha[:,:,None] # current region to attend: BS x DIN
        # print '\n\ncheck\n\n'

        input_gate = quick_activation(TT.dot(attend, self.W_xi)
                                      + TT.dot(h_tm1, self.W_hi)
                                      + self.b_i, "sigmoid")
        forget_gate = quick_activation(TT.dot(attend, self.W_xf)
                                       + TT.dot(h_tm1, self.W_hf)
                                       + self.b_f, "sigmoid")
        c_t = forget_gate * c_tm1 \
              + input_gate * quick_activation(TT.dot(attend, self.W_xc)
                                              + TT.dot(h_tm1, self.W_hc)
                                              + self.b_c, "tanh")
        output_gate = quick_activation(TT.dot(attend, self.W_xo)
                                       + TT.dot(h_tm1, self.W_ho)
                                       + self.b_o, "sigmoid")
        h_t = output_gate * quick_activation(c_t, "tanh")

        return [h_t, c_t, alpha, attend]

    def step_masked_fprop(self, x_t, mask_t, h_tm1, c_tm1, alpha_, attend_, *args):
        h_t, c_t, alpha, attend = self.step_fprop(x_t, h_tm1, c_tm1, alpha_, attend_, *args)

        h_t = TT.switch(mask_t, h_t, h_tm1)
        c_t = TT.switch(mask_t, c_t, c_tm1)

        return [h_t, c_t, alpha, attend]

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

        self.input = self.input.dimshuffle((0, 1, 3, 2))
        if self.mask is None:
            scan_input = [self.input]
            scan_fn = self.step_fprop
        else:
            scan_input = [self.input, TT.shape_padright(self.mask, 1)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        [self.output, self.cell_output, self.alpha, self.attend], self.output_update = quick_unroll_scan(fn=scan_fn,
        #[self.output, self.cell_output, self.alpha, self.attend], self.output_update = theano.scan(fn=scan_fn,
                                                                        outputs_info=[self.init_hidden_state,
                                                                                      self.init_cell_state,
                                                                                      quick_theano_zero((self.minibatch_size, self.region_size)),
                                                                                      quick_theano_zero((self.minibatch_size, self.feature_in))],
                                                                        sequences=scan_input,
                                                                        non_sequences=non_seqs,
                                                                        n_steps=self.n_steps
                                                                        )