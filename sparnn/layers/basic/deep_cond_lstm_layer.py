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
    Conditional (simplified) LSTM with Attention by stacking two streams (rgb+flow).
    Memory cell is not connected with any gate, i.e. input gate, output gate, or forget gate.
'''
class DeepCondLSTMLayer(Layer):
    def __init__(self, layer_param):
        super(DeepCondLSTMLayer, self).__init__(layer_param)
        assert 4 == self.input.ndim
        assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)
        assert ("init_context_hidden_state" in layer_param or "init_context_cell_state" in layer_param)
        
        # get dims of context input and output
        self.ctx_dim_in = layer_param.get('ctx_dim_in', None)
        self.ctx_dim_out = layer_param.get('ctx_dim_out', None)
        self.context = layer_param.get('context', None)
        self.context_in = self.ctx_dim_in[0]
        self.context_out = self.ctx_dim_out[0]
        assert 3 == self.context.ndim

        self.gate_activation = layer_param.get('gate_activation', 'sigmoid')
        self.modular_activation = layer_param.get('modular_activation', 'tanh')
        self.hidden_activation = layer_param.get('hidden_activation', 'tanh')

        self.init_hidden_state = layer_param.get("init_hidden_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_cell_state = layer_param.get("init_cell_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_hidden_state = TT.unbroadcast(self.init_hidden_state, *range(self.init_hidden_state.ndim))
        self.init_cell_state = TT.unbroadcast(self.init_cell_state, *range(self.init_cell_state.ndim))
        self.init_context_hidden_state = layer_param.get("init_context_hidden_state", quick_theano_zero((self.minibatch_size,) + self.ctx_dim_out))
        self.init_context_cell_state = layer_param.get("init_context_cell_state", quick_theano_zero((self.minibatch_size,) + self.ctx_dim_out))
        self.init_context_hidden_state = TT.unbroadcast(self.init_context_hidden_state, *range(self.init_context_hidden_state.ndim))
        self.init_context_cell_state = TT.unbroadcast(self.init_context_cell_state, *range(self.init_context_cell_state.ndim))
        if 'n_steps' in layer_param:
            self.n_steps = layer_param['n_steps']
        else:
            self.n_steps = layer_param.get('n_steps', self.input.shape[0])
        self.input_mat_size = (self.feature_in, self.feature_out)
        self.transition_mat_size = (self.feature_out, self.feature_out)
        self.context_input_mat_size = (self.context_in, self.context_out)
        self.context_transition_mat_size = (self.context_out, self.context_out)

        self.temperature_inverse = numpy_floatX(layer_param.get('temperature_inverse', 1.))
        self.alpha = None
        self.attend = None
        self.ctx_output = None
        self.region_size = self.dim_in[1]
        
        ###############################
        # feature input to LSTM-pred
        self.W_pred_xi = quick_init_norm(self.rng, self.input_mat_size, self._s("W_pred_xi"), scale=0.1)
        self.W_pred_xf = quick_init_norm(self.rng, self.input_mat_size, self._s("W_pred_xf"), scale=0.1)
        self.W_pred_xo = quick_init_norm(self.rng, self.input_mat_size, self._s("W_pred_xo"), scale=0.1)
        self.W_pred_xc = quick_init_norm(self.rng, self.input_mat_size, self._s("W_pred_xc"), scale=0.1)

        # LSTM-pred to LSTM-pred
        self.W_pred_hi = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_pred_hi"), scale=0.1)
        self.W_pred_hf = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_pred_hf"), scale=0.1)
        self.W_pred_ho = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_pred_ho"), scale=0.1)
        self.W_pred_hc = quick_init_norm(self.rng, self.transition_mat_size, self._s("W_pred_hc"), scale=0.1)

        # bias to LSTM-pred
        self.b_pred_i = quick_zero((self.feature_out, ), self._s("b_pred_i"))
        self.b_pred_f = quick_zero((self.feature_out, ), self._s("b_pred_f"))
        self.b_pred_o = quick_zero((self.feature_out, ), self._s("b_pred_o"))
        self.b_pred_c = quick_zero((self.feature_out, ), self._s("b_pred_c"))

        ###############################
        # context input to LSTM-infer
        self.W_infer_xi = quick_init_norm(self.rng, self.context_input_mat_size, self._s("W_infer_xi"), scale=0.1)
        self.W_infer_xf = quick_init_norm(self.rng, self.context_input_mat_size, self._s("W_infer_xf"), scale=0.1)
        self.W_infer_xo = quick_init_norm(self.rng, self.context_input_mat_size, self._s("W_infer_xo"), scale=0.1)
        self.W_infer_xc = quick_init_norm(self.rng, self.context_input_mat_size, self._s("W_infer_xc"), scale=0.1)

        # LSTM-infer to LSTM-infer
        self.W_infer_hi = quick_init_norm(self.rng, self.context_transition_mat_size, self._s("W_infer_hi"), scale=0.1)
        self.W_infer_hf = quick_init_norm(self.rng, self.context_transition_mat_size, self._s("W_infer_hf"), scale=0.1)
        self.W_infer_ho = quick_init_norm(self.rng, self.context_transition_mat_size, self._s("W_infer_ho"), scale=0.1)
        self.W_infer_hc = quick_init_norm(self.rng, self.context_transition_mat_size, self._s("W_infer_hc"), scale=0.1)

        # bias to LSTM-infer
        self.b_infer_i = quick_zero((self.context_out, ), self._s("b_infer_i"))
        self.b_infer_f = quick_zero((self.context_out, ), self._s("b_infer_f"))
        self.b_infer_o = quick_zero((self.context_out, ), self._s("b_infer_o"))
        self.b_infer_c = quick_zero((self.context_out, ), self._s("b_infer_c"))

        # LSTM-pred to LSTM-infer (as contextual input)
        self.W_infer_ci = quick_init_norm(self.rng, (self.feature_out, self.context_out), self._s("W_infer_ci"), scale=0.1)
        self.W_infer_cf = quick_init_norm(self.rng, (self.feature_out, self.context_out), self._s("W_infer_cf"), scale=0.1)
        self.W_infer_co = quick_init_norm(self.rng, (self.feature_out, self.context_out), self._s("W_infer_co"), scale=0.1)
        self.W_infer_cc = quick_init_norm(self.rng, (self.feature_out, self.context_out), self._s("W_infer_cc"), scale=0.1)

        ###############################
        # attention: input -> hidden
        self.Wc_att = quick_init_norm(self.rng, (self.feature_in, self.feature_in), self._s("Wc_att"), scale=0.1)
        # attention: LSTM-infer -> hidden
        self.Wd_att = quick_init_norm(self.rng, (self.context_out, self.feature_in), self._s("Wd_att"), scale=0.1)
        # attention: hidden bias
        self.b_att = quick_zero((self.feature_in, ), self._s("b_att"))
        # optional "deep" attention
        # if options['n_layers_att'] > 1:
        #     for lidx in xrange(1, options['n_layers_att']):
        # attention:
        self.U_att = quick_init_norm(self.rng, (self.feature_in, 1), self._s("U_att"), scale=0.1)
        self.c_att = quick_zero((1, ), self._s("c_att"))

        # all parameters
        self.param = [self.W_pred_xi, self.W_pred_hi, self.b_pred_i,
                      self.W_pred_xf, self.W_pred_hf, self.b_pred_f,
                      self.W_pred_xo, self.W_pred_ho, self.b_pred_o,
                      self.W_pred_xc, self.W_pred_hc, self.b_pred_c,
                      self.W_infer_xi, self.W_infer_hi, self.b_infer_i, self.W_infer_ci,
                      self.W_infer_xf, self.W_infer_hf, self.b_infer_f, self.W_infer_cf,
                      self.W_infer_xo, self.W_infer_ho, self.b_infer_o, self.W_infer_co,
                      self.W_infer_xc, self.W_infer_hc, self.b_infer_c, self.W_infer_cc,
                      self.Wc_att, self.Wd_att, self.b_att,
                      self.U_att, self.c_att]

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "DeepCondLSTMLayer-" + str(self.id)

    def step_fprop(self, x_t, ctx_t, h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, attend_, *args):
        # x_t input @ t
        # ctx_t context @ t
        # h_pred_tm1 lstm pred hidden state @ t-1
        # c_pred_tm1 lstm pred cell state @ t-1
        # h_infer_tm1 lstm infer hidden state @ t-1
        # c_infer_tm1 lstm infer cell state @ t-1

        # LSTM-infer (inference layer)
        input_gate_infer = quick_activation(TT.dot(ctx_t, self.W_infer_xi)
                                      + TT.dot(h_infer_tm1, self.W_infer_hi)
                                      + TT.dot(h_pred_tm1, self.W_infer_ci)
                                      + self.b_infer_i, "sigmoid")
        forget_gate_infer = quick_activation(TT.dot(ctx_t, self.W_infer_xf)
                                       + TT.dot(h_infer_tm1, self.W_infer_hf)
                                       + TT.dot(h_pred_tm1, self.W_infer_cf)
                                       + self.b_infer_f, "sigmoid")
        c_infer_t = forget_gate_infer * c_infer_tm1 \
              + input_gate_infer * quick_activation(TT.dot(ctx_t, self.W_infer_xc)
                                              + TT.dot(h_infer_tm1, self.W_infer_hc)
                                              + TT.dot(h_pred_tm1, self.W_infer_cc)
                                              + self.b_infer_c, "tanh")
        output_gate_infer = quick_activation(TT.dot(ctx_t, self.W_infer_xo)
                                       + TT.dot(h_infer_tm1, self.W_infer_ho)
                                       + TT.dot(h_pred_tm1, self.W_infer_co)
                                       + self.b_infer_o, "sigmoid")
        h_infer_t = output_gate_infer * quick_activation(c_infer_t, "tanh")

        # attention mechanism
        pstate = TT.dot(h_infer_t, self.Wd_att) # BS x DIN
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

        # LSTM-pred (prediction layer)
        input_gate_pred = quick_activation(TT.dot(attend, self.W_pred_xi)
                                      + TT.dot(h_pred_tm1, self.W_pred_hi)
                                      + self.b_pred_i, "sigmoid")
        forget_gate_pred = quick_activation(TT.dot(attend, self.W_pred_xf)
                                       + TT.dot(h_pred_tm1, self.W_pred_hf)
                                       + self.b_pred_f, "sigmoid")
        c_pred_t = forget_gate_pred * c_pred_tm1 \
              + input_gate_pred * quick_activation(TT.dot(attend, self.W_pred_xc)
                                              + TT.dot(h_pred_tm1, self.W_pred_hc)
                                              + self.b_pred_c, "tanh")
        output_gate_pred = quick_activation(TT.dot(attend, self.W_pred_xo)
                                       + TT.dot(h_pred_tm1, self.W_pred_ho)
                                       + self.b_pred_o, "sigmoid")
        h_pred_t = output_gate_pred * quick_activation(c_pred_t, "tanh")

        return [h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha, attend]

    def step_masked_fprop(self, x_t, ctx_t, mask_t, h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, attend_, *args):

        h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha, attend = self.step_fprop(x_t, ctx_t, \
                                               h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, attend_, *args)

        h_pred_t = TT.switch(mask_t, h_pred_t, h_pred_tm1)
        c_pred_t = TT.switch(mask_t, c_pred_t, c_pred_tm1)
        h_infer_t = TT.switch(mask_t, h_infer_t, h_infer_tm1)
        c_infer_t = TT.switch(mask_t, c_infer_t, c_infer_tm1)

        return [h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha, attend]

    def init_states(self):
        return self.init_hidden_state, self.init_cell_state, self.init_context_hidden_state, self.init_context_cell_state

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
            scan_input = [self.input, self.context]
            scan_fn = self.step_fprop

        else:
            scan_input = [self.input, self.context, TT.shape_padright(self.mask, 1)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        [self.output, self.cell_output, self.ctx_output, self.ctx_cell_output, self.alpha, self.attend], self.output_update = \
                                                                        quick_unroll_scan(fn=scan_fn,
                                                                        #theano.scan(fn=scan_fn,
                                                                        outputs_info=[self.init_hidden_state,
                                                                                      self.init_cell_state,
                                                                                      self.init_context_hidden_state,
                                                                                      self.init_context_cell_state,
                                                                                      quick_theano_zero((self.minibatch_size, self.region_size)),
                                                                                      quick_theano_zero((self.minibatch_size, self.feature_in))],
                                                                        sequences=scan_input,
                                                                        non_sequences=non_seqs,
                                                                        n_steps=self.n_steps
                                                                        )