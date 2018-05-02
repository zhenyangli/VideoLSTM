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
    Conditional (simplified) Convolutional LSTM with Attention by stacking two streams (rgb+flow).
    Cell is not connected with any gate, i.e. input gate, output gate, or forget gate.
'''
class DeepCondConvLSTMDecpLayer(Layer):
    def __init__(self, layer_param):
        super(DeepCondConvLSTMDecpLayer, self).__init__(layer_param)
        assert 5 == self.input.ndim
        assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)
        assert ("init_context_hidden_state" in layer_param or "init_context_cell_state" in layer_param)

        # get dims of context input and output
        self.ctx_dim_in = layer_param.get('ctx_dim_in', None)
        self.ctx_dim_out = layer_param.get('ctx_dim_out', None)
        self.context = layer_param.get('context', None)
        self.context_in = self.ctx_dim_in[0]
        self.context_out = self.ctx_dim_out[0]
        assert 5 == self.context.ndim

        self.input_receptive_field = layer_param['input_receptive_field']
        self.transition_receptive_field = layer_param['transition_receptive_field']
        self.context_input_receptive_field = layer_param['context_input_receptive_field']
        self.context_transition_receptive_field = layer_param['context_transition_receptive_field']

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
        self.context_kernel_size = (self.context_out, self.context_in,
                                    self.context_input_receptive_field[0], self.context_input_receptive_field[1])
        self.context_transition_mat_size = (self.context_out, self.context_out,
                                            self.context_transition_receptive_field[0], self.context_transition_receptive_field[1])
        # self.hybrid_transition_mat_size = (self.context_out, self.feature_out,
        #                                    self.context_transition_receptive_field[0], self.context_transition_receptive_field[1])

        self.temperature_inverse = numpy_floatX(layer_param.get('temperature_inverse', 1.))
        self.fmap_size = (self.dim_in[1], self.dim_in[2])
        self.alpha = None
        self.ctx_output = None

        ###############################
        # feature input to LSTM-pred
        self.W_pred_xi = quick_init_xavier(self.rng, self.kernel_size, self._s("W_pred_xi"))
        self.W_pred_xf = quick_init_xavier(self.rng, self.kernel_size, self._s("W_pred_xf"))
        self.W_pred_xo = quick_init_xavier(self.rng, self.kernel_size, self._s("W_pred_xo"))
        self.W_pred_xc = quick_init_xavier(self.rng, self.kernel_size, self._s("W_pred_xc"))

        # LSTM-pred to LSTM-pred
        self.W_pred_hi = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_pred_hi"))
        self.W_pred_hf = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_pred_hf"))
        self.W_pred_ho = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_pred_ho"))
        self.W_pred_hc = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_pred_hc"))
        if self.learn_padding:
            self.pred_hidden_padding = quick_zero((self.feature_out, ), self._s("pred_hidden_padding"))
        else:
            self.pred_hidden_padding = None

        # bias to LSTM-pred
        self.b_pred_i = quick_zero((self.feature_out, ), self._s("b_pred_i"))
        self.b_pred_f = quick_zero((self.feature_out, ), self._s("b_pred_f"))
        self.b_pred_o = quick_zero((self.feature_out, ), self._s("b_pred_o"))
        self.b_pred_c = quick_zero((self.feature_out, ), self._s("b_pred_c"))

        ###############################
        # context input to LSTM-infer
        self.W_infer_xi = quick_init_xavier(self.rng, self.context_kernel_size, self._s("W_infer_xi"))
        self.W_infer_xf = quick_init_xavier(self.rng, self.context_kernel_size, self._s("W_infer_xf"))
        self.W_infer_xo = quick_init_xavier(self.rng, self.context_kernel_size, self._s("W_infer_xo"))
        self.W_infer_xc = quick_init_xavier(self.rng, self.context_kernel_size, self._s("W_infer_xc"))

        # LSTM-infer to LSTM-infer
        self.W_infer_hi = quick_init_xavier(self.rng, self.context_transition_mat_size, self._s("W_infer_hi"))
        self.W_infer_hf = quick_init_xavier(self.rng, self.context_transition_mat_size, self._s("W_infer_hf"))
        self.W_infer_ho = quick_init_xavier(self.rng, self.context_transition_mat_size, self._s("W_infer_ho"))
        self.W_infer_hc = quick_init_xavier(self.rng, self.context_transition_mat_size, self._s("W_infer_hc"))
        if self.learn_padding:
            self.infer_hidden_padding = quick_zero((self.context_out, ), self._s("infer_hidden_padding"))
        else:
            self.infer_hidden_padding = None

        # bias to LSTM-infer
        self.b_infer_i = quick_zero((self.context_out, ), self._s("b_infer_i"))
        self.b_infer_f = quick_zero((self.context_out, ), self._s("b_infer_f"))
        self.b_infer_o = quick_zero((self.context_out, ), self._s("b_infer_o"))
        self.b_infer_c = quick_zero((self.context_out, ), self._s("b_infer_c"))

        # LSTM-pred to LSTM-infer (as contextual input)
        # self.W_infer_ci = quick_init_xavier(self.rng, self.hybrid_transition_mat_size, self._s("W_infer_ci"))
        # self.W_infer_cf = quick_init_xavier(self.rng, self.hybrid_transition_mat_size, self._s("W_infer_cf"))
        # self.W_infer_co = quick_init_xavier(self.rng, self.hybrid_transition_mat_size, self._s("W_infer_co"))
        # self.W_infer_cc = quick_init_xavier(self.rng, self.hybrid_transition_mat_size, self._s("W_infer_cc"))

        ###############################
        # attention: input -> hidden
        self.Wc_att = quick_init_xavier(self.rng, (self.feature_in, self.feature_in), self._s("Wc_att"))
        # attention: LSTM-infer -> hidden
        self.Wd_att = quick_init_xavier(self.rng, (self.context_out, self.feature_in), self._s("Wd_att"))
        # attention: LSTM-pred -> hidden
        self.We_att = quick_init_xavier(self.rng, (self.feature_out, self.feature_in), self._s("We_att"))
        # attention: hidden bias
        self.b_att = quick_zero((self.feature_in, ), self._s("b_att"))
        # attention:
        self.U_att = quick_init_xavier(self.rng, (self.feature_in, 1), self._s("U_att"))
        self.c_att = quick_zero((1, ), self._s("c_att"))

        # collect all parameters
        self.param = [self.W_pred_xi, self.W_pred_hi, self.b_pred_i,
                      self.W_pred_xf, self.W_pred_hf, self.b_pred_f,
                      self.W_pred_xo, self.W_pred_ho, self.b_pred_o,
                      self.W_pred_xc, self.W_pred_hc, self.b_pred_c,
                      self.W_infer_xi, self.W_infer_hi, self.b_infer_i,
                      self.W_infer_xf, self.W_infer_hf, self.b_infer_f,
                      self.W_infer_xo, self.W_infer_ho, self.b_infer_o,
                      self.W_infer_xc, self.W_infer_hc, self.b_infer_c,
                      self.Wc_att, self.Wd_att, self.We_att, self.b_att,
                      self.U_att, self.c_att]
        if self.learn_padding:
            self.param.append(self.pred_hidden_padding)
            self.param.append(self.infer_hidden_padding)

        self.is_recurrent = True
        self.fprop()

    def set_name(self):
        self.name = "DeepCondConvLSTMDecpLayer-" + str(self.id)

    def step_fprop(self, x_t, ctx_t, h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, *args):
        # x_t input @ t (BS, IN, H, W)
        # h_pred_tm1 lstm-pred hidden state @ t-1 (BS, OUT, H, W)
        # c_pred_tm1 lstm-pred cell state @ t-1 (BS, OUT, H, W)
        # h_infer_tm1 lstm-infer hidden state @ t-1 (BS, OUT, H, W)
        # c_infer_tm1 lstm-infer cell state @ t-1 (BS, OUT, H, W)

        # LSTM-infer (inference layer)
        input_gate_infer = quick_activation(conv2d_same(ctx_t, self.W_infer_xi, (None, ) + self.ctx_dim_in,
                                                   self.context_kernel_size, self.input_padding)
                                            + conv2d_same(h_infer_tm1, self.W_infer_hi, (None, ) + self.ctx_dim_out,
                                                     self.context_transition_mat_size, self.infer_hidden_padding)
                                            + self.b_infer_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        forget_gate_infer = quick_activation(conv2d_same(ctx_t, self.W_infer_xf, (None, ) + self.ctx_dim_in,
                                                   self.context_kernel_size, self.input_padding)
                                             + conv2d_same(h_infer_tm1, self.W_infer_hf, (None, ) + self.ctx_dim_out,
                                                      self.context_transition_mat_size, self.infer_hidden_padding)
                                             + self.b_infer_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        c_infer_t = forget_gate_infer * c_infer_tm1 \
                    + input_gate_infer * quick_activation(conv2d_same(ctx_t, self.W_infer_xc, (None, ) + self.ctx_dim_in,
                                                                 self.context_kernel_size, self.input_padding)
                                                          + conv2d_same(h_infer_tm1, self.W_infer_hc, (None, ) + self.ctx_dim_out,
                                                                   self.context_transition_mat_size, self.infer_hidden_padding)
                                                          + self.b_infer_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
        output_gate_infer = quick_activation(conv2d_same(ctx_t, self.W_infer_xo, (None, ) + self.ctx_dim_in,
                                                    self.context_kernel_size, self.input_padding)
                                             + conv2d_same(h_infer_tm1, self.W_infer_ho, (None, ) + self.ctx_dim_out,
                                                      self.context_transition_mat_size, self.infer_hidden_padding)
                                             + self.b_infer_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        h_infer_t = output_gate_infer * quick_activation(c_infer_t, "tanh")

        # attention mechanism
        pstate_infer = TT.tensordot(self.Wd_att, h_infer_t, axes=[[0], [1]]) # IN x BS x H x W
        pstate_pred = TT.tensordot(self.We_att, h_pred_tm1, axes=[[0], [1]]) # IN x BS x H x W
        pattend = TT.tensordot(self.Wc_att, x_t, axes=[[0], [1]]) + self.b_att.dimshuffle(0, 'x', 'x', 'x') # IN x BS x H x W
        pattend = quick_activation(pattend + pstate_infer + pstate_pred, 'tanh')
        
        alpha = TT.tensordot(self.U_att, pattend, axes=[[0], [0]]) + self.c_att.dimshuffle(0, 'x', 'x', 'x') # 1 x BS x H x W
        alpha_shp = alpha.shape
        #alpha = quick_activation(alpha.reshape((alpha_shp[1],alpha_shp[2],alpha_shp[3])), 'sigmoid') # BS x H x W
        alpha = quick_activation(alpha.reshape((alpha_shp[1],alpha_shp[2]*alpha_shp[3])), 'softmax') # BS x (H x W)
        alpha = alpha.reshape((alpha_shp[1],alpha_shp[2],alpha_shp[3])) # BS x H x W
        attend = x_t * alpha.dimshuffle(0, 'x', 1, 2) # BS x IN X H x W
        # print '\n\ncheck\n\n'

        # LSTM-pred (prediction layer)
        input_gate_pred = quick_activation(conv2d_same(attend, self.W_pred_xi, (None, ) + self.dim_in,
                                                  self.kernel_size, self.input_padding)
                                           + conv2d_same(h_pred_tm1, self.W_pred_hi, (None, ) + self.dim_out,
                                                    self.transition_mat_size, self.pred_hidden_padding)
                                           + self.b_pred_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        forget_gate_pred = quick_activation(conv2d_same(attend, self.W_pred_xf, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                            + conv2d_same(h_pred_tm1, self.W_pred_hf, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.pred_hidden_padding)
                                            + self.b_pred_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        c_pred_t = forget_gate_pred * c_pred_tm1 \
                   + input_gate_pred * quick_activation(conv2d_same(attend, self.W_pred_xc, (None, ) + self.dim_in,
                                                               self.kernel_size, self.input_padding)
                                                        + conv2d_same(h_pred_tm1, self.W_pred_hc, (None, ) + self.dim_out,
                                                                 self.transition_mat_size, self.pred_hidden_padding)
                                                        + self.b_pred_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
        output_gate_pred = quick_activation(conv2d_same(attend, self.W_pred_xo, (None, ) + self.dim_in,
                                                   self.kernel_size, self.input_padding)
                                            + conv2d_same(h_pred_tm1, self.W_pred_ho, (None, ) + self.dim_out,
                                                     self.transition_mat_size, self.pred_hidden_padding)
                                            + self.b_pred_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
        h_pred_t = output_gate_pred * quick_activation(c_pred_t, "tanh")

        return [h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha]

    def step_masked_fprop(self, x_t, ctx_t, mask_t, h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, *args):

        h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha = self.step_fprop(x_t, ctx_t, \
                                                        h_pred_tm1, c_pred_tm1, h_infer_tm1, c_infer_tm1, alpha_, *args)

        h_pred_t = TT.switch(mask_t, h_pred_t, h_pred_tm1)
        c_pred_t = TT.switch(mask_t, c_pred_t, c_pred_tm1)
        h_infer_t = TT.switch(mask_t, h_infer_t, h_infer_tm1)
        c_infer_t = TT.switch(mask_t, c_infer_t, c_infer_tm1)

        return [h_pred_t, c_pred_t, h_infer_t, c_infer_t, alpha]

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

        #self.input = self.input.dimshuffle((0, 1, 3, 2))
        if self.mask is None:
            scan_input = [self.input, self.context]
            scan_fn = self.step_fprop
        else:
            scan_input = [self.input, self.context, TT.shape_padright(self.mask, 3)]
            scan_fn = self.step_masked_fprop

        non_seqs = self.param
        #[self.output, self.cell_output, self.ctx_output, self.ctx_cell_output, self.alpha], self.output_update = quick_unroll_scan(fn=scan_fn,
        [self.output, self.cell_output, self.ctx_output, self.ctx_cell_output, self.alpha], self.output_update = theano.scan(fn=scan_fn,
                                                                        outputs_info=[self.init_hidden_state,
                                                                                      self.init_cell_state,
                                                                                      self.init_context_hidden_state,
                                                                                      self.init_context_cell_state,
                                                                                      quick_theano_zero(((self.minibatch_size,) + self.fmap_size))],
                                                                        sequences=scan_input,
                                                                        non_sequences=non_seqs,
                                                                        n_steps=self.n_steps
                                                                        )