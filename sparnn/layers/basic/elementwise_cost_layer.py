__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ElementwiseCostLayer(Layer):
    def __init__(self, layer_param):
        super(ElementwiseCostLayer, self).__init__(layer_param)
        # TODO !Important In fact, when the input dimension of the cost layer is 4 means, you are actually mapping
        # TODO a sequence to a time-independent variable. Here I just add an if statement to do the decision
        self.target = layer_param['target']
        self.weight = layer_param.get('weight', None)
        self.cost_func = layer_param['cost_func']
        self.regularization = layer_param.get('regularization', None)
        self.param_layers = layer_param.get('param_layers', None)
        self.penalty_rate = layer_param.get('penalty_rate', None)
        if self.regularization is not None:
            assert self.param_layers is not None and self.penalty_rate is not None
        if self.weight is not None:
            assert (type(self.input) is list) and (type(self.target) is list) and (type(self.weight) is list) and (
                type(self.cost_func) is list) and (type(self.mask) is list)
            assert len(self.input) == len(self.target) == len(self.weight)
        self.fprop()

    def set_name(self):
        self.name = "ElementwiseCostLayer-" + str(self.id)

    def fprop(self):
        if self.weight is not None:
            output = sum(
                weight * quick_cost(input, target, cost_func, mask) for
                weight, target, input, cost_func, mask in
                zip(self.weight, self.target, self.input, self.cost_func, self.mask))
        else:
            output = quick_cost(self.input, self.target, self.cost_func, self.mask)

        if self.regularization is not None:
            all_params = []
            for layer in self.param_layers:
                all_params += layer.param
            output = output + self.penalty_rate*quick_penalty(all_params, self.regularization)

        self.output = output

    def print_stat(self):
        logger.info(self.name + " : ")
        logger.info("   Cost Function: " + str(self.cost_func))
        if self.regularization is not None:
            logger.info("   Regularization: " + str(self.regularization))
            logger.info("   Penalty Rate: " + str(self.penalty_rate))
        if self.weight is not None:
            logger.info("   Weights: " + str(self.weight))
