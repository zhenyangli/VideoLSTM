__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
import cPickle

from scipy import stats
from sparnn.utils import *
import sys

sys.setrecursionlimit(15000)

logger = logging.getLogger(__name__)

'''

In Model,
middle_layers is a list
cost_layer is a layer that stores the optimizing target
outputs, errors are all lists, with format [{"name":string, "value":tensor}]

'''


class Model(object):
    def __init__(self, model_param):
        self.interface_layer = model_param['interface_layer']
        self.middle_layers = model_param['middle_layers']
        self.cost_layer = model_param['cost_layer']
        self.last_n = model_param['last_n']
        self.outputs = model_param.get('outputs', None)
        self.errors = model_param.get('errors', None)
        self.name = model_param["name"]
        self.problem_type = model_param["problem_type"]
        self.mode = "train"
        self.param = []
        for layer in self.middle_layers:
            self.param += layer.param
        self.param += self.cost_layer.param
        self.set_mode(self.mode)
        self.grad = self.get_grad()
        self.cost_func = self.get_cost_func()
        self.output_func_dict = self.get_output_func_dict()
        self.error_func_dict = self.get_error_func_dict()

    @staticmethod
    def save(model, path):
        f = open(path, 'wb')
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(path):
        logger.info("Loading Model From " + path)
        f = open(path, 'rb')
        model = cPickle.load(f)
        f.close()
        return model

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['grad']
        del state['cost_func']
        del state['output_func_dict']
        del state['error_func_dict']
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        logger.info("Rebuilding Cost Function, Output Functions and Error Functions")
        self.grad = self.get_grad()
        self.cost_func = self.get_cost_func()
        self.output_func_dict = self.get_output_func_dict()
        self.error_func_dict = self.get_error_func_dict()

    def set_mode(self, mode):
        self.mode = mode
        for layer in self.middle_layers:
            layer.set_mode(mode)
        self.cost_layer.set_mode(mode)

    def get_grad(self):
        return TT.grad(self.cost_layer.output, self.param)

    # TODO Add Updates
    def get_cost_func(self):
        return theano.function(inputs=self.interface_layer.symbols(), outputs=self.cost_layer.output,
                               on_unused_input='warn')

    def get_update_func(self, updates, other_param_list):
        inner_updates = []
        for layer in self.middle_layers:
            if layer.output_update is not None:
                #print 'Layer updates not None', str(layer.output_update)
                inner_updates += layer.output_update
        return theano.function(inputs=self.interface_layer.symbols() + other_param_list,
                               outputs=self.cost_layer.output, updates=updates + inner_updates,
                               on_unused_input='warn')

    # TODO Add Updates and need future revision
    def get_output_func_dict(self):
        if self.outputs is None:
            return {}
        else:
            return {output['name']: theano.function(inputs=self.interface_layer.symbols(), outputs=output["value"],
                                                    on_unused_input='warn') for output in self.outputs}

    # TODO Add Updates
    def get_error_func_dict(self):
        if self.errors is None:
            return {}
        else:
            return {error['name']: theano.function(inputs=self.interface_layer.symbols(), outputs=error['value'],
                                                   on_unused_input='warn')
                    for error in self.errors}

    def get_cost(self, data_iterator):
        ret = 0
        old_mode = self.mode
        self.set_mode('predict')
        data_iterator.begin(do_shuffle=False)
        while True:
            ret += self.cost_func(*(data_iterator.get_batch()))
            data_iterator.next()
            if data_iterator.no_batch_left():
                break
        self.set_mode(old_mode)
        return ret / data_iterator.total()

    def get_error_dict(self, data_iterator):
        if len(self.error_func_dict) > 0:
            l = {}
            for key in self.error_func_dict:
                ret = 0
                old_mode = self.mode
                self.set_mode('predict')
                data_iterator.begin(do_shuffle=False)
                while True:
                    ret += self.error_func_dict[key](*(data_iterator.get_batch()))
                    data_iterator.next()
                    if data_iterator.no_batch_left():
                        break
                self.set_mode(old_mode)
                l['key'] = ret / data_iterator.total()
            return l
        #else: # disable, since only for binary predictions
            #error = 0
            #old_mode = self.mode
            #self.set_mode('predict')
            #data_iterator.begin(do_shuffle=False)
            #while True:
            #    output = self.output_func_dict[0](*data_iterator.input_batch())
            #    target = data_iterator.output_batch()[0]
            #    pred = output.reshape((output.shape[0])) > 0.5
            #    target = target.reshape(target.shape[0]).astype("bool")
            #    error += (pred == target).sum()
            #    data_iterator.next()
            #    if data_iterator.no_batch_left():
            #        break
            #error = 1 - (error / numpy_floatX(data_iterator.total()))
            #self.set_mode(old_mode)
            #return [error]

    def get_acc(self, data_iterator):
        preds = numpy.zeros((data_iterator.total(),)).astype(theano.config.floatX)
        old_mode = self.mode
        self.set_mode('predict')
        data_iterator.begin(do_shuffle=False)
        while True:
            #preds[data_iterator.current_batch_indices] = self.output_func_dict['prediction'](*(data_iterator.get_batch()))
            output = self.output_func_dict['prediction'](*(data_iterator.get_batch()))
            if data_iterator.current_batch_indices.size < output.size:
                print 'last minibatch size smaller than', output.size
            preds[data_iterator.current_batch_indices] = output[:data_iterator.current_batch_size]
            data_iterator.next()
            if data_iterator.no_batch_left():
                break
        self.set_mode(old_mode)

        fileprefix = 'results-last{}-'.format(self.last_n)
        tempfilename = fileprefix + self.name + '-' + data_iterator.name + '.txt'
        f = open(tempfilename, 'w')
        vid_idx = 0
        resultstr='{} '.format(vid_idx)
        for i in xrange(data_iterator.total()):
            if data_iterator.video_indices[i] == vid_idx:
                resultstr=resultstr+'{},'.format(int(preds[i]))
            else:
                vid_idx = vid_idx+1
                resultstr=resultstr[:-1]+'\n'
                f.write(resultstr)
                resultstr='{} '.format(vid_idx)
                resultstr=resultstr+'{},'.format(int(preds[i]))
        resultstr=resultstr[:-1]+'\n'
        f.write(resultstr)
        f.close()

        f = open(tempfilename,'r')
        lines = f.readlines()
        f.close()
        pred  = numpy.zeros(len(lines)).astype('int64')
        for i in xrange(len(lines)):
            try:
                s=lines[i].split(' ')[1]
                s=s[0:-1]
                s=s.split(',')
                s = [int(x) for x in s]
                s = numpy.array(s)
                s = stats.mode(s)[0][0]
                pred[i] = int(s)
            except IndexError:
                print 'One blank index skipped'
                pred[i] = -1

        f = open(data_iterator.labels_file,'r')
        lines = f.readlines()
        f.close()
        truth = numpy.zeros(len(lines)).astype('int64')
        for i in xrange(len(lines)):
            s=lines[i][0:-1]
            truth[i] = int(s)
        return (truth==pred).mean()

    def get_mAP(self, data_iterator):
        ret = 0
        return ret

    def total_param_num(self):
        ret = 0
        for layer in self.middle_layers:
            ret += layer.total_param_num()
        ret += self.cost_layer.total_param_num()
        return ret

    def print_stat(self):
        logger.info("Model Name: " + self.name)
        self.interface_layer.print_stat()
        for layer in self.middle_layers:
            layer.print_stat()
        self.cost_layer.print_stat()
        logger.info("The Total Model Param is " + str(self.total_param_num()))
        if self.outputs is not None:
            logger.info("Output List:")
            for output in self.outputs:
                logger.info("   name: " + str(output['name']) + ", value: " + str(output['value']))
        if self.errors is not None:
            logger.info("Error List:")
            for error in self.errors:
                logger.info("   name: " + error['name'] + ", value: " + str(error['value']))
