__author__ = 'sxjscience'
import numpy
import time
import theano
import logging
import cPickle
import os
import theano.tensor as TT

logger = logging.getLogger(__name__)
from sparnn.models import Model
from sparnn.utils import *


'''
"autosave_mode": "best"

'''


class Optimizer(object):
    def __init__(self,
                 model,
                 train_data_iterator,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param):
        self.model = model
        self.train_data_iterator = train_data_iterator
        self.valid_data_iterator = valid_data_iterator
        self.test_data_iterator = test_data_iterator
        self.id = hyper_param["id"]
        self.start_epoch = hyper_param.get("start_epoch", 0)
        self.max_epoch = hyper_param["max_epoch"]
        self.autosave_mode = hyper_param.get("autosave_mode", None)
        self.do_shuffle = hyper_param.get("do_shuffle", None)
        self.save_path = hyper_param.get("save_path", "./")
        self.save_interval = hyper_param.get("save_interval", None)
        self.max_epochs_no_best = hyper_param.get("max_epochs_no_best", None)
        self.clip_threshold = numpy_floatX(hyper_param['clip_threshold']) if 'clip_threshold' in hyper_param else None
        self.verbose = hyper_param.get("verbose", None)
        self.best_validation_error = numpy.inf
        self.current_validation_error = numpy.inf
        self.current_epoch = self.start_epoch

        self.get_grad_param()
        if self.verbose:
            self.grad_norm_func = theano.function(self.model.interface_layer.symbols(), self.grad_norm)
        self.set_name()
        logger.info("...Begin Building " + self.name + " Updating Function...")
        self.update_func = self.get_update_func()
        logger.info("...Finished, Update Function Saved to " + os.path.abspath(self.save_path))

    def set_name(self):
        self.name = "Optimizer-" + self.id

    def get_grad_param(self):
        self.grad_norm = TT.sqrt(sum(TT.sqr(g).sum() for g in self.model.grad)) / TT.cast(
            self.model.interface_layer.input.shape[1], 'float32')
        # self.has_numeric_error = TT.or_(TT.isnan(self.grad_norm), TT.isinf(self.grad_norm))
        # self.grad = [TT.switch(self.has_numeric_error, numpy_floatX(0.1) * p, g)
        # for g, p in zip(self.model.grad, self.model.param)]
        self.grad =[g / TT.cast(
            self.model.interface_layer.input.shape[1], 'float32') for g in self.model.grad]
        if self.clip_threshold is not None:
            self.grad = [TT.switch(TT.ge(self.grad_norm, self.clip_threshold),
                                   g * self.clip_threshold / self.grad_norm, g) for g in self.grad]

    def get_update_func(self):
        return lambda x: x

    def learning_param(self):
        return None

    def autosave(self, mode):
        if "interval" in mode:
            if 0 == (self.current_epoch + 1) % self.save_interval:
                save_path = self.save_path + "/" + self.model.name + "-epoch-" + str(self.current_epoch) + ".pkl"
                Model.save(self.model, save_path)
                logger.info("....Saving to " + os.path.abspath(save_path))
        if "best" in mode:
            if self.current_validation_error < self.best_validation_error:
                save_path = self.save_path + "/" + self.model.name + "-validation-best.pkl"
                Model.save(self.model, save_path)
                logger.info("....Saving to " + os.path.abspath(save_path))
        if "final" in mode:
            if self.current_epoch == self.max_epoch:
                save_path = self.save_path + "/" + self.model.name + "-epoch-" + str(self.current_epoch) + ".pkl"
                Model.save(self.model, save_path)
                logger.info("....Saving to " + os.path.abspath(save_path))

    def train(self):
        self.model.set_mode("train")
        no_better_validation_step = 0
        for i in range(self.start_epoch, self.start_epoch + self.max_epoch):
            start = time.time()
            self.current_epoch = i + 1
            self.train_data_iterator.begin(do_shuffle=True)
            logger.info("Epoch: " + str(self.current_epoch) + "/" + str(self.max_epoch))
            while True:
                if self.verbose:
                    quick_timed_log_eval(logger.debug, "    Gradient Norm:", self.grad_norm_func,
                                         *(self.train_data_iterator.input_batch() +
                                           self.train_data_iterator.output_batch()))
                minibatch_cost = quick_timed_log_eval(logger.debug, "Minibatch Cost:", self.update_func,
                                     *(self.train_data_iterator.input_batch() +
                                       self.train_data_iterator.output_batch() +
                                       self.learning_param()))
                self.train_data_iterator.next()
                if self.train_data_iterator.no_batch_left():
                    break
            # quick_timed_log_eval(logger.info, "Training Cost", self.model.get_cost, self.train_data_iterator)
            self.current_validation_error = quick_timed_log_eval(logger.info, "Validation Cost", self.model.get_cost,
                                                                 self.valid_data_iterator)
            if len(self.model.error_func_dict) > 0:
                quick_timed_log_eval(logger.info, "Validation Error List", self.model.get_error_dict,
                                     self.valid_data_iterator)
            self.autosave(self.autosave_mode)
            no_better_validation_step += 1
            if self.current_validation_error < self.best_validation_error:
                self.best_validation_error = self.current_validation_error
                no_better_validation_step = 0
            end = time.time()
            if no_better_validation_step >= self.max_epochs_no_best:
                break
            logger.info("Total Duration For Epoch " + str(self.current_epoch) + ":" + str(end - start))

    def _s(self, s):
        return '%s.%s' % (self.name, s)

    def print_stat(self):
        logger.info("Optimizer Name: " + self.name)
        logger.info("   Common Parameters: ")
        logger.info("      Max Epoch: " + str(self.max_epoch))
        logger.info("      Start Epoch: " + str(self.start_epoch))
        logger.info("      Autosave Mode: " + str(self.autosave_mode))
        logger.info("      Save Interval: " + str(self.save_interval))
        logger.info("      Max Epochs No Best: " + str(self.max_epochs_no_best))
