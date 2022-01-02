cimport Pyllab
from Pyllab import *

cdef class training:
    cdef float lr
    cdef float start_lr
    cdef float momentum
    cdef int batch_size
    cdef int gradient_descent_flag
    cdef float current_beta1
    cdef float current_beta2
    cdef int regularization
    cdef uint64_t total_number_weights
    cdef float lambda
    cdef unsigned long long int t
    cdef float start_beta1
    cdef float start_beta2
    cdef int lr_decay_flag
    cdef float lr_minimum
    cdef float lr_maximum
    cdef int timestep_threshold
    cdef float decay
    def __cinit__(self, float lr = 0.1, float momentum = 0.9, int batch_size = 0, int gradient_descent_flag = NESTEROV, float current_beta1 = BETA1_ADAM, float current_beta2 = BETA2_ADAM, int regularization = NO_REGULARIZATION, uint64_t total_number_weights = 0, float lambda = 0, int lr_decay_flag = LR_NO_DECAY, int timestep_threshold = 0, float lr_minimum = 0, float lr_maximum = 1, float decay = 0):
        check_float(lr)
        check_float(decay)
        check_float(momentum)
        check_float(lambda)
        check_float(current_beta1)
        check_float(current_beta2)
        check_float(lr_minimum)
        check_float(lr_maximum)
        check_int(batch_size)
        check_int(lr_decay_flag)
        check_int(timestep_threshold)
        check_int(gradient_descent_flag)
        check_int(regularization)
        if(total_number_weights < 0)
            print("total_number weights can't be < 0")
            exit(1)
        
        self.decay = decay
        self.lr_minimum = lr_minimum
        self.lr_maximum = lr_maximum
        self.timestep_threshold = timestep_threshold
        self.lr = lr
        self.lr_decay_flag = lr_decay_flag
        self.start_lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.gradient_descent_flag = gradient_descent_flag
        self.start_beta1 = current_beta1
        self.current_beta1 = current_beta1
        self.current_beta2 = current_beta2
        self.start_beta2 = current_beta2
        self.regularization = regularization
        self.total_number_weights = total_number_weights
        self.lambda = lambda
        
    def set_lr(self, float lr):
        check_float(lr)
        if lr > 1 or lr < -1
            print("Error lr must be in [-1,1]")
            exit(1)
        self.lr = lr
    def set_momentum(self, float momentum):
        check_float(momentum)
        if momentum > 1 or momentum < 0:
            print("Error: your momentum mmust be in [0,1]"
            exit(1)
        self.momentum = momentum
    def set_batch_size(self, ModelBatch m):
        self.batch_size = len(m.models)
    
    def set_gradient_descent(self, int flag):
        check_int(flag)
        if flag != NESTEROV and flag != ADAMOD and flag != ADAM and flag != DIFF_GRAD and flag != RADAM:
            print("Error: flag not recognized")
            exit(1)
        self.gradient_descent_flag = flag
    
    def set_current_beta1(self, float beta):
        check_float(beta)
        if beta > 1 or beta < 0:
            print("Error: beta must be in [0,1]")
            exit(1)
        self.current_beta1 = beta
    
    def set_current_beta2(self, float beta):
        check_float(beta)
        if beta > 1 or beta < 0:
            print("Error: beta must be in [0,1]")
            exit(1)
        self.current_beta2 = beta
    
    def set_regularization(self, int regularization):
        check_int(regulazitazion)
        if regularization != NO_REGULARIZATION and regularization != L2_REGULARIZATION:
            print("Error: either no regularization or l2 regularization must be applied")
            exit(1)
    
    def set_weights_number(self, Model m):
        self.total_number_weights = m.get_number_of_weights()
        
    def set_lambda(self, float lambda):
        check_float(lambda)
        if lambda < 0 or lambda > 1:
            print("Error: lambda must be in [0,1]")
            exit(1)
        self.lambda = lambda
    
    def set_lr_decay(self,int lr_decay_flag, float lr_minimum, float lr_maximum, int timestep_threshold, float decay):
        check_int(lr_decay_flag) 
        check_int(timestep_threshold) 
        check_float(lr_minimum) 
        check_float(decay) 
        check_float(lr_maximum)
        if lr_decay_flag != LR_ANNEALING_DECAY and lr_decay_flag != LR_NO_DECAY and lr_decay_flag != LR_CONSTANT_DECAY and lr_decay_flag != LR_STEP_DECAY and lr_decay_flag != LR_TIME_BASED_DECAY:
            print("Error: no decay flag recognized")
            exit(1)
        if lr_minimum > 1 or lr_minimum < -1 or lr_maximum > 1 or lr_maximum < -1:
            print("Error: lr_minimum or lr_maximum not correct")
            exit(1)
        if decay < 0 or decay > 1:
            print("Error: decay must be set in [0,1]")
            exit(1)
        self.lr_decay_flag = lr_decay_flag
        self.lr_minimum = lr_minimum
        self.lr_maximum = lr_maximum
        self.timestep_threshold = timestep_threshold
        self.decay = decay
    def update_model(self, Model m):
        Pyllab.update_model(m._model, self.lr, self.momentum,self.batch_size,self.gradient_descent_flag,&self.current_beta1,&self.current_beta2,self.regularization, self.total_number_weights,self.lambda,&self.t)
    
    def update_parameters(self):
        Pyllab.update_training_parameters(&self.current_beta1, &self.current_beta2, &self.t, self.start_beta1, self.start_beta2)
        Pyllab.update_lr(&self.lr, self.lr_minimum, self.lr_maximum,self.start_lr, self.decay, <int>self.t, self.timestep_threshold, self.lr_decay_flag)
    
        
