cimport Pyllab
from Pyllab import *

cdef class rainbow:
    cdef Pyllab.rainbow* _r
    cdef duelingCategoricalDQN online_net
    cdef duelingCategoricalDQN target_net
    
    cdef float max_epsilon
    cdef float min_epsilon
    cdef float epsilon_decay
    cdef float epsilon
    cdef float alpha_priorization
    cdef float beta_priorization
    cdef float lambda_value
    cdef float tau_copying
    cdef float momentum
    cdef float gamma
    cdef float beta1
    cdef float beta2
    cdef float beta3
    cdef float k_percentage
    cdef float clipping_gradient_value
    cdef float adaptive_clipping_gradient_value
    cdef float diversity_driven_threshold
    cdef float lr
    cdef float lr_minimum
    cdef float lr_maximum
    cdef float initial_lr
    cdef float lr_decay
    cdef float beta_priorization_increase
    cdef float diversity_driven_decay
    cdef float diversity_driven_minimum
    cdef float diversity_driven_maximum
    cdef uint64_t lr_epoch_threshold
    cdef int lr_decay_flag
    cdef int feed_forward_flag
    cdef int training_mode
    cdef int adaptive_clipping_flag
    cdef int batch_size
    cdef int threads
    cdef int gd_flag
    cdef int sampling_flag
    cdef uint64_t max_buffer_size
    cdef uint64_t n_step_rewards
    cdef uint64_t stop_epsilon_greedy
    cdef uint64_t epochs_to_copy_target
    cdef uint64_t diversity_driven_q_functions
    
    def __cinit__(self,duelingCategoricalDQN online_net, duelingCategoricalDQN target_net, float beta_priorization_increase = 0.05, float max_epsilon = 1, float min_epsilon = 0.001,
                  float diversity_driven_decay = 0, float diversity_driven_minimum = 0.001, float diversity_driven_maximum = 1, float epsilon_decay = 0.05,float epsilon = 1, float alpha_priorization = 0.4,
                  float beta_priorization = 0.4, float tau_copying = 0.8, float momentum = 0.9, float gamma = 0.99, float beta1 = BETA1_ADAM, float beta2 = BETA2_ADAM, float beta3 = BETA3_ADAMOD,
                  float k_percentage = 1, float clipping_gradient_value = 1, float adaptive_clipping_gradient_value = 0.01, float diversity_driven_threshold = 0.05, float lr = 0.001, float lr_minimum = 0.0001, float lr_maximum = 0.1,
                  float lr_decay = 0.0001, int lr_epoch_threshold = 100, int lr_decay_flag = LR_NO_DECAY, int feed_forward_flag = FULLY_FEED_FORWARD, int training_mode = GRADIENT_DESCENT, int adaptive_clipping_flag = 1,
                  int batch_size = 32,int threads = 32, int gd_flag = ADAM, int max_buffer_size = 10000, int n_step_rewards = 3, int stop_epsilon_greedy = 100, int epochs_to_copy_target = 10,
                  int sampling_flag = REWARD_SAMPLING, int diversity_driven_q_functions = 100):
        cdef int th = threads
        if(th < 2):
            return
        try:
            if(target_net.threads != th):
                return
            
            if(online_net.threads != th):
                return
        except:
            return
        check_float(diversity_driven_minimum)
        check_float(diversity_driven_maximum)
        check_float(diversity_driven_decay)
        check_float(beta_priorization_increase)
        check_float(max_epsilon)    
        check_float(min_epsilon)    
        check_float(epsilon_decay)    
        check_float(epsilon)    
        check_float(alpha_priorization)    
        check_float(beta_priorization)  
        check_float(tau_copying)    
        check_float(momentum)    
        check_float(gamma)    
        check_float(beta1)    
        check_float(beta2)    
        check_float(beta3)    
        check_float(k_percentage)    
        check_float(clipping_gradient_value)    
        check_float(adaptive_clipping_gradient_value)    
        check_float(diversity_driven_threshold)    
        check_float(lr)    
        check_float(lr_minimum)    
        check_float(lr_maximum)    
        check_float(lr_decay)
        check_int(lr_epoch_threshold)    
        check_int(lr_decay_flag)    
        check_int(feed_forward_flag)    
        check_int(training_mode)    
        check_int(batch_size)    
        check_int(threads)    
        check_int(sampling_flag)    
        check_int(gd_flag)    
        check_int(max_buffer_size)    
        check_int(n_step_rewards)    
        check_int(stop_epsilon_greedy)    
        check_int(adaptive_clipping_flag)    
        check_int(epochs_to_copy_target)    
        check_int(diversity_driven_q_functions)    
        
        self.online_net = online_net
        self.target_net = target_net
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.alpha_priorization = alpha_priorization
        self.beta_priorization = beta_priorization
        self.lambda_value = gamma
        self.tau_copying = tau_copying
        self.momentum = momentum
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.k_percentage = k_percentage
        self.clipping_gradient_value = clipping_gradient_value
        self.adaptive_clipping_gradient_value = adaptive_clipping_gradient_value
        self.diversity_driven_threshold = diversity_driven_threshold
        self.lr = lr
        self.lr_minimum = lr_minimum
        self.lr_maximum = lr_maximum
        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.lr_epoch_threshold = lr_epoch_threshold
        self.lr_decay_flag = lr_decay_flag
        self.feed_forward_flag = feed_forward_flag
        self.training_mode = training_mode
        self.adaptive_clipping_flag = adaptive_clipping_flag
        self.batch_size = batch_size
        self.threads = th
        self.gd_flag = gd_flag
        self.max_buffer_size = max_buffer_size
        self.n_step_rewards = n_step_rewards
        self.stop_epsilon_greedy = stop_epsilon_greedy
        self.epochs_to_copy_target = epochs_to_copy_target
        self.diversity_driven_q_functions = diversity_driven_q_functions
        self.diversity_driven_decay = diversity_driven_decay
        self.diversity_driven_maximum = diversity_driven_maximum
        self.diversity_driven_minimum = diversity_driven_minimum
        self.beta_priorization_increase = beta_priorization_increase
        self.sampling_flag = sampling_flag
        
        self._r = Pyllab.init_rainbow(self.sampling_flag, self.gd_flag,self.lr_decay_flag,self.feed_forward_flag,self.training_mode,1,self.adaptive_clipping_flag,self.batch_size,self.threads, 
                      self.diversity_driven_q_functions,self.epochs_to_copy_target,self.max_buffer_size,self.n_step_rewards,self.stop_epsilon_greedy,0,self.lr_epoch_threshold,
                      self.max_epsilon,self.min_epsilon,self.epsilon_decay,self.epsilon,self.alpha_priorization,self.beta_priorization,self.lambda_value,self.gamma,self.tau_copying,self.beta1,self.beta2,
                      self.beta3,self.k_percentage,self.clipping_gradient_value,self.adaptive_clipping_gradient_value,self.lr,self.lr_minimum,self.lr_maximum,self.lr_decay,self.momentum,
                      self.diversity_driven_threshold,self.diversity_driven_decay,self.diversity_driven_minimum,self.diversity_driven_maximum,self.beta_priorization_increase, self.online_net._dqn, self.target_net._dqn, self.online_net._dqns, self.target_net._dqns)
        
        if self._r is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        Pyllab.free_rainbow(self._r)
    
    
    def get_action(self, inputs):
        check_size(inputs,self.online_net.get_input_size())
        cdef float[:] i = vector_is_valid(inputs)
        cdef float* dynamic_array = <float*> malloc(self.online_net.get_input_size() * sizeof(float))
        cdef int j
        for j in range(self.online_net.get_input_size()):
            dynamic_array[j] = i[j]
        cdef int action = Pyllab.get_action_rainbow(self._r,<float*>&dynamic_array[0], self.online_net.get_input_size(), 1)
        return action
    
    def add_experience(self,state_t,state_t_1,int action, float reward, bint done, bint train_flag = False):
        check_size(state_t,self.online_net.get_input_size())
        check_size(state_t_1,self.online_net.get_input_size())
        check_int(action)
        check_float(reward)
        if action >= self.online_net.action_size:
            return
        cdef float[:] i = vector_is_valid(state_t)
        cdef float[:] k = vector_is_valid(state_t_1)
        cdef float* dynamic_array_t = <float*> malloc(self.online_net.get_input_size() * sizeof(float))
        cdef float* dynamic_array_t_1 = <float*> malloc(self.online_net.get_input_size() * sizeof(float))
        cdef int j
        for j in range(self.online_net.get_input_size()):
            dynamic_array_t[j] = i[j]
            dynamic_array_t_1[j] = k[j]
        cdef int nonterminal = 1
        if done:
            nonterminal = 0
        Pyllab.add_experience(<Pyllab.rainbow*>self._r, <float*>&dynamic_array_t[0], <float*>&dynamic_array_t_1[0], action,reward, nonterminal)
        if done or train_flag:
            self.train()
    def train(self):
        Pyllab.train_rainbow(<Pyllab.rainbow*>self._r, 1)
