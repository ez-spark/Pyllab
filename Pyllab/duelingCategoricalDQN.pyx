cimport Pyllab
from Pyllab import *

cdef class duelingCategoricalDQN:
    cdef Pyllab.dueling_categorical_dqn* _dqn
    cdef Pyllab.dueling_categorical_dqn** _dqns
    cdef model shared
    cdef model v_hid
    cdef model v_lin
    cdef model a_hid
    cdef model a_lin
    cdef int threads
    cdef bint _is_multithread
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    cdef bint _is_from_char
    cdef int input_size
    cdef int action_size
    cdef int n_atoms
    cdef float v_min
    cdef float v_max
    cdef int reset_value
    '''
    @ filename := filename from which the model is loaded can be a .txt (setup file) or a .bin (an entire model with also weights and stuff
                  if the mod is true is a .bin basically, or it should be
    @ string := a char* type array, it is basically the same as filename, but all the file has been read into string
    @ bint mod:= is a flag, if true we must read from a .bin if filename is != None, if it is true and filename == None we just set some parameters and we don't have any model (is used by copy_model)
     all the other parameters from layers and going forward, are used if we pass the class cl, ,fcl, rl. Basically
     we are building the model from each class defined in python, we pass all the classes defined in python and et voilà we build in C the struct model
    '''
    def __cinit__(self,filename=None, dict d = None,bint mode = False, model shared=None, model v_hid=None, model v_lin=None, model a_hid = None, model a_lin = None,int input_size = 0, int action_size = 0, int n_atoms = 0, float v_min = 0, float v_max = 0, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(input_size)
        check_int(action_size)
        check_int(n_atoms)
        check_float(v_min)
        check_float(v_max)
        self.input_size = input_size
        self.action_size = action_size
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self._is_multithread = False
        self.threads = 1
        self._dqn = NULL
        self.reset_value = 1
        cdef char* ss
        # filename != none and mode = false, a txt configuration file is loaded
        # filename != none and mode = True a binary model file is loaded
        # if filename == none and d != none d is a dictionary got from a configuration file
        # if filename = none and dict = none but mod = true nothing is allocated only the flags are init
        if filename != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            
            if mode == False:
                if not dict_to_pass_to_dueling_categorical_dqn_is_good(get_dict_from_dueling_categorical_dqn_setup_file(filename)):
                    print("Error: your setup file is not correct")
                    exit(1)
                if not self._does_have_arrays:
                    ss = <char*>PyUnicode_AsUTF8(filename)
                    self._dqn = Pyllab.parse_dueling_categorical_dqn_without_arrays_file(ss)
                    
                elif not self._does_have_learning_parameters:
                    ss = <char*>PyUnicode_AsUTF8(filename)
                    self._dqn = Pyllab.parse_dueling_categorical_dqn_without_learning_parameters_file(ss)
                    
                else:
                    ss = <char*>PyUnicode_AsUTF8(filename)
                    self._dqn = Pyllab.parse_dueling_categorical_dqn_file(ss)
                    
            else:
                if not does_have_arrays or not does_have_learning_parameters:
                    print("Error: leave the default parameters for the flags!")
                    exit(1)
                ss = <char*>PyUnicode_AsUTF8(filename)
                self._dqn = Pyllab.load_dueling_categorical_dqn(ss)
                
                
            if self._dqn is NULL:
                raise MemoryError()
        
        elif d != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            s = from_dict_to_str_dueling_categorical_dqn(d)
            if s == None:
                print("Dict passed is not good, sorry")
                exit(1)
            if not self._does_have_arrays:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._dqn = Pyllab.parse_dueling_categorical_dqn_without_arrays_str(ss, len(s))
                
            elif not self._does_have_learning_parameters:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._dqn = Pyllab.parse_dueling_categorical_dqn_without_learning_parameters_str(ss, len(s))
            else:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._dqn = Pyllab.parse_dueling_categorical_dqn_str(ss, len(s))
            if self._dqn is NULL:
                raise MemoryError()
            
                
        elif mode == True:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
        
        else:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self.shared = shared
            self.v_hid = v_hid
            self.v_lin = v_lin
            self.a_hid = a_hid
            self.a_lin = a_lin
            self._dqn = Pyllab.dueling_categorical_dqn_init(input_size, action_size, n_atoms, v_min, v_max, shared._model, v_hid._model, a_hid._model, v_lin._model, a_lin._model)
            
            if self._dqn is NULL:
                raise MemoryError()
            if self._is_only_for_feedforward:
                self.make_it_only_for_ff()
        
    def __dealloc__(self):
        
        if self._dqn is not NULL:
            if self._does_have_arrays:
                if self._does_have_learning_parameters:
                    Pyllab.free_dueling_categorical_dqn(self._dqn)
                else:
                    Pyllab.free_dueling_categorical_dqn_without_learning_parameters(self._dqn)
            else:
                Pyllab.free_dueling_categorical_dqn_without_arrays(self._dqn)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_dueling_categorical_dqn_without_learning_parameters(self._dqns[i])
            free(self._dqns)
            
    def get_input_size(self):
        if self._dqn is NULL:
            return
        size = Pyllab.get_input_layer_size_dueling_categorical_dqn(self._dqn)
        return size
    
    def make_multi_thread(self, int threads):
        if self._dqn is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            if threads <= 1:
                print("Error: the number of threads must be >= 1")
                exit(1)
            if self._is_multithread:
                for i in range(self.threads):
                    Pyllab.free_dueling_categorical_dqn_without_learning_parameters(self._dqns[i])
                free(self._dqns)
            self._dqns = <Pyllab.dueling_categorical_dqn**>malloc(sizeof(Pyllab.dueling_categorical_dqn*)*threads)
            if self._dqns is NULL:
                raise MemoryError()
            for i in range(threads):
                self._dqns[i] = Pyllab.copy_dueling_categorical_dqn_without_learning_parameters(self._dqn)
            self._is_multithread = True
            self.threads = threads
        else:
            print("Error: you model is without learning parameters can't be multi thread!")
            exit(1)
        
    def make_single_thread(self):
        if self._dqn is NULL:
            return
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_dueling_categorical_dqn_without_learning_parameters(self._dqns[i])
            free(self._dqns)
        self._is_multithread = False
        self.threads = 1
    
    
    def save(self,int number_of_file, str directory):
        if self._dqn is NULL:
            return
        check_int(number_of_file)
        cdef char* ss
        if self._does_have_arrays and self._does_have_learning_parameters:
            ss = <char*>PyUnicode_AsUTF8(directory) 
            Pyllab.save_dueling_categorical_dqn_given_directory(self._dqn, number_of_file,ss)
    
    def get_size(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_dueling_categorical_dqn(self._dqn)
            else:
                return Pyllab.size_of_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        return 0
    
    def make_it_only_for_ff(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays:
            if not self._is_from_char and not self._is_only_for_feedforward:
                self.shared.make_it_only_for_ff()
                self.v_hid.make_it_only_for_ff()
                self.v_lin.make_it_only_for_ff()
                self.a_hid.make_it_only_for_ff()
                self.a_lin.make_it_only_for_ff()
                self._is_only_for_feedforward = True
            elif not self._is_only_for_feedforward:
                Pyllab.make_the_dueling_categorical_dqn_only_for_ff(self._dqn)
                self._is_only_for_feedforward = True
            
    def reset(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_dueling_categorical_dqn(self._dqn)
            else:
                Pyllab.reset_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        if self._is_multithread and self.reset_value == 1:
            self.reset_value = 0
            Pyllab.dueling_categorical_reset_without_learning_parameters_reset(self._dqns, self.threads)
    def reset_all(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_dueling_categorical_dqn(self._dqn)
            else:
                Pyllab.reset_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        if self._is_multithread:
            self.reset_value = 0
            Pyllab.dueling_categorical_reset_without_learning_parameters_reset(self._dqns, self.threads)
    def clip(self, float threshold):
        if self._dqn is NULL:
            return
        check_float(threshold)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.dueling_categorical_dqn_clipping_gradient(self._dqn, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
        if self._dqn is NULL:
            return
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_dueling_categorical_dqn(self._dqn,threshold,epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_dueling_categorical_dqn(self._dqn)
    
    def get_array_size_weights(self):
        if self._dqn is NULL:
            return
        return Pyllab.get_array_size_weights_dueling_categorical_dqn(self._dqn)
        
    def get_array_size_scores(self):
        if self._dqn is NULL:
            return
        return Pyllab.get_array_size_scores_dueling_categorical_dqn(self._dqn)
    
    def set_biases_to_zero(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_dueling_categorical_dqn_biases_to_zero(self._dqn)
    
    def set_unused_weights_to_zero(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_dueling_categorical_dqn_unused_weights_to_zero(self._dqn)    
        
    def reset_scores(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_score_dueling_categorical_dqn(self._dqn)
    def set_low_scores(self):
        if self._dqn is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_low_score_dueling_categorical_dqn(self._dqn)
    
    def get_number_of_weights(self):
        if self._dqn is NULL:
            return
        return Pyllab.count_weights_dueling_categorical_dqn(self._dqn)
    
    def compute_probability_distribution(self, inputs , int input_size):
        if self._dqn is NULL:
            return
        check_int(input_size)
        cdef float[:] i
        if self._does_have_arrays and self._does_have_learning_parameters:
            check_size(inputs,input_size)
            i = vector_is_valid(inputs)
            Pyllab.compute_probability_distribution(<float*>&i[0], input_size, self._dqn)
    
    def compute_q_function(self, inputs, int input_size):
        if self._dqn is NULL:
            return
        self.compute_probability_distribution(inputs,input_size)
        if self._does_have_arrays and self._does_have_learning_parameters:
            return from_float_to_ndarray(Pyllab.compute_q_functions(self._dqn),self.action_size)
        return None
    def get_best_action(self,inputs,int input_size):
        if self._dqn is NULL:
            return
        arr = self.compute_q_function(inputs,input_size)
        try:
            m = arr[0]
            index = 0
            for i in range(1,len(arr)):
                if arr[i] > m:
                    m = arr[i]
                    index = i
            return index
        except:
            return None
    
    def set_training_edge_popup(self, float k_percentage):
        if self._dqn is NULL:
            return
        check_float(k_percentage)
        if k_percentage > 1 or k_percentage <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_dueling_categorical_dqn_training_edge_popup(self._dqn,k_percentage)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_dueling_categorical_dqn_training_edge_popup(self._dqns[i],k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        if self._dqn is NULL:
            return
        check_float(percentage)
        check_float(goodness)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reinitialize_weights_according_to_scores_dueling_categorical_dqn(self._dqn,percentage,goodness)
    
    def set_training_gd(self):
        if self._dqn is NULL:
            return
        Pyllab.set_dueling_categorical_dqn_training_gd(self._dqn)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_dueling_categorical_dqn_training_gd(self._dqns[i])
        
    def set_beta1(self, float beta):
        if self._dqn is NULL:
            return
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta(self._dqn, beta, self.get_beta2())
    
    def set_beta2(self, float beta):
        if self._dqn is NULL:
            return
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta(self._dqn, self.get_beta1(), beta)
        
    def set_beta3(self, float beta):
        if self._dqn is NULL:
            return
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta_adamod(self._dqn, beta)
    
    def get_beta1(self):
        if self._dqn is NULL:
            return
        b = Pyllab.get_beta1_from_dueling_categorical_dqn(self._dqn)
        return b
    
    def get_beta2(self):
        if self._dqn is NULL:
            return
        b = Pyllab.get_beta2_from_dueling_categorical_dqn(self._dqn)
        return b
        
    def get_beta3(self):
        if self._dqn is NULL:
            return
        b = Pyllab.get_beta3_from_dueling_categorical_dqn(self._dqn)
        return b
    
    def set_ith_layer_training_mode_shared(self, int ith, int training_flag):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_shared(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_v_hid(self, int ith, int training_flag):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_v_hid(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_v_lin(self, int ith, int training_flag):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_v_lin(self._dqn,ith,training_flag)
        
    def set_ith_layer_training_mode_a_hid(self, int ith, int training_flag):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_a_hid(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_a_lin(self, int ith, int training_flag):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_a_lin(self._dqn,ith,training_flag)
    
    def set_k_percentage_of_ith_layer_shared(self,int ith, float k):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_shared(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_v_hid(self,int ith, float k):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_hid(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_v_lin(self,int ith, float k):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_lin(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_a_hid(self,int ith, float k):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_hid(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_a_lin(self,int ith, float k):
        if self._dqn is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_lin(self._dqn, ith, k)
        
    

    def train(self, states_t, states_t_1, rewards_t, actions_t, nonterminals_t_1, duelingCategoricalDQN dqn, float lambda_value):
        if self._dqn is NULL:
            return
        check_float(lambda_value)
        if dqn.threads != self.threads:
            return
        if not self._is_multithread:
            return
        if(get_first_dimension_size(states_t) != get_first_dimension_size(states_t_1) != self.get_input_size()):
            print("Error: the state dimensions doesn't match")
            exit(1)
        actions = vector_is_valid_int(actions_t)
        
        if actions.ndim != 1:
            print("Error: your dimension for actions must be 1")
            exit(1)
        check_size(actions, get_first_dimension_size(states_t))
        nonterminals = vector_is_valid_int(nonterminals_t_1)
        
        if nonterminals.ndim != 1:
            print("Error: your dimension for actions must be 1")
            exit(1)
        check_size(nonterminals, get_first_dimension_size(states_t))
        rewards = vector_is_valid(rewards_t)
        
        if rewards.ndim != 1:
            print("Error: your dimension for actions must be 1")
            exit(1)
        check_size(rewards, get_first_dimension_size(states_t))
        cdef int batch_size = self.threads
        if get_first_dimension_size(states_t) < batch_size:
            batch_size = get_first_dimension_size(states_t)
        value_s_t = None
        value_s_t_1 = None
        value_a_t = None
        value_r_t = None
        value_n_t_1 = None
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] s_t_buff
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] s_t_1_buff
        cdef npc.ndarray[npc.npy_int, ndim=1, mode = 'c'] a_t_buff
        cdef npc.ndarray[npc.npy_int, ndim=1, mode = 'c'] n_t_1_buff
        cdef npc.ndarray[npc.npy_float32, ndim=1, mode = 'c'] r_t_buff
        for j in range(batch_size):
            check_size(states_t[j], Pyllab.get_input_layer_size_dueling_categorical_dqn(self._dqn))
            check_size(states_t[j], Pyllab.get_input_layer_size_dueling_categorical_dqn(dqn._dqn))
            check_size(states_t_1[j], Pyllab.get_input_layer_size_dueling_categorical_dqn(self._dqn))
            check_size(states_t_1[j], Pyllab.get_input_layer_size_dueling_categorical_dqn(dqn._dqn))
            if j == 0:
                value_s_t = np.array([vector_is_valid(states_t[j])])
                value_s_t_1 = np.array([vector_is_valid(states_t_1[j])])
            else:
                value_s_t = np.append(value_s_t, np.array([vector_is_valid(states_t[j])]),axis=0)
                value_s_t_1 = np.append(value_s_t_1, np.array([vector_is_valid(states_t_1[j])]),axis=0)
        s_t_buff = np.ascontiguousarray(value_s_t,dtype=np.float32)
        s_t_1_buff = np.ascontiguousarray(value_s_t_1,dtype=np.float32)
        cdef float** s_t = <float**>malloc(sizeof(float*)*batch_size)
        cdef float** s_t_1 = <float**>malloc(sizeof(float*)*batch_size)
        for j in range(batch_size):
            s_t[j] = &s_t_buff[j, 0]
            s_t_1[j] = &s_t_1_buff[j, 0]
        
        
        a_t_buff = np.ascontiguousarray(actions,dtype=np.intc)
        n_t_1_buff = np.ascontiguousarray(nonterminals,dtype=np.intc)
        r_t_buff = np.ascontiguousarray(rewards,dtype=np.float32)
        
        cdef int* a_t = <int*>&a_t_buff[0]
        cdef int* n_t_1 = <int*>&n_t_1_buff[0]
        cdef float* r_t = <float*>&r_t_buff[0]
        Pyllab.dueling_categorical_dqn_train(batch_size, self._dqn,dqn._dqn, self._dqns, dqn._dqns, &s_t[0], &r_t[0], &a_t[0], &s_t_1[0], &n_t_1[0], lambda_value, Pyllab.get_input_layer_size_dueling_categorical_dqn(self._dqn))
        self.reset_value = 1
        free(s_t)
        free(s_t_1)
        return None
    
    
    def sum_dueling_categorical_dqn_partial_derivatives(self):
        if self._dqn is NULL:
            return
        if self._is_multithread:
            Pyllab.sum_dueling_categorical_dqn_partial_derivatives_multithread(self._dqns, self._dqn, self.threads, 0)
            
def paste_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2):
    if dqn1._dqn is NULL or dqn2._dqn is NULL:
        return
    if dqn1._does_have_arrays and dqn1._does_have_learning_parameters and dqn2._does_have_arrays and dqn2._does_have_learning_parameters and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.paste_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn)
    
def paste_dueling_categorical_dqn_without_learning_parameters(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2):
    if dqn1._dqn is NULL or dqn2._dqn is NULL:
        return
    if dqn1._does_have_arrays and dqn2._does_have_arrays and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.paste_dueling_categorical_dqn_without_learning_parameters(dqn1._dqn,dqn2._dqn)

def copy_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    if dqn._dqn is NULL:
        return
    cdef Pyllab.dueling_categorical_dqn* copy
    if dqn._does_have_learning_parameters and dqn._does_have_arrays and not dqn._is_only_for_feedforward:
        if not dqn._is_from_char:
            shared = copy_model(dqn.shared)
            v_hid = copy_model(dqn.v_hid)
            v_lin = copy_model(dqn.v_lin)
            a_hid = copy_model(dqn.a_hid)
            a_lin = copy_model(dqn.a_lin)
            return duelingCategoricalDQN(shared = shared, v_hid = v_hid, v_lin = v_lin , a_hid = a_hid, a_lin = a_lin, input_size = dqn.input_size,action_size = dqn.action_size,n_atoms = dqn.n_atoms,v_min = dqn.v_min,v_max = dqn.v_max)
        copy = Pyllab.copy_dueling_categorical_dqn(dqn._dqn)
        mod = duelingCategoricalDQN(mode = True)
        mod._dqn = copy
        return mod
    
def copy_dueling_categorical_dqn_without_learning_parameters(duelingCategoricalDQN dqn):
    if dqn._dqn is NULL:
        return
    cdef Pyllab.dueling_categorical_dqn* copy 
    if dqn._does_have_learning_parameters and dqn._does_have_arrays and not dqn._is_only_for_feedforward:
        if not dqn._is_from_char:
            shared = copy_model_without_learning_parameters(dqn.shared)
            v_hid = copy_model_without_learning_parameters(dqn.v_hid)
            v_lin = copy_model_without_learning_parameters(dqn.v_lin)
            a_hid = copy_model_without_learning_parameters(dqn.a_hid)
            a_lin = copy_model_without_learning_parameters(dqn.a_lin)
            return duelingCategoricalDQN(shared = shared, v_hid = v_hid, v_lin = v_lin , a_hid = a_hid, a_lin = a_lin, input_size = dqn.input_size,action_size = dqn.action_size,n_atoms = dqn.n_atoms,v_min = dqn.v_min,v_max = dqn.v_max,does_have_learning_parameters = False)
        copy = Pyllab.copy_dueling_categorical_dqn_without_learning_parameters(dqn._dqn)
        mod = duelingCategoricalDQN(mod = True,does_have_learning_parameters = False)
        mod._dqn = copy
        return mod
    

def slow_paste_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2, float tau):
    check_float(tau)
    if dqn1._dqn is NULL or dqn2._dqn is NULL:
        return
    if dqn1._does_have_arrays and dqn1._does_have_learning_parameters and dqn2._does_have_arrays and dqn2._does_have_learning_parameters and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.slow_paste_dueling_categorical_dqn(dqn1._dqn, dqn2._dqn,tau)

def copy_vector_to_params_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    if dqn._dqn is NULL:
        return
    check_size(vector,dqn.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_params_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    if dqn._dqn is NULL:
        return
    vector = np.arange(dqn.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    if dqn._dqn is NULL:
        return
    check_size(vector,dqn.get_array_size_weights())
    cdef float[:]v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_weights_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    if dqn._dqn is NULL:
        return
    vector = np.arange(dqn.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    if dqn._dqn is NULL:
        return
    check_size(vector,dqn.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_scores_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    if dqn._dqn is NULL:
        return
    vector = np.arange(dqn.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None
    
def compare_score_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2, duelingCategoricalDQN dqn_output):
    if dqn1._dqn is NULL or dqn2._dqn is NULL or dqn_output._dqn is NULL:
        return
    Pyllab.compare_score_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn,dqn_output._dqn)
    
def compare_score_dueling_categorical_dqn_with_vector(duelingCategoricalDQN dqn1, vector, duelingCategoricalDQN dqn_output):
    if dqn1._dqn is NULL or dqn2._dqn is NULL or dqn_output._dqn is NULL:
        return
    check_size(vector,dqn1.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    Pyllab.compare_score_dueling_categorical_dqn_with_vector(dqn1._dqn, <float*>&v[0],dqn_output._dqn)
    
def sum_score_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2, duelingCategoricalDQN dqn_output):
    if dqn1._dqn is NULL or dqn2._dqn is NULL or dqn_output._dqn is NULL:
        return
    Pyllab.sum_score_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn,dqn_output._dqn)
             
