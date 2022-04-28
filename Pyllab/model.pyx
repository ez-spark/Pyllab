cimport Pyllab
from Pyllab import *

cdef class model:
    cdef Pyllab.model* _model
    cdef Pyllab.model** _models
    cdef fcls
    cdef cls
    cdef rls
    cdef Pyllab.fcl** _fcls
    cdef Pyllab.cl** _cls
    cdef Pyllab.rl** _rls
    cdef int _n_fcl
    cdef int _n_cl
    cdef int _n_rl
    cdef int threads
    cdef bint _is_multithread
    cdef int _layers
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    cdef bint _is_from_char
    '''
    @ filename := filename from which the model is loaded can be a .txt (setup file) or a .bin (an entire model with also weights and stuff
                  if the mod is true is a .bin basically, or it should be
    @ string := a char* type array, it is basically the same as filename, but all the file has been read into string
    @ bint mod:= is a flag, if true we must read from a .bin if filename is != None, if it is true and filename == None we just set some parameters and we don't have any model (is used by copy_model)
     all the other parameters from layers and going forward, are used if we pass the class cl, ,fcl, rl. Basically
     we are building the model from each class defined in python, we pass all the classes defined in python and et voilà we build in C the struct model
    '''
    def __cinit__(self,filename=None, dict d = None, bint mod = False, int layers=0, int n_fcl=0, int n_cl=0, int n_rl=0, list fcls=None, list cls=None, list rls=None, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(layers)
        check_int(n_fcl)
        check_int(n_cl)
        check_int(n_rl)
        self._is_multithread = False
        self.threads = 1
        self._models = NULL
        cdef char* ss
        # filename != none and mod = true, a txt configuration file is loaded
        # filename != none and mod = false a binary model file is loaded
        # if filename == none and d != none d is a dictionary got from a configuration file
        # if filename = none and dict = none but mod = true nothing is allocated only the flags are init
        if filename != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            
            
            if mod == False:
                if not dict_to_pass_to_model_is_good(get_dict_from_model_setup_file(filename)):
                    print("Error: your setup file is not correct")
                    exit(1)
                ss = <char*>PyUnicode_AsUTF8(filename)
                if not self._does_have_arrays:
                    self._model = Pyllab.parse_model_without_arrays_file(ss)
                    
                elif not self._does_have_learning_parameters:
                    self._model = Pyllab.parse_model_without_learning_parameters_file(ss)
                    
                else:
                    self._model = Pyllab.parse_model_file(ss)
                    
            else:
                if not does_have_arrays or not does_have_learning_parameters:
                    print("Error: leave the default parameters for the flags!")
                    exit(1)
                self._model = Pyllab.load_model(ss)
            if self._model is NULL:
                raise MemoryError()
            if self._is_only_for_feedforward:
                self._is_only_for_feedforward = False
                self.make_it_only_for_ff()
                
                
            if self._model is NULL:
                raise MemoryError()
        
        elif d != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            s = from_dict_to_str_model(d)
            if s == None:
                print("Dict passed is not good, sorry")
                exit(1)
            
            ss = <char*>PyUnicode_AsUTF8(s)
            if not self._does_have_arrays:
                self._model = Pyllab.parse_model_without_arrays_str(ss,len(s))
                
            elif not self._does_have_learning_parameters:
                self._model = Pyllab.parse_model_without_learning_parameters_str(ss,len(s))
            else:
                self._model = Pyllab.parse_model_str(ss,len(s))
            if self._model is NULL:
                raise MemoryError()
            if self._is_only_for_feedforward:
                self._is_only_for_feedforward = False
                self.make_it_only_for_ff()
            
                
        elif mod == True:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
        
        else:
            try:
                self._layers = layers
                self._n_fcl = n_fcl
                self._n_cl = n_cl
                self._n_rl = n_rl
                self._fcls = NULL
                self._cls = NULL
                self._rls = NULL
                self.fcls = fcls
                self.cls = cls
                self.rls = rls
                self._does_have_arrays = does_have_arrays
                self._does_have_learning_parameters = does_have_learning_parameters
                self._is_only_for_feedforward = is_only_for_feedforward
                self._is_from_char = False
                if n_fcl != len(fcls):
                    print("Your number of fully connected (n_fcl) does not match the list size of fcls")
                    return
                if n_cl != len(cls):
                    print("Your number of convolutional (n_cl) does not match the list size of cls")
                    return
                if n_rl != len(rls):
                    print("Your number of residual (n_rl) does not match the list size of rls")
                    return
            except:
                print("Error: an error occurred when initializing the model")
                exit(1)
            
            
            if n_fcl > 0:
                self._fcls = <Pyllab.fcl**>malloc(n_fcl*sizeof(Pyllab.fcl*))
                for i in range(0,n_fcl):
                    self._fcls[i] = <Pyllab.fcl*>fcls[i]._fcl
            if n_cl > 0:
                self._cls = <Pyllab.cl**>malloc(n_cl*sizeof(Pyllab.cl*))
                for i in range(0,n_cl):
                    self._cls[i] = <Pyllab.cl*>cls[i]._cl
            if n_rl > 0:
                self._rls = <Pyllab.rl**>malloc(n_rl*sizeof(Pyllab.rl*))
                for i in range(0,n_rl):
                    self._rls[i] = <Pyllab.rl*>rls[i]._rl
            
            self._model = Pyllab.network(layers,n_rl,n_cl, n_fcl,self._rls,self._cls,self._fcls)
            
            if self._model is NULL:
                raise MemoryError()
            if self._is_only_for_feedforward:
                self._is_only_for_feedforward = False
                self.make_it_only_for_ff()
        
    def __dealloc__(self):
        
        if not (self._model is NULL):
            if self._does_have_arrays:
                if self._does_have_learning_parameters:
                    Pyllab.free_model(self._model)
                else:
                    Pyllab.free_model_without_learning_parameters(self._model)
            else:
                Pyllab.free_model_without_arrays(self._model)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_model_without_learning_parameters(self._models[i])
            free(self._models)
    
    def make_multi_thread(self, int threads):
        check_int(threads)
        if not self._models is NULL and self._does_have_arrays and self._does_have_learning_parameters:
            if threads < 1:
                print("Error: the number of threads must be >= 1")
                exit(1)
            if self._is_multithread:
                for i in range(self.threads):
                    Pyllab.free_model_without_learning_parameters(self._models[i])
                free(self._models)
            self._models = <Pyllab.model**>malloc(sizeof(Pyllab.model*)*threads)
            if self._models is NULL:
                raise MemoryError()
            for i in range(threads):
                self._models[i] = Pyllab.copy_model_without_learning_parameters(self._model)
            self._is_multithread = True
            self.threads = threads
        else:
            print("Error: you model is without learning parameters can't be multi thread!")
            exit(1)
        
    def make_single_thread(self):
        if self._is_multithread and not (self._model is NULL):
            for i in range(self.threads):
                Pyllab.free_model_without_learning_parameters(self._models[i])
            free(self._models)
        self._is_multithread = False
        self.threads = 1
    
    
    def save(self,number_of_file):
        if not (self._model is NULL) and self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_model(self._model, number_of_file)
    
    def get_size(self):
        if not (self._model is NULL) and self._does_have_arrays:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_model(self._model)
            else:
                return Pyllab.size_of_model_without_learning_parameters(self._model)
        return 0
    
    def make_it_only_for_ff(self):
        if not (self._model is NULL) and self._does_have_arrays:
            if not self._is_from_char and not self._is_only_for_feedforward:
                for i in range(0,self._n_fcl):
                    self.fcls[i].make_it_only_for_ff()
                for i in range(0,self._n_cl):
                    self.cls[i].make_it_only_for_ff()
                for i in range(0,self._n_rl):
                    self.rls[i].make_it_only_for_ff()
                self._is_only_for_feedforward = True
            elif not self._is_only_for_feedforward:
                Pyllab.make_the_model_only_for_ff(self._model)
                self._is_only_for_feedforward = True
            
    def reset(self):
        if not (self._model is NULL):
            if not self._is_from_char:
                for i in range(0,self._n_fcl):
                    self.fcls[i].reset()
                for i in range(0,self._n_cl):
                    self.cls[i].reset()
                for i in range(0,self._n_rl):
                    self.rls[i].reset()
            else:
                if self._does_have_arrays:
                    if self._does_have_learning_parameters:
                        Pyllab.reset_model(self._model)
                else:
                    Pyllab.reset_model_without_learning_parameters(self._model)
            if self._is_multithread:
                for i in range(self.threads):
                    Pyllab.reset_model_without_learning_parameters(self._models[i])
    def clip(self, float threshold):
        check_float(threshold)
        if not (self._model is NULL) and self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.clipping_gradient(self._model, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if not (self._model is NULL) and self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_model(self._model,threshold,epsilon)

    def get_array_size_params(self):
        if not (self._model is NULL):
            return Pyllab.get_array_size_params_model(self._model)
    
    def get_array_size_weights(self):
        if not (self._model is NULL):
            return Pyllab.get_array_size_weights_model(self._model)
        
    def get_array_size_scores(self):
        if not (self._model is NULL):
            return Pyllab.get_array_size_scores_model(self._model)
    
    def set_biases_to_zero(self):
        if not (self._model is NULL):
            if self._does_have_arrays and self._does_have_learning_parameters:
                Pyllab.set_model_biases_to_zero(self._model)
    
    def set_unused_weights_to_zero(self):
        if not (self._model is NULL):
            if self._does_have_arrays and self._does_have_learning_parameters:
                Pyllab.set_model_unused_weights_to_zero(self._model)    
        
    def reset_scores(self):
        if not (self._model is NULL):
            if self._does_have_arrays and self._does_have_learning_parameters:
                Pyllab.reset_score_model(self._model)
    def set_low_scores(self):
        if not (self._model is NULL):
            if self._does_have_arrays and self._does_have_learning_parameters:
                Pyllab.set_low_score_model(self._model)
    
    def get_number_of_weights(self):
        if not (self._model is NULL):
            return Pyllab.count_weights(self._model)
    
    def set_model_error(self, int error_flag, int output_dimension, float threshold1 = 0, float threshold2 = 0, float gamma = 0, alpha = None):
        if self._model is NULL:
            return
        check_int(error_flag)
        check_int(output_dimension)
        check_float(threshold1)
        check_float(threshold2)
        check_float(gamma)
        cdef float[:] a
        if alpha == None:
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,NULL,output_dimension)
            if self._is_multithread:
                for i in range(self.threads):
                    Pyllab.set_model_error(self._models[i],error_flag,threshold1,threshold2,gamma,NULL,output_dimension)
        else:
            check_size(alpha,output_dimension)
            a = vector_is_valid(alpha)
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,<float*>&a[0],output_dimension)
            if self._is_multithread:
                for i in range(self.threads):
                    Pyllab.set_model_error(self._models[i],error_flag,threshold1,threshold2,gamma,<float*>&a[0],output_dimension)
        
    
    def feed_forward(self, int tensor_depth, int tensor_i, int tensor_j, inputs):
        if self._model is NULL:
            return
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(tensor_j*tensor_i*tensor_depth)
        cdef float[:] i
        if self._does_have_arrays and self._does_have_learning_parameters:
            check_size(inputs,tensor_depth*tensor_i*tensor_j)
            i = vector_is_valid(inputs)
            Pyllab.model_tensor_input_ff(self._model,tensor_depth,tensor_i,tensor_j,<float*>&i[0])
    
    def back_propagation(self, int tensor_depth, int tensor_i, int tensor_j, inputs, error, int error_dimension):
        if self._model is NULL:
            return
        cdef float[:] i
        cdef float[:] e
        cdef float* ret
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(tensor_j*tensor_i*tensor_depth)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            check_size(inputs,tensor_depth*tensor_i*tensor_j)
            check_size(error,error_dimension)
            if error_dimension != self.get_output_dimension_from_model():
                print("Error: your error dimension doesn't match the output dimension of the model")
                exit(1)
            i = vector_is_valid(inputs)
            e = vector_is_valid(error)
            ret = Pyllab.model_tensor_input_bp(self._model,tensor_depth,tensor_i,tensor_j,<float*>&i[0],<float*>&e[0],error_dimension)
            return <float[:tensor_depth*tensor_i*tensor_j]> ret
        
    def ff_error_bp(self, int tensor_depth, int tensor_i, int tensor_j, inputs, outputs):
        if self._model is NULL:
            return
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(tensor_j*tensor_i*tensor_depth)
        cdef float[:] i
        cdef float[:] o
        cdef float* ret
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            check_size(inputs,tensor_depth*tensor_i*tensor_j)
            check_size(outputs,self.get_output_dimension_from_model())
            i = vector_is_valid(inputs)
            o = vector_is_valid(outputs)
            ret = Pyllab.ff_error_bp_model_once(self._model,tensor_depth,tensor_i,tensor_j,<float*>&i[0],<float*>&o[0])
            return <float[:tensor_depth*tensor_i*tensor_j]> ret
        
    
    def set_training_edge_popup(self, float k_percentage):
        if self._model is NULL:
            return
        check_float(k_percentage)
        if k_percentage > 1 or k_percentage <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_model_training_edge_popup(self._model,k_percentage)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_model_training_edge_popup(self._models[i],k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        if self._model is NULL:
            return
        check_float(percentage)
        check_float(goodness)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reinitialize_weights_according_to_scores_model(self._model,percentage,goodness)
    
    def set_training_gd(self):
        if self._model is NULL:
            return
        Pyllab.set_model_training_gd(self._model)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_model_training_gd(self._models[i])
    
    def reset_edge_popup_d_params(self):
        if self._model is NULL:
            return
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_edge_popup_d_model(self._model)
    
    def output_layer(self):
        if self._model is NULL:
            return
        if self._does_have_arrays:
            return from_float_to_ndarray(Pyllab.get_output_layer_from_model(self._model),Pyllab.get_output_dimension_from_model(self._model))
    
    def get_output_dimension_from_model(self):
        if self._model is NULL:
            return
        n = Pyllab.get_output_dimension_from_model(self._model)
        return n
        
    def set_beta1(self, float beta):
        if self._model is NULL:
            return
        check_float(beta)
        Pyllab.set_model_beta(self._model, beta, self.get_beta2())
    
    def set_beta2(self, float beta):
        if self._model is NULL:
            return
        check_float(beta)
        Pyllab.set_model_beta(self._model, self.get_beta1(), beta)
        
    def set_beta3(self, float beta):
        if self._model is NULL:
            return
        check_float(beta)
        Pyllab.set_model_beta_adamod(self._model, beta)
    
    def get_beta1(self):
        if self._model is NULL:
            return
        b = Pyllab.get_beta1_from_model(self._model)
        return b
    
    def get_beta2(self):
        if self._model is NULL:
            return
        b = Pyllab.get_beta2_from_model(self._model)
        return b
        
    def get_beta3(self):
        if self._model is NULL:
            return
        b = Pyllab.get_beta3_from_model(self._model)
        return b
    
    def set_ith_layer_training_mode(self, int ith, int training_flag):
        if self._model is NULL:
            return
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_model(self._model,ith,training_flag)
    
    def set_k_percentage_of_ith_layer(self,int ith, float k):
        if self._model is NULL:
            return
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_model(self._model, ith, k)

    def feed_forward_opt_multi_thread(self, int tensor_depth, int tensor_i, int tensor_j, inputs):
        if self._model is NULL:
            return
        if not self._is_multithread:
            return
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(tensor_j*tensor_i*tensor_depth)
        value = None
        cdef int batch_size = self.threads
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] i_buff
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            if j == 0:
                value = np.array([vector_is_valid(inputs[j])]) 
            else:
                value = np.append(value, np.array([vector_is_valid(inputs[j])]),axis=0)
        i_buff = np.ascontiguousarray(value,dtype=np.float32)
        cdef float** i = <float**>malloc(sizeof(float*)*batch_size)
        for j in range(batch_size):
            i[j] = &i_buff[j, 0]
        Pyllab.model_tensor_input_ff_multicore_opt(self._models, self._model, tensor_depth, tensor_i, tensor_j, &i[0], batch_size, batch_size)
        free(i)
    
    def back_propagation_opt_multi_thread(self, int tensor_depth, int tensor_i, int tensor_j, inputs, error, int error_dimension, ret_err = False):
        if self._model is NULL:
            return None
        if not self._is_multithread:
            return None
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(error_dimension)
        check_int(tensor_j*tensor_i*tensor_depth)
        if error_dimension != self.get_output_dimension_from_model():
            print("Error: the error dimension doesn't match the output dimension of the model")
            exit(1)
        
        if(get_first_dimension_size(inputs) != get_first_dimension_size(error)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        cdef int batch_size = self.threads
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        value_i = None
        value_e = None
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] i_buff
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] e_buff
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if j == 0:
                value_i = np.array([vector_is_valid(inputs[j])])
            else:
                value_i = np.append(value_i, np.array([vector_is_valid(inputs[j])]),axis=0)
            if j == 0:
                value_e = np.array([vector_is_valid(error[j])])
            else:
                value_e = np.append(value_e, np.array([vector_is_valid(error[j])]),axis=0)
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        e_buff = np.ascontiguousarray(value_e,dtype=np.float32)
        cdef float** i = <float**>malloc(sizeof(float*)*batch_size)
        cdef float** e = <float**>malloc(sizeof(float*)*batch_size)
        for j in range(batch_size):
            i[j] = &i_buff[j, 0]
            e[j] = &e_buff[j, 0]
            
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.model_tensor_input_bp_multicore_opt(self._models,self._model, tensor_depth, tensor_i, tensor_j, &i[0], batch_size, batch_size,&e[0], error_dimension, ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            free(i)
            free(e)
            return np.array(l, dtype='float')
        free(i)
        free(e)
        return None
    
    def ff_error_bp_opt_multi_thread(self,int tensor_depth, int tensor_i, int tensor_j, inputs, output, int error_dimension, ret_err = False):
        if self._model is NULL:
            return None
        if not self._is_multithread:
            return None
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(error_dimension)
        check_int(tensor_j*tensor_i*tensor_depth)
        if error_dimension != self.get_output_dimension_from_model():
            print("Error: the error dimension doesn't match the output dimension of the model")
            exit(1)
        
        
        error = output
        value_i = None
        value_e = None
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] i_buff
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] e_buff
        if(get_first_dimension_size(inputs) != get_first_dimension_size(error)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        cdef int batch_size = self.threads
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if j == 0:
                value_i = np.array([vector_is_valid(inputs[j])])
            else:
                value_i = np.append(value_i, np.array([vector_is_valid(inputs[j])]),axis=0)
            if j == 0:
                value_e = np.array([vector_is_valid(error[j])])
            else:
                value_e = np.append(value_e, np.array([vector_is_valid(error[j])]),axis=0)
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        e_buff = np.ascontiguousarray(value_e,dtype=np.float32)
        cdef float** i = <float**>malloc(sizeof(float*)*batch_size)
        cdef float** e = <float**>malloc(sizeof(float*)*batch_size)
        for j in range(batch_size):
            i[j] = &i_buff[j, 0]
            e[j] = &e_buff[j, 0]
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.ff_error_bp_model_multicore_opt(self._models, self._model, tensor_depth, tensor_i, tensor_j, &i[0], batch_size, batch_size,&e[0], ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            free(i)
            free(e)
            return np.array(l, dtype='float')
        free(i)
        free(e)
        return None
    
    
    
    def output_layer_of_ith(self, int index):
        if self._model is NULL:
            return None
        check_int(index)
        if self._is_multithread:
            if index < self.threads:
                return from_float_to_ndarray(Pyllab.get_output_layer_from_model(self._models[index]),Pyllab.get_output_dimension_from_model(self._models[index]))
        return None
    
    def sum_models_partial_derivatives(self):
        if self._model is NULL:
            return None
        if self._is_multithread:
            Pyllab.sum_models_partial_derivatives_multithread(self._models, self._model, self.threads, 0)
        
    def set_ith_layer_training_mode_models(self, int ith, int training_flag):
        if self._model is NULL:
            return None
        check_int(training_flag)
        check_int(ith)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_ith_layer_training_mode_model(self._models[i],ith,training_flag)
    
    def set_k_percentage_of_ith_layer_models(self,int ith, float k):
        if self._model is NULL:
            return None
        check_float(k)
        check_int(ith)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_k_percentage_of_ith_layer_model(self._models[i], ith, k)
            
            
            
       
def paste_model(model m1, model m2):
    if m1._model is NULL or m2._model is NULL:
        return
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.paste_model(m1._model,m2._model)
    
def paste_model_without_learning_parameters(model m1, model m2):
    if m1._model is NULL or m2._model is NULL:
        return
    if m1._does_have_arrays and m2._does_have_arrays and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.paste_model_without_learning_parameters(m1._model,m2._model)

def copy_model(model m):
    if m._model is NULL:
        return
    cdef Pyllab.model* copy
    if m._does_have_learning_parameters and m._does_have_arrays and not m._is_only_for_feedforward:
        if not m._is_from_char:
            l1 = []
            for i in range(0,len(m.fcls)):
                l1.append(copy_fcl(m.fcls[i]))
            l2 = []
            for i in range(0,len(m.cls)):
                l2.append(copy_cl(m.cls[i]))
            l3 = []
            for i in range(0,len(m.rls)):
                l3.append(copy_rl(m.rls[i]))
            mm = model(m._n_fcl,m._n_cl,m._n_rl,l1,l2,l3)
            paste_model(m,mm)
            return mm
        copy = Pyllab.copy_model(m._model)
        mod = model(mod = True)
        mod._model = copy
        return mod
    
def copy_model_without_learning_parameters(model m):
    if m._model is NULL:
        return
    cdef Pyllab.model* copy 
    if not m._does_have_learning_parameters and m._does_have_arrays and not m._is_only_for_feedforward:
        if not m._is_from_char:
            l1 = []
            for i in range(0,len(m.fcls)):
                l1.append(copy_fcl_without_learning_parameters(m.fcls[i]))
            l2 = []
            for i in range(0,len(m.cls)):
                l2.append(copy_cl_without_learning_parameters(m.cls[i]))
            l3 = []
            for i in range(0,len(m.rls)):
                l3.append(copy_rl_without_learning_parameters(m.rls[i]))
            mm = model(m._n_fcl,m._n_cl,m._n_rl,l1,l2,l3)
            paste_model_without_learning_parameters(m,mm)
            return mm
        copy = Pyllab.copy_model_without_learning_parameters(m._model)
        mod = model(mod = True)
        mod._model = copy
        mod._does_have_learning_parameters = False
        return mod

def slow_paste_model(model m1, model m2, float tau):
    if m1._model is NULL or m2._model is NULL:
        return
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.slow_paste_model(m1._model, m2._model,tau)

def copy_vector_to_params_model(model m, vector):
    if m._model is NULL:
        return
    check_size(vector,m.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_model(m._model,<float*>&v[0])

def copy_params_to_vector_model(model m):
    if m._model is NULL:
        return
    vector = np.arange(m.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_model(model m, vector):
    if m._model is NULL:
        return
    check_size(vector,m.get_array_size_weights())
    cdef float[:]v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_model(m._model,<float*>&v[0])

def copy_weights_to_vector_model(model m):
    if m._model is NULL:
        return
    vector = np.arange(m.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_model(model m, vector):
    if m._model is NULL:
        return
    check_size(vector,m.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_model(m._model,<float*>&v[0])

def copy_scores_to_vector_model(model m):
    if m._model is NULL:
        return
    vector = np.arange(m.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_derivative_params_model(model m, vector):
    if m._model is NULL:
        return
    check_size(vector,m.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_derivative_params_model(m._model,<float*>&v[0])

def copy_derivative_params_to_vector_model(model m):
    if m._model is NULL:
        return
    vector = np.arange(m.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_derivative_params_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def compare_score_model(model m1, model m2, model m_output):
    if m1._model is NULL or m2._model is NULL or m_output._model is NULL:
        return
    Pyllab.compare_score_model(m1._model, m2._model,m_output._model)
    
def compare_score_model_with_vector(model m1, vector, model m_output):
    if m1._model is NULL or m2._model is NULL or m_output._model is NULL:
        return    
    check_size(vector,m.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    Pyllab.compare_score_model_with_vector(m1._model, <float*>&v[0],m_output._model)
    
def sum_score_model(model m1,model m2, model m_output):
    f m1._model is NULL or m2._model is NULL or m_output._model is NULL:
        return    
    Pyllab.sum_score_model(m1._model,m2._model,m_output._model)
