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
    cdef int _layers
    cdef int threads
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    cdef bint _is_from_char
    cdef bint _is_multithread
    '''
    @ filename := filename from which the model is loaded can be a .txt (setup file) or a .bin (an entire model with also weights and stuff
                  if the mod is true is a .bin basically, or it should be
    @ string := a char* type array, it is basically the same as filename, but all the file has been read into string
    @ bint mod:= is a flag, if true we must read from a .bin if filename is != None, if it is true and filename == None we just set some parameters and we don't have any model (is used by copy_model)
     all the other parameters from layers and going forward, are used if we pass the class cl, ,fcl, rl. Basically
     we are building the model from each class defined in python, we pass all the classes defined in python and et voilà we build in C the struct model
    '''
    def __cinit__(self,filename=None, string = None, bint mod = True, int layers=0, int n_fcl=0, int n_cl=0, int n_rl=0, list fcls=None, list cls=None, list rls=None, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(layers)
        check_int(n_fcl)
        check_int(n_cl)
        check_int(n_rl)
        self._is_multithread = False
        self.threads = 1
        if filename != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            
            
            if mod == False:
                self._model = Pyllab.parse_model_file(PyUnicode_AsUTF8(filename))
            else:
                if not dict_to_pass_to_model_is_good(get_dict_from_model_setup_file(filename)):
                    print("Error: your setup file is not correct")
                    exit(1)
                self._model = Pyllab.load_model(PyUnicode_AsUTF8(filename))
            if self._model is NULL:
                raise MemoryError()
        elif mod == True:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
        else:
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
              
        
    def __dealloc__(self):
        
        if self._model is not NULL:
            if self._does_have_arrays:
                if self._does_have_learning_parameters:
                    Pyllab.free_model(self._model)
                else:
                    Pyllab.free_model_without_learning_parameters(self._model)
            else:
                Pyllab.free_model_without_arrays(self._model)
        if self.is_multithread:
            for i in range(self.threads):
                Pyllab.free_model_without_learning_parameters(self._models[i])
                free(self._models)
    
    def make_multi_thread(self, int threads):
        if threads < 1:
            print("Error: the number of threads must be >= 1")
            exit(1)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_model_without_learning_parameters(self._models[i])
            free(self._models):
        self._models = malloc(sizeof(<Pyllab.model**>)*threads)
        for i in range(threads):
            self._models[i] = Pyllab.copy_model_without_learning_parameters(self._model)
        self._is_multithread = True
        self.threads = threads
        
    def make_single_thread(self):
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_model_without_learning_parameters(self._models[i])
            free(self._models)
        self._is_multithread = False
        self.threads = 1
        
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_model(self._model, number_of_file)
    
    def get_size(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_model(self._model)
            else:
                return Pyllab.size_of_model_without_learning_parameters(self._model)
        return 0
    
    def make_it_only_for_ff(self):
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
    def clip(self, float threshold):
        check_float(threshold)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.clipping_gradient(self._model, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.adaptive_gradient_clipping_model(self._model,threshold,epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_model(self._model)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights_model(self._model)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_model(self._model)
    
    def set_biases_to_zero(self):
        Pyllab.set_model_biases_to_zero(self._model)
    
    def set_unused_weights_to_zero(self):
        Pyllab.set_model_unused_weights_to_zero(self._model)    
        
    def reset_scores(self):
        Pyllab.reset_score_model(self._model)
    def set_low_scores(self):
        Pyllab.set_low_score_model(self._model)
    
    def get_number_of_weights(self):
        return Pyllab.count_weights(self._model)
    
    def set_model_error(self, int error_flag, int output_dimension, float threshold1 = 0, float threshold2 = 0, float gamma = 0, alpha = None):
        cdef array.array a
        if alpha == None:
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,NULL,output_dimension)
        else:
            check_size(alpha,output_dimension)
            a = vector_is_valid(alpha)
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,<float*>a.data.as_floats,output_dimension)
    
    def feed_forward(self, int tensor_depth, int tensor_i, int tensor_j, inputs):
        check_size(inputs,tensor_depth*tensor_i*tensor_j)
        cdef array.array i = vector_is_valid(inputs)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.model_tensor_input_ff(self._model,tensor_depth,tensor_i,tensor_j,<float*>i.data.as_floats)
    
    def back_propagation(self, int tensor_depth, int tensor_i, int tensor_j, inputs, error, int error_dimension):
        check_size(inputs,tensor_depth*tensor_i*tensor_j)
        check_size(error,error_dimension)
        cdef array.array i = vector_is_valid(inputs)
        cdef array.array e = vector_is_valid(error)
        cdef float* ret = Pyllab.model_tensor_input_bp(self._model,tensor_depth,tensor_i,tensor_j,<float*>i.data.as_floats,<float*>e.data.as_floats,error_dimension)
        return from_float_to_ndarray(ret,tensor_depth*tensor_i*tensor_j)
        
    def ff_error_bp(self, int tensor_depth, int tensor_i, int tensor_j, inputs, outputs):
        check_size(inputs,tensor_depth*tensor_i*tensor_j)
        check_size(outputs,self.get_output_dimension_from_model())
        cdef array.array i = vector_is_valid(inputs)
        cdef array.array o = vector_is_valid(outputs)
        cdef float* ret = Pyllab.ff_error_bp_model_once(self._model,tensor_depth,tensor_i,tensor_j,<float*>i.data.as_floats,<float*>o.data.as_floats)
        return from_float_to_ndarray(ret,tensor_depth*tensor_i*tensor_j)
        
    
    def set_training_edge_popup(self, float k_percentage):
        Pyllab.set_model_training_edge_popup(self._model,k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        Pyllab.reinitialize_weights_according_to_scores_model(self._model,percentage,goodness)
    
    def set_training_gd(self):
        Pyllab.set_model_training_gd(self._model)
    
    def reset_edge_popup_d_params(self):
        Pyllab.reset_edge_popup_d_model(self._model)
    
    def output_layer(self):
        return from_float_to_list(Pyllab.get_output_layer_from_model(self._model),Pyllab.get_output_dimension_from_model(self._model))
    
    def get_output_dimension_from_model(self):
        n = Pyllab.get_output_dimension_from_model(self._model)
        return int(n)
    def set_beta1(self, float beta):
        check_float(beta)
        Pyllab.set_model_beta(self._model, beta, self.get_beta2())
    
    def set_beta2(self, float beta):
        check_float(beta)
        Pyllab.set_model_beta(self._model, self.get_beta1(), beta)
        
    def set_beta3(self, float beta):
        check_float(beta)
        Pyllab.set_model_beta_adamod(self._model, beta)
    
    def get_beta1(self):
        b = Pyllab.get_beta1_from_model(self._model)
        return b
    
    def get_beta2(self):
        b = Pyllab.get_beta2_from_model(self._model)
        return b
        
    def get_beta3(self):
        b = Pyllab.get_beta2_from_model(self._model)
        return b
    
    def set_ith_layer_training_mode(self, int ith, int training_flag):
        Pyllab.set_ith_layer_training_mode(self._model,ith,training_flag)
    
    def set_k_percentage_of_ith_layer(self,int ith, float k):
        check_float(k)
        Pyllab.set_k_percentage_of_ith_layer_model(self._model, ith, k)
       
def paste_model(model m1, model m2):
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters:
        Pyllab.paste_model(m1._model,m2._model)
    
def paste_model_without_learning_parameters(model m1, model m2):
    if m1._does_have_arrays and m2._does_have_arrays:
        Pyllab.paste_model_without_learning_parameters(m1._model,m2._model)

def copy_model_c(model m):
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
    cdef Pyllab.model* copy = Pyllab.copy_model(m._model)
    mod = model(mod = True)
    mod._model = copy
    mod._is_from_char = True
    return mod
    
def copy_model_without_learning_parameters(model m):
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
    cdef Pyllab.model* copy = Pyllab.copy_model_without_learning_parameters(m._model)
    mod = model(mod = True,does_have_learning_parameters = False)
    mod._model = copy
    mod._is_from_char = True
    return mod

def slow_paste_model(model m1, model m2, float tau):
    Pyllab.slow_paste_model(m1._model, m2._model,tau)

def copy_vector_to_params_model(model m, vector):
    check_size(vector,m.get_array_size_params())
    cdef array.array v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_model(m._model,<float*>v.data.as_floats)

def copy_params_to_vector_model(model m, vector):
    v = np.arange(m.get_array_size_params(),dtype=np.dtype("f"))
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_model(m._model,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_weights_model(model m, vector):
    check_size(vector,m.get_array_size_weights())
    cdef array.array v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_model(m._model,<float*>v.data.as_floats)

def copy_weights_to_vector_model(model m, vector):
    v = np.arange(m.get_array_size_weights(),dtype=np.dtype("f"))
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_model(m._model,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_scores_model(model m, vector):
    check_size(vector,m.get_array_size_scores())
    cdef array.array v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_model(m._model,<float*>v.data.as_floats)

def copy_scores_to_vector_model(model m, vector):
    v = np.arange(m.get_array_size_scores(),dtype=np.dtype("f"))
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_model(m._model,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_derivative_params_model(model m, vector):
    check_size(vector,m.get_array_size_params())
    cdef array.array v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_derivative_params_model(m._model,<float*>v.data.as_floats)

def copy_derivative_params_to_vector_model(model m, vector):
    v = np.arange(m.get_array_size_params(),dtype=np.dtype("f"))
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_derivative_params_to_vector_model(m._model,<float*>v.data.as_floats)
        return v
    return None

def compare_score_model(model m1, model m2, model m_output):
    Pyllab.compare_score_model(m1._model, m2._model,m_output._model)
    
def compare_score_model_with_vector(model m1, vector, model m_output):
    cdef array.array v = vector_is_valid(vector)
    Pyllab.compare_score_model_with_vector(m1._model, <float*>v.data.as_floats,m_output._model)
    
def sum_score_model(model m1,model m2, model m_output):
    Pyllab.sum_score_model(m1._model,m2._model,m_output._model)


cdef class modelBatch:
    cdef Pyllab.model** _models
    list models
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    def __cinit__(self,list models,int n_models, bint does_have_learning_parameters = False, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        self.models = models
        self._does_have_arrays = does_have_arrays
        self._does_have_learning_parameters = does_have_learning_parameters
        self._is_only_for_feedforward = is_only_for_feedforward
        if n_models != len(models):
            print("Your number of models (n_models) does not match the list size of models")
            return
        
        if n_models > 0:
            self._models = <Pyllab.model**>malloc(n_models*sizeof(Pyllab.model*))
            if self._models is NULL:
                raise MemoryError()
            for i in range(0,n_models):
                self._models[i] = <Pyllab.model*>models[i]._model
                
    def __dealloc__(self):
        n_models = len(self.models)
        if n_models > 0:
            for i in range(0,n_models):
                self.models[i].__dealloc__()
            free(self._models)
    
    def reset(self):
        n_models = len(self.models)
        for i in range(0,n_models):
            self.models[i].reset()
    
    def set_model_error(self, int error_flag, int output_dimension, float threshold1 = 0, float threshold2 = 0, float gamma = 0, alpha = None):
        n_models = len(self.models)
        for i in range(n_models):
            self.models[i].set_model_error(error_flag,output_dimension,threshold1,threshold2, gamma, alpha)
    
    def feed_forward(self, int tensor_depth, int tensor_i, int tensor_j, inputs):
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
        Pyllab.model_tensor_input_ff_multicore(self._models, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size)
        free(i)
    
    def feed_forward_opt(self, model m, int tensor_depth, int tensor_i, int tensor_j, inputs):
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
        Pyllab.model_tensor_input_ff_multicore_opt(self._models, m._model, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size)
        free(i)
    
    def back_propagation(self, int tensor_depth, int tensor_i, int tensor_j, inputs, error, int error_dimension, ret_err = False):
        if(get_first_dimension_size(inputs) != get_first_dimension_size(error)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        cdef float** e = <float**>malloc(batch_size*sizeof(float*))
        if e is NULL:
            raise MemoryError()
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if type(inputs[j]) == array.array:
                assign_float_pointer_from_array(i,inputs[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
            if type(error[j]) == array.array:
                assign_float_pointer_from_array(e,error[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(e,vector_is_valid(error[j]).data.as_floats,j)
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.model_tensor_input_bp_multicore(self._models, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size,e, error_dimension, ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            return np.ndarray(l)
        free(i)
        free(e)
        return None
        
    def back_propagation_opt(self, model m, int tensor_depth, int tensor_i, int tensor_j, inputs, error, int error_dimension, ret_err = False):
        if(get_first_dimension_size(inputs) != get_first_dimension_size(error)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        cdef float** e = <float**>malloc(batch_size*sizeof(float*))
        if e is NULL:
            raise MemoryError()
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if type(inputs[j]) == array.array:
                assign_float_pointer_from_array(i,inputs[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
            if type(error[j]) == array.array:
                assign_float_pointer_from_array(e,error[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(e,vector_is_valid(error[j]).data.as_floats,j)
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.model_tensor_input_bp_multicore_opt(self._models,m._model, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size,e, error_dimension, ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            return np.ndarray(l)
        free(i)
        free(e)
        return None
        
    def ff_error_bp(self, int tensor_depth, int tensor_i, int tensor_j, inputs, output, int error_dimension, ret_err = False):
        error = output
        if(get_first_dimension_size(inputs) != get_first_dimension_size(output)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        cdef float** e = <float**>malloc(batch_size*sizeof(float*))
        if e is NULL:
            raise MemoryError()
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if type(inputs[j]) == array.array:
                assign_float_pointer_from_array(i,inputs[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
            if type(error[j]) == array.array:
                assign_float_pointer_from_array(e,error[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(e,vector_is_valid(error[j]).data.as_floats,j)
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.ff_error_bp_model_multicore(self._models, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size,e, ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            return np.ndarray(l)
        free(i)
        free(e)
        return None
        
    def ff_error_bp_opt(self, model m, int tensor_depth, int tensor_i, int tensor_j, inputs, output, int error_dimension, ret_err = False):
        error = output
        if(get_first_dimension_size(inputs) != get_first_dimension_size(error)):
            print("Error: your input size does not match the error size in batch dimension")
            exit(1)
        
        cdef int batch_size = len(self.models)
        if get_first_dimension_size(inputs) < batch_size:
            batch_size = get_first_dimension_size(inputs)
        cdef float** i = <float**>malloc(batch_size*sizeof(float*))
        if i is NULL:
            raise MemoryError()
        cdef float** e = <float**>malloc(batch_size*sizeof(float*))
        if e is NULL:
            raise MemoryError()
        cdef float** ret = NULL
        for j in range(batch_size):
            check_size(inputs[j], tensor_depth*tensor_i*tensor_j)
            check_size(error[j], self.get_output_dimension_from_model())
            if type(inputs[j]) == array.array:
                assign_float_pointer_from_array(i,inputs[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(i,vector_is_valid(inputs[j]).data.as_floats,j)
            if type(error[j]) == array.array:
                assign_float_pointer_from_array(e,error[j].data.as_floats,j)
            else:
                assign_float_pointer_from_array(e,vector_is_valid(error[j]).data.as_floats,j)
        if ret_err:
            ret == <float**>malloc(sizeof(float*)*batch_size)
            if ret is NULL:
                raise MemoryError()
        Pyllab.ff_error_bp_model_multicore_opt(self._models, m._model, tensor_depth, tensor_i, tensor_j, i, batch_size, batch_size,e, ret)
        if not (ret is NULL):
            l = []
            for j in range(batch_size):
                l.append(from_float_to_list(ret[j], tensor_depth*tensor_i*tensor_j))
            free(ret)
            return np.ndarray(l)
        free(i)
        free(e)
        return None
        
    
    def set_training_edge_popup(self, float k_percentage):
        n_models = len(self.models)
        for i in range(n_models):
            self.models[i].set_training_edge_popup(k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        n_models = len(self.models)
        for i in range(n_models):
            self.models[i].reinitialize_weights_according_to_scores(percentage,goodness)
    
    def set_training_gd(self):
        n_models = len(self.models)
        for i in range(n_models):
            self.models[i].set_training_gd()
    
    def reset_edge_popup_d_params(self):
        n_models = len(self.models)
        for i in range(n_models):
            self.models[i].reset_edge_popup_d_params()
    def output_layer_of_ith(self, index):
        if index < len(self.models):
            return self.models[index].output_layer()
        return None
    
    def sum_models_partial_derivatives(self, model m):
        Pyllab.sum_models_partial_derivatives_multithread(self._models, m._model, len(self.models), 0)
        
    def set_beta1(self, float beta):
        for i in range(len(self.models)):
            self.models[i].set_beta1(beta)
    
    def set_beta2(self, float beta):
        for i in range(len(self.models)):
            self.models[i].set_beta2(beta)
    
    def set_ith_layer_training_mode(self, int ith, int training_flag):
        for i in range(len(self.models)):
            self.models[i].set_ith_layer_training_mode(ith, training_flag)
    
    def set_k_percentage_of_ith_layer(self,int ith, float k):
        for i in range(len(self.models)):
            self.models[i].set_k_percentage_of_ith_layer(ith, k)

