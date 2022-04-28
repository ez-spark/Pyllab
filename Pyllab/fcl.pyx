cimport Pyllab
from Pyllab import *

cdef class fcl:
    cdef Pyllab.fcl* _fcl
    cdef int _input
    cdef int _output
    cdef int _layer
    cdef int _dropout_flag
    cdef int _activation_flag
    cdef float _dropout_threshold
    cdef int _n_groups
    cdef int _normalization_flag
    cdef int _training_mode
    cdef int _feed_forward_flag
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    
    def __cinit__(self,int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(input)
        check_int(output)
        check_int(layer)
        check_int(dropout_flag)
        check_int(activation_flag)
        check_int(n_groups)
        check_int(normalization_flag)
        check_int(training_mode)
        check_int(feed_forward_flag)
        check_float(dropout_threshold)
        
        self._input = input
        self._output = output
        self._layer = layer
        self._dropout_flag = dropout_flag
        self._activation_flag = activation_flag
        self._dropout_threshold = dropout_threshold
        self._n_groups = n_groups
        self._normalization_flag = normalization_flag
        self._training_mode = training_mode
        self._feed_forward_flag = feed_forward_flag
        self._does_have_learning_parameters = does_have_learning_parameters
        self._does_have_arrays = does_have_arrays
        self._is_only_for_feedforward = is_only_for_feedforward
        
        if does_have_arrays:
            if does_have_learning_parameters:
                self._fcl = Pyllab.fully_connected(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)
            else:
                self._fcl = Pyllab.fully_connected_without_learning_parameters(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)

            
            if is_only_for_feedforward:
                Pyllab.make_the_fcl_only_for_ff(self._fcl)
        else:
            self._fcl = Pyllab.fully_connected_without_arrays(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)
        
        if self._fcl is NULL:
            raise MemoryError()
        
    def __dealloc__(self):
        if self._fcl is not NULL and self._does_have_arrays:
            Pyllab.free_fully_connected(self._fcl)
        elif self._fcl is not NULL:
            Pyllab.free_fully_connected_without_arrays(self._fcl)
    
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_fcl(self._fcl, number_of_file)
        
    def get_size(self):
        if not self._is_only_for_feedforward:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_fcls(self._fcl)
            else:
                return Pyllab.size_of_fcls_without_learning_parameters(self._fcl)
        return 0
    
    def make_it_only_for_ff(self):
        if not self._is_only_for_feedforward and self._does_have_arrays:
            self._is_only_for_feedforward = True
            Pyllab.make_the_fcl_only_for_ff(self._fcl)
    
    def reset(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_fcl(self._fcl)
            else:
                Pyllab.reset_fcl_without_learning_parameters(self._fcl)
                
    def clip(self, float threshold, float norm):
        check_float(threshold)
        check_float(norm)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_fcls(&self._fcl,1, threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_fcl(self._fcl, threshold, epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params(self._fcl)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights(self._fcl)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_fcl(self._fcl)
    
    def set_biases_to_zero(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_fully_connected_biases_to_zero(self._fcl)
    
    def set_unused_weights_to_zero(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_fully_connected_unused_weights_to_zero(self._fcl)    
        
    def reset_scores(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.reset_score_fcl(self._fcl)
    def set_low_scores(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_low_score_fcl(self._fcl)    
    
    def get_number_of_weights(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            return Pyllab.count_weights_fcl(self._fcl)


def paste_fcl(fcl f1, fcl f2):
    if f1._does_have_arrays and f1._does_have_learning_parameters and not f1._is_only_for_feedforward and f2._does_have_arrays and f2._does_have_learning_parameters and not f2._is_only_for_feedforward:
        Pyllab.paste_fcl(f1._fcl,f2._fcl)
    
def paste_fcl_without_learning_parameters(fcl f1, fcl f2):
    if f1._does_have_arrays and f2._does_have_arrays and not f1._is_only_for_feedforward and not f2._is_only_for_feedforward:
        Pyllab.paste_fcl_without_learning_parameters(f1._fcl,f2._fcl)

def copy_fcl(fcl f):
    cdef fcl ff = (f._input,f._output,f._layer,f._dropout_flag,f._activation_flag, f._dropout_threshold,f._n_groups,f._normalization_flag,f._training_mode,f._feed_forward_flag,f._does_have_learning_parameters, f._does_have_arrays, f._is_only_for_feedforward)
    paste_fcl(f,ff)
    return ff

def copy_fcl_without_learning_parameters(fcl f):
    cdef fcl ff = (f._input,f._output,f._layer,f._dropout_flag,f._activation_flag, f._dropout_threshold,f._n_groups,f._normalization_flag,f._training_mode,f._feed_forward_flag,False, f._does_have_arrays, f._is_only_for_feedforward)
    paste_fcl_without_learning_parameters(f,ff)
    return ff

def slow_paste_fcl(fcl f1, fcl f2, float tau):
    if f1._does_have_arrays and f1._does_have_learning_parameters and f2._does_have_arrays and f2._does_have_learning_parameters and not f1._is_only_for_feedforward and not f2._is_only_for_feedforward:
        Pyllab.slow_paste_fcl(f1._fcl, f2._fcl,tau)

def copy_vector_to_params_fcl(fcl f, vector):
    check_size(vector,f.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params(f._fcl,<float*>&v[0])

def copy_params_to_vector_fcl(fcl f):
    vector = np.arange(f.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector(f._fcl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_fcl(fcl f, vector):
    check_size(vector,f.get_array_size_weights())
    cdef float[:] v = vector_is_valid(vector)
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights(f._fcl,<float*>&v[0])

def copy_weights_to_vector_fcl(fcl f):
    vector = np.arange(f.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector(f._fcl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_fcl(fcl f, vector):
    check_size(vector,f.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores(f._fcl,<float*>&v[0])

def copy_scores_to_vector_fcl(fcl f):
    vector = np.arange(f.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector(f._fcl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_derivative_params_fcl(fcl f, vector):
    check_size(vector,f.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if f._does_have_arrays and f._does_have_learning_parameters and not f._is_only_for_feedforward:
        Pyllab.memcopy_vector_to_derivative_params(f._fcl,<float*>&v[0])

def copy_derivative_params_to_vector_fcl(fcl f):
    vector = np.arange(f.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if f._does_have_arrays and f._does_have_learning_parameters and not f._is_only_for_feedforward:
        Pyllab.memcopy_derivative_params_to_vector(f._fcl,<float*>&v[0])
        return vector
    return None

def compare_score_fcl(fcl f1, fcl f2, fcl f_output):
    if f1._does_have_arrays and f1._does_have_learning_parameters and f2._does_have_arrays and f2._does_have_learning_parameters and f_output._does_have_arrays and f_output._does_have_learning_parameters: 
        Pyllab.compare_score_fcl(f1._fcl, f2._fcl,f_output._fcl)
    
def compare_score_fcl_with_vector(fcl f1, vector, fcl f_output):
	check_size(vector,f1.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if f1._does_have_arrays and f1._does_have_learning_parameters and f_output._does_have_arrays and f_output._does_have_learning_parameters:
        Pyllab.compare_score_fcl_with_vector(f1._fcl, <float*>&v[0],f_output._fcl)
