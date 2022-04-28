cimport Pyllab
from Pyllab import *

cdef class cl:
    cdef Pyllab.cl* _cl
    cdef int _channels
    cdef int _input_rows
    cdef int _input_cols
    cdef int _kernel_rows
    cdef int _kernel_cols
    cdef int _n_kernels
    cdef int _stride1_rows
    cdef int _stride2_rows
    cdef int _stride1_cols
    cdef int _stride2_cols
    cdef int _padding1_rows
    cdef int _padding2_rows
    cdef int _padding1_cols
    cdef int _padding2_cols
    cdef int _pooling_rows
    cdef int _pooling_cols
    cdef int _normalization_flag
    cdef int _activation_flag
    cdef int _pooling_flag
    cdef int _group_norm_channels
    cdef int _convolutional_flag
    cdef int _training_mode
    cdef int _feed_forward_flag
    cdef int _layer
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    
    def __cinit__(self,int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(channels)
        check_int(input_rows)
        check_int(input_cols)
        check_int(kernel_rows)
        check_int(kernel_cols)
        check_int(n_kernels)
        check_int(stride1_rows)
        check_int(stride2_rows)
        check_int(stride1_cols)
        check_int(stride2_cols)
        check_int(padding1_rows)
        check_int(padding1_cols)
        check_int(padding2_rows)
        check_int(padding2_cols)
        check_int(pooling_rows)
        check_int(pooling_cols)
        check_int(normalization_flag)
        check_int(activation_flag)
        check_int(pooling_flag)
        check_int(group_norm_channels)
        check_int(convolutional_flag)
        check_int(training_mode)
        check_int(feed_forward_flag)
        check_int(layer)
        self._channels = channels
        self._input_rows = input_rows
        self._input_cols = input_cols
        self._kernel_rows = kernel_rows
        self._kernel_cols = kernel_cols
        self._n_kernels = n_kernels
        self._stride1_rows = stride1_rows
        self._stride1_cols = stride1_cols
        self._stride2_rows = stride2_rows
        self._stride2_cols = stride2_cols
        self._padding1_rows = padding1_rows
        self._padding1_cols = padding1_cols
        self._padding2_rows = padding2_rows
        self._padding2_cols = padding2_cols
        self._pooling_rows = pooling_rows
        self._pooling_cols = pooling_cols
        self._normalization_flag = normalization_flag
        self._activation_flag = activation_flag
        self._pooling_flag = pooling_flag
        self._group_norm_channels = group_norm_channels
        self._convolutional_flag = convolutional_flag
        self._training_mode = training_mode
        self._feed_forward_flag = feed_forward_flag
        self._layer = layer
        self._does_have_learning_parameters = does_have_learning_parameters
        self._does_have_arrays = does_have_arrays
        self._is_only_for_feedforward = is_only_for_feedforward
        
        if does_have_arrays:
            if does_have_learning_parameters :
                self._cl = Pyllab.convolutional(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)
            else:
                self._cl = Pyllab.convolutional_without_learning_parameters(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)

            
            if is_only_for_feedforward:
                Pyllab.make_the_cl_only_for_ff(self._cl)
        else:
            self._cl = Pyllab.convolutional_without_arrays(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)
        
        if self._cl is NULL:
            raise MemoryError()
        
    def __dealloc__(self):
        if self._cl is not NULL and self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.free_convolutional(self._cl)
        elif self._cl is not NULL and self._does_have_arrays:
            Pyllab.free_convolutional_without_learning_parameters(self._cl)
        elif self._cl is not NULL:
            Pyllab.free_convolutional_without_arrays(self._cl)
    
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.save_cl(self._cl, number_of_file)
        
    def get_size(self):
        if not self._is_only_for_feedforward:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_cls(self._cl)
            else:
                return Pyllab.size_of_cls_without_learning_parameters(self._cl)
        return 0
    
    def make_it_only_for_ff(self):
        if not self._is_only_for_feedforward and self._does_have_arrays and self._does_have_learning_parameters:
            self._is_only_for_feedforward = True
            Pyllab.make_the_cl_only_for_ff(self._cl)
    
    def reset(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_cl(self._cl)
            else:
                Pyllab.reset_cl_without_learning_parameters(self._cl)
                
    def clip(self, float threshold, float norm):
        check_float(threshold)
        check_float(norm)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_cls(&self._cl, 1, threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_cl(self._cl, threshold, epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_cl(self._cl)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights_cl(self._cl)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_cl(self._cl)
    
    def set_biases_to_zero(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_convolutional_biases_to_zero(self._cl)
    
    def set_unused_weights_to_zero(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_convolutional_unused_weights_to_zero(self._cl)    
        
    def reset_scores(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.reset_score_cl(self._cl)
    def set_low_scores(self):
        if self._does_have_learning_parameters and self._does_have_arrays:
            Pyllab.set_low_score_cl(self._cl)    
    
    def get_number_of_weights(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            return Pyllab.count_weights_cl(self._cl)


def paste_cl(cl c1, cl c2):
    if c1._does_have_arrays and c1._does_have_learning_parameters and not c1._is_only_for_feedforward and c2._does_have_arrays and c2._does_have_learning_parameters and not c2._is_only_for_feedforward:
        Pyllab.paste_cl(c1._cl,c2._cl)
    
def paste_cl_without_learning_parameters(cl c1, cl c2):
    if c1._does_have_arrays and c2._does_have_arrays and not c1._is_only_for_feedforward and not c2._is_only_for_feedforward:
        Pyllab.paste_cl_without_learning_parameters(c1._cl,c2._cl)

def copy_cl(cl c):
    cdef cl cc = cl(c._channels,c._input_rows,c._input_cols,c._kernel_rows,c._kernel_cols,c._n_kernels,c._stride1_rows,c._stride1_cols,c._padding1_rows,c._padding1_cols,c._stride2_rows,c._stride2_cols,c._padding2_rows,c._padding2_cols,c._pooling_rows,c._pooling_cols,c._normalization_flag,c._activation_flag,c._pooling_flag,c._group_norm_channels,c._convolutional_flag,c._training_mode,c._feed_forward_flag,c._layer,c._does_have_learning_parameters,c._does_have_arrays,c._is_only_for_feedforward)
    paste_cl(c,cc)
    return cc

def copy_cl_without_learning_parameters(cl c):
    cdef cl cc = cl(c._channels,c._input_rows,c._input_cols,c._kernel_rows,c._kernel_cols,c._n_kernels,c._stride1_rows,c._stride1_cols,c._padding1_rows,c._padding1_cols,c._stride2_rows,c._stride2_cols,c._padding2_rows,c._padding2_cols,c._pooling_rows,c._pooling_cols,c._normalization_flag,c._activation_flag,c._pooling_flag,c._group_norm_channels,c._convolutional_flag,c._training_mode,c._feed_forward_flag,c._layer,False,c._does_have_arrays,c._is_only_for_feedforward)
    paste_cl_without_learning_parameters(c,cc)
    return cc

def slow_paste_cl(cl c1, cl c2, float tau):
    if c1._does_have_arrays and c1._does_have_learning_parameters and not c1._is_only_for_feedforward and c2._does_have_arrays and c2._does_have_learning_parameters and not c2._is_only_for_feedforward:
        Pyllab.slow_paste_cl(c1._cl, c2._cl,tau)

def copy_vector_to_params_cl(cl c, vector):
    check_size(vector,c.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_cl(c._cl,<float*>&v[0])

def copy_params_to_vector_cl(cl c):
    vector = np.arange(c.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_cl(c._cl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_cl(cl c, vector):
    check_size(vector,c.get_array_size_weights())
    cdef float[:] v = vector_is_valid(vector)
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_cl(c._cl,<float*>&v[0])

def copy_weights_to_vector_cl(cl c):
    vector = np.arange(c.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_cl(c._cl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_cl(cl c, vector):
    check_size(vector,c.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_cl(c._cl,<float*>&v[0])

def copy_scores_to_vector_cl(cl c):
    vector = np.arange(c.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_cl(c._cl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_derivative_params_cl(cl c, vector):
    check_size(vector,c.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if c._does_have_arrays and c._does_have_learning_parameters and not c._is_only_for_feedforward:
        Pyllab.memcopy_vector_to_derivative_params_cl(c._cl,<float*>&v[0])

def copy_derivative_params_to_vector_cl(cl c):
    vector = np.arange(c.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if c._does_have_arrays and c._does_have_learning_parameters and not c._is_only_for_feedforward:
        Pyllab.memcopy_derivative_params_to_vector_cl(c._cl,<float*>&v[0])
        return vector
    return None

def compare_score_cl(cl c1, cl c2, cl c_output):
    if c1._does_have_arrays and c1._does_have_learning_parameters and c2._does_have_arrays and c2._does_have_learning_parameters and c_output._does_have_arrays and c_output._does_have_learning_parameters: 
        Pyllab.compare_score_cl(c1._cl, c2._cl,c_output._cl)
    
def compare_score_cl_with_vector(cl c1, vector, cl c_output):
	check_size(vector,c1.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if c1._does_have_arrays and c1._does_have_learning_parameters and c_output._does_have_arrays and c_output._does_have_learning_parameters:
        Pyllab.compare_score_cl_with_vector(c1._cl, <float*>&v[0],c_output._cl)


