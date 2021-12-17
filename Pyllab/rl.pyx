cimport Pyllab
from Pyllab import *

cdef class rl:
    cdef Pyllab.rl* _rl
    cdef Pyllab.cl** _c_cls
    cdef int _channels
    cdef int _input_rows
    cdef int _input_cols
    cdef int _n_cl
    cdef list _cls
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    
    def __cinit__(self,int channels, int input_rows, int input_cols, int n_cl, list cls, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        self._channels = channels
        self._input_rows = input_rows
        self._input_cols = input_cols
        self._n_cl = n_cl
        self._cls = cls
        self._c_cls = <Pyllab.cl**> malloc(len(cls) * sizeof(Pyllab.cl*))
        self._does_have_arrays = does_have_arrays
        self._does_have_learning_parameters = does_have_learning_parameters
        self._is_only_for_feedforward = is_only_for_feedforward
        for i in range(len(cls)):
            self._c_cls[i] = <Pyllab.cl*>cls[i]._cl
        
        self._rl = Pyllab.residual(channels,input_rows,input_cols,n_cl, self._c_cls)
            
        if self._rl is NULL:
            raise MemoryError()
        
    def __dealloc__(self):
        if self._rl is not NULL:
            if self._does_have_arrays:
                if self._does_have_learning_parameters:
                    Pyllab.free_residual(self._rl)
                else:
                    Pyllab.free_residual_without_learning_parameters(self._rl)
            else:
                Pyllab.free_residual_without_arrays(self._rl)
                
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_rl(self._rl, number_of_file)
    
    def reset(self):
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.reset_rl(self._rl)
        elif not self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.reset_rl_without_learning_parameters(self._rl)
                
    def clip(self, float threshold, float norm):
         if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_rls(&self._rl,1,threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_rl(self._rl,threshold, epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_rl(self._rl)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights_rl(self._rl)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_rl(self._rl)
    
    def set_biases_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_residual_biases_to_zero(self._rl)
    
    def set_unused_weights_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            for i in range(len(self._cls)):
                self._cls[i].set_unused_weights_to_zero()
        
    def reset_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_score_rl(self._rl)
            
    def set_low_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_low_score_rl(self._rl)
    def make_it_only_for_ff(self):
        if not self.is_only_for_feedforward:
            for i in range(0,self._n_cl):
                self._cls[i].make_it_only_for_ff()
         
def paste_rl(rl r1, rl r2):
    if r1.does_have_arrays and r1.does_have_learning_parameters and r2.does_have_arrays and r2.does_have_learning_parameters:
        Pyllab.paste_rl(r1._rl,r2._rl)
    
def paste_rl_without_learning_parameters(rl r1, rl r2):
    if r1.does_have_arrays and r2.does_have_arrays:
        Pyllab.paste_rl_without_learning_parameters(r1._rl,r2._rl)

def copy_rl(rl r):
    l = []
    for i in range(0,len(r._cls)):
        l.append(copy_cl(r._cls[i]))
    cdef rl rr = rl(r._channels,r._input_rows,r._input_cols,r._n_cl,l)
    paste_rl(r,rr)
    return rr

def copy_rl_without_learning_parameters(rl r):
    l = []
    for i in range(0,len(r._cls)):
        l.append(copy_cl(r._cls[i]))
    cdef rl rr = rl(r._channels,r._input_rows,r._input_cols,r._n_cl,l)
    paste_rl_without_learning_parameters(r,rr)
    return rr

def slow_paste_rl(rl r1, rl r2, float tau):
    if len(r1._cls) != len(r2._cls):
        return
    for i in range(0,len(r1._cls)):
        slow_paste_cl(r1._cls[i],r2._cls[i])

def copy_vector_to_params_rl(rl r, vector):
    check_size(vector,r.get_array_size_params())
    cdef array.array v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_rl(r._rl,<float*>v.data.as_floats)

def copy_params_to_vector_rl(rl r, vector):
    v = np.arange(r.get_array_size_params(),dtype=np.dtype("f"))
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_rl(r._rl,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_weights_rl(rl r, vector):
    check_size(vector,r.get_array_size_weights())
    cdef array.array v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_rl(r._rl,<float*>v.data.as_floats)

def copy_weights_to_vector_rl(rl r, vector):
    v = np.arange(r.get_array_size_weights(),dtype=np.dtype("f"))
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_rl(r._rl,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_scores_rl(rl r, vector):
    check_size(vector,r.get_array_size_scores())
    cdef array.array v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_rl(r._rl,<float*>v.data.as_floats)

def copy_scores_to_vector_rl(rl r, vector):
    v = np.arange(r.get_array_size_scores(),dtype=np.dtype("f"))
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_rl(r._rl,<float*>v.data.as_floats)
        return v
    return None

def copy_vector_to_derivative_params_rl(rl r, vector):
    check_size(vector,r.get_array_size_params())
    cdef array.array v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_derivative_params_rl(r._rl,<float*>v.data.as_floats)

def copy_derivative_params_to_vector_rl(rl r, vector):
    v = np.arange(r.get_array_size_scores(),dtype=np.dtype("f"))
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_derivative_params_to_vector_rl(r._rl,<float*>v.data.as_floats)
        return v
    return None

def compare_score_rl(rl r1, rl r2, rl r_output):
    Pyllab.compare_score_rl(r1._rl, r2._rl,r_output._rl)
    
def compare_score_rl_with_vector(rl r1, vector, rl r_output):
    cdef array.array v = vector_is_valid(vector)
    Pyllab.compare_score_rl_with_vector(r1._rl, <float*>v.data.as_floats,r_output._rl)  
