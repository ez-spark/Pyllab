cimport Pyllab
from Pyllab import *

cdef class bn:
    cdef Pyllab.bn* _bn 
    cdef int _batch_size
    cdef int _vector_input_dimension
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    
    def __cinit__(self,int batch_size,int vector_input_dimension, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        check_int(batch_size)
        check_int(vector_input_dimension)
        self._batch_size = batch_size
        self._vector_input_dimension = vector_input_dimension
        self._does_have_learning_parameters = does_have_learning_parameters
        self._does_have_arrays = does_have_arrays
        self._is_only_for_feedforward = is_only_for_feedforward
        
        if does_have_arrays:
            if(does_have_learning_parameters):
                self._bn = Pyllab.batch_normalization(batch_size,vector_input_dimension)
            else:
                self._bn = Pyllab.batch_normalization_without_learning_parameters(batch_size,vector_input_dimension)

        else:
            self._bn = Pyllab.batch_normalization_without_arrays(batch_size,vector_input_dimension)
        
        if self._bn is NULL:
                raise MemoryError()
            if is_only_for_feedforward and does_have_arrays:
                Pyllab.make_the_bn_only_for_ff(<Pyllab.bn*>self._bn)
        
    def __dealloc__(self):
        if self._bn is not NULL and self._does_have_arrays and not self._is_only_for_feedforward:
            Pyllab.free_batch_normalization(self._bn)
        elif self._bn is not NULL and self._does_have_arrays:
            Pyllab.free_the_bn_only_for_ff(self._bn)
        elif self._bn is not NULL:
            free(self._bn)
    
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_bn(<Pyllab.bn*> self._bn, number_of_file)
        
    def get_size(self):
        if not self._is_only_for_feedforward:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_bn(<Pyllab.bn*>self._bn)
            else:
                return Pyllab.size_of_bn_without_learning_parameters(<Pyllab.bn*>self._bn)
        
    
    def make_it_only_for_ff(self):
        if not self._is_only_for_feedforward and self._does_have_arrays:
            self._is_only_for_feedforward = True
            Pyllab.make_the_bn_only_for_ff(<Pyllab.bn*>self._bn)
    
    def reset(self):
        if self._does_have_arrays:
            Pyllab.reset_bn(<Pyllab.bn*>self._bn)
    
    def clip(self, float threshold, float norm):
        check_float(threshold)
        check_float(norm)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_bns(&self._bn, 1, threshold, norm)
    
    


def paste_bn(bn b1, bn b2):
    if b1._does_have_arrays and b1._does_have_learning_parameters and b2._does_have_arrays and b2._does_have_learning_parameters
       and not b1._is_only_for_feedforward and not b2._is_only_for_feedforward:
        Pyllab.paste_bn(b1._bn,b2._bn)
    
def paste_bn_without_learning_parameters(bn b1, bn b2):
    if b1._does_have_arrays and b2._does_have_arrays and not b1._is_only_for_feedforward and not b2._is_only_for_feedforward:
        Pyllab.paste_bn_without_learning_parameters(b1._bn,b2._bn)

def slow_paste_bn(bn b1, bn b2, float tau):
    if b1._does_have_arrays and b1._does_have_learning_parameters and b2._does_have_arrays and b2._does_have_learning_parameters
       and not b1._is_only_for_feedforward and not b2._is_only_for_feedforward:
        Pyllab.slow_paste_bn(b1._bn, b2._bn,tau)

def copy_bn(bn b):
    cdef bn bb = bn(b._batch_size, b._vector_input_dimension,does_have_learning_parameters = b._does_have_learning_parameters, does_have_arrays = b._does_have_arrays
    is_only_for_feedforward = b._is_only_for_feedforward)
    paste_bn(b,bb)
    return bb

def copy_bn_without_learning_parameters(bn b):
    cdef bn bb = bn(b._batch_size, b._vector_input_dimension,does_have_learning_parameters = False, does_have_arrays = b._does_have_arrays
    is_only_for_feedforward = b._is_only_for_feedforward)
    paste_bn_without_learning_parameters(b,bb)
    return bb
