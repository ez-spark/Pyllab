cimport Pyllab
from Pyllab import *


cdef class genome:
    cdef Pyllab.genome* _g
    cdef int _global_innovation_numb_nodes
    cdef int _global_innovation_numb_connections
    cdef int _input_size
    cdef int _output_size
    
    def __cinit__(self,filename, int global_innovation_numb_nodes, int global_innovation_numb_connections, int input_size, int output_size):
        check_int(global_innovation_numb_nodes)
        check_int(global_innovation_numb_connections)
        check_int(input_size)
        check_int(output_size)
        self._global_innovation_numb_connections = global_innovation_numb_connections
        self._global_innovation_numb_nodes = global_innovation_numb_nodes
        self._input_size = input_size
        self._output_size = output_size
        cdef char* ss = <char*>PyUnicode_AsUTF8(filename)
        self._g = Pyllab.load_genome(global_innovation_numb_connections,ss)
        if self._g is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        Pyllab.free_genome(self._g,self.global_inn_numb_connections)
    
    def ff(self, inputs):
        check_size(inputs,self.input_size)
        cdef float[:] i = vector_is_valid(inputs)
        cdef float* output = Pyllab.feed_forward(self._g, <float*>&i[0], self.global_inn_numb_nodes, self.global_inn_numb_connections)
        nd_output = from_float_to_ndarray(output,self.output_size)
        free(output)
        return nd_output
