cimport Pyllab
from libc.stdlib cimport free, malloc
from cpython cimport array


N_NORMALIZATION = Pyllab.N_NORMALIZATION
BETA_NORMALIZATION = Pyllab.BETA_NORMALIZATION
ALPHA_NORMALIZATION = Pyllab.ALPHA_NORMALIZATION
K_NORMALIZATION = Pyllab.K_NORMALIZATION
NESTEROV = Pyllab.NESTEROV
ADAM = Pyllab.ADAM
RADAM = Pyllab.RADAM
DIFF_GRAD = Pyllab.DIFF_GRAD
ADAMOD = Pyllab.ADAMOD
FCLS = Pyllab.FCLS
CLS = Pyllab.CLS
RLS = Pyllab.RLS
BNS = Pyllab.BNS
LSTMS = Pyllab.LSTMS
TRANSFORMER_ENCODER = Pyllab.TRANSFORMER_ENCODER
TRANSFORMER_DECODER = Pyllab.TRANSFORMER_DECODER
TRANSFORMER = Pyllab.TRANSFORMER
MODEL = Pyllab.MODEL
RMODEL = Pyllab.RMODEL
ATTENTION = Pyllab.ATTENTION
MULTI_HEAD_ATTENTION = Pyllab.MULTI_HEAD_ATTENTION
L2_NORM_CONN = Pyllab.L2_NORM_CONN
VECTOR = Pyllab.VECTOR
NOTHING = Pyllab.NOTHING
TEMPORAL_ENCODING_MODEL = Pyllab.TEMPORAL_ENCODING_MODEL
NO_ACTIVATION = Pyllab.NO_ACTIVATION
SIGMOID = Pyllab.SIGMOID
RELU = Pyllab.RELU
SOFTMAX = Pyllab.SOFTMAX
TANH = Pyllab.TANH
LEAKY_RELU = Pyllab.LEAKY_RELU
ELU = Pyllab.ELU
NO_POOLING = Pyllab.NO_POOLING
MAX_POOLING = Pyllab.MAX_POOLING
AVARAGE_POOLING = Pyllab.AVARAGE_POOLING
NO_DROPOUT = Pyllab.NO_DROPOUT
DROPOUT = Pyllab.DROPOUT
DROPOUT_TEST = Pyllab.DROPOUT_TEST
NO_NORMALIZATION = Pyllab.NO_NORMALIZATION
LOCAL_RESPONSE_NORMALIZATION = Pyllab.LOCAL_RESPONSE_NORMALIZATION
BATCH_NORMALIZATION = Pyllab.BATCH_NORMALIZATION
GROUP_NORMALIZATION = Pyllab.GROUP_NORMALIZATION
LAYER_NORMALIZATION = Pyllab.LAYER_NORMALIZATION
SCALED_L2_NORMALIZATION = Pyllab.SCALED_L2_NORMALIZATION
COSINE_NORMALIZATION = Pyllab.COSINE_NORMALIZATION
BETA1_ADAM = Pyllab.BETA1_ADAM
BETA2_ADAM = Pyllab.BETA2_ADAM
BETA3_ADAMOD = Pyllab.BETA3_ADAMOD
EPSILON_ADAM = Pyllab.EPSILON_ADAM
EPSILON = Pyllab.EPSILON
RADAM_THRESHOLD = Pyllab.RADAM_THRESHOLD
NO_REGULARIZATION = Pyllab.NO_REGULARIZATION
L2_REGULARIZATION = Pyllab.L2_REGULARIZATION
NO_CONVOLUTION = Pyllab.NO_CONVOLUTION
CONVOLUTION  = Pyllab.CONVOLUTION 
TRANSPOSED_CONVOLUTION = Pyllab.TRANSPOSED_CONVOLUTION 
BATCH_NORMALIZATION_TRAINING_MODE = Pyllab.BATCH_NORMALIZATION_TRAINING_MODE 
BATCH_NORMALIZATION_FINAL_MODE  = Pyllab.BATCH_NORMALIZATION_FINAL_MODE 
STATEFUL = Pyllab.STATEFUL
STATELESS = Pyllab.STATELESS
LEAKY_RELU_THRESHOLD = Pyllab.LEAKY_RELU_THRESHOLD 
ELU_THRESHOLD = Pyllab.ELU_THRESHOLD
LSTM_RESIDUAL  = Pyllab.LSTM_RESIDUAL 
LSTM_NO_RESIDUAL = Pyllab.LSTM_NO_RESIDUAL
TRANSFORMER_RESIDUAL = Pyllab.TRANSFORMER_RESIDUAL
TRANSFORMER_NO_RESIDUAL  = Pyllab.TRANSFORMER_NO_RESIDUAL 
NO_SET  = Pyllab.NO_SET 
NO_LOSS = Pyllab.NO_LOSS
CROSS_ENTROPY_LOSS = Pyllab.CROSS_ENTROPY_LOSS
FOCAL_LOSS = Pyllab.FOCAL_LOSS
HUBER1_LOSS = Pyllab.HUBER1_LOSS
HUBER2_LOSS = Pyllab.HUBER2_LOSS
MSE_LOSS = Pyllab.MSE_LOSS
KL_DIVERGENCE_LOSS = Pyllab.KL_DIVERGENCE_LOSS
ENTROPY_LOSS = Pyllab.ENTROPY_LOSS
TOTAL_VARIATION_LOSS_2D = Pyllab.TOTAL_VARIATION_LOSS_2D
CONTRASTIVE_2D_LOSS = Pyllab.CONTRASTIVE_2D_LOSS
LOOK_AHEAD_ALPHA = Pyllab.LOOK_AHEAD_ALPHA
LOOK_AHEAD_K = Pyllab.LOOK_AHEAD_K
GRADIENT_DESCENT  = Pyllab.GRADIENT_DESCENT 
EDGE_POPUP = Pyllab.EDGE_POPUP
FULLY_FEED_FORWARD = Pyllab.FULLY_FEED_FORWARD
FREEZE_TRAINING = Pyllab.FREEZE_TRAINING
FREEZE_BIASES = Pyllab.FREEZE_BIASES
ONLY_FF = Pyllab.ONLY_FF
ONLY_DROPOUT = Pyllab.ONLY_DROPOUT
STANDARD_ATTENTION = Pyllab.STANDARD_ATTENTION
MASKED_ATTENTION = Pyllab.MASKED_ATTENTION
RUN_ONLY_DECODER = Pyllab.RUN_ONLY_DECODER
RUN_ONLY_ENCODER = Pyllab.RUN_ONLY_ENCODER
RUN_ALL_TRANSF = Pyllab.RUN_ALL_TRANSF
SORT_SWITCH_THRESHOLD = Pyllab.SORT_SWITCH_THRESHOLD
NO_ACTION  = Pyllab.NO_ACTION 
ADDITION = Pyllab.ADDITION
SUBTRACTION  = Pyllab.SUBTRACTION 
MULTIPLICATION = Pyllab.MULTIPLICATION
RESIZE = Pyllab.RESIZE
CONCATENATE = Pyllab.CONCATENATE
DIVISION = Pyllab.DIVISION
INVERSE = Pyllab.INVERSE
CHANGE_SIGN = Pyllab.CHANGE_SIGN
GET_MAX = Pyllab.GET_MAX
NO_CONCATENATE = Pyllab.NO_CONCATENATE
POSITIONAL_ENCODING = Pyllab.POSITIONAL_ENCODING
SPECIES_THERESHOLD = Pyllab.SPECIES_THERESHOLD
INITIAL_POPULATION = Pyllab.INITIAL_POPULATION
GENERATIONS = Pyllab.GENERATIONS
PERCENTAGE_SURVIVORS_PER_SPECIE = Pyllab.PERCENTAGE_SURVIVORS_PER_SPECIE
CONNECTION_MUTATION_RATE = Pyllab.CONNECTION_MUTATION_RATE
NEW_CONNECTION_ASSIGNMENT_RATE = Pyllab.NEW_CONNECTION_ASSIGNMENT_RATE
ADD_CONNECTION_BIG_SPECIE_RATE = Pyllab.ADD_CONNECTION_BIG_SPECIE_RATE
ADD_CONNECTION_SMALL_SPECIE_RATE = Pyllab.ADD_CONNECTION_SMALL_SPECIE_RATE
ADD_NODE_SPECIE_RATE = Pyllab.ADD_NODE_SPECIE_RATE
ACTIVATE_CONNECTION_RATE = Pyllab.ACTIVATE_CONNECTION_RATE
REMOVE_CONNECTION_RATE = Pyllab.REMOVE_CONNECTION_RATE
CHILDREN = Pyllab.CHILDREN
CROSSOVER_RATE = Pyllab.CROSSOVER_RATE
SAVING = Pyllab.SAVING
LIMITING_SPECIES = Pyllab.LIMITING_SPECIES
LIMITING_THRESHOLD = Pyllab.LIMITING_THRESHOLD
MAX_POPULATION = Pyllab.MAX_POPULATION
SAME_FITNESS_LIMIT = Pyllab.SAME_FITNESS_LIMIT
AGE_SIGNIFICANCE = Pyllab.AGE_SIGNIFICANCE


cdef from_float_to_array(float *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return array.array('f',lst)

cdef class bn:
    cdef Pyllab.bn* _bn 
    cdef int _batch_size
    cdef int _vector_input_dimension
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    
    def __cinit__(self,int batch_size,int vector_input_dimension, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
        
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

            if self._bn is NULL:
                raise MemoryError()
            if is_only_for_feedforward:
                self.make_it_only_for_ff()
        else:
            self._bn = Pyllab.batch_normalization_without_arrays(batch_size,vector_input_dimension)
    
        
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
        if not self._is_only_for_feedforward:
            self._is_only_for_feedforward = True
            Pyllab.make_the_bn_only_for_ff(<Pyllab.bn*>self._bn)
    
    def reset(self):
        if self._does_have_arrays:
            Pyllab.reset_bn(<Pyllab.bn*>self._bn)
    
    def clip(self, float threshold, float norm):
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_bns(&self._bn, 1, threshold, norm)
    
    


def paste_bn(bn b1, bn b2):
    if b1._does_have_arrays and b1._does_have_learning_parameters and b2._does_have_arrays and b2._does_have_learning_parameters:
        Pyllab.paste_bn(b1._bn,b2._bn)
    
def paste_bn_without_learning_parameters(bn b1, bn b2):
    if b1._does_have_arrays and b2._does_have_arrays:
        Pyllab.paste_bn_without_learning_parameters(b1._bn,b2._bn)

def slow_paste_bn(bn b1, bn b2, float tau):
    if b1._does_have_arrays and b1._does_have_learning_parameters and b2._does_have_arrays and b2._does_have_learning_parameters:
        Pyllab.slow_paste_bn(b1._bn, b2._bn,tau)

def copy_bn(bn b):
    cdef bn bb = bn(b._batch_size, b._vector_input_dimension)
    paste_bn(b,bb)
    return bb

def copy_bn_without_learning_parameters(bn b):
    cdef bn bb = bn(b._batch_size, b._vector_input_dimension,False)
    paste_bn_without_learning_parameters(b,bb)
    return bb

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
            if(does_have_learning_parameters):
                self._cl = Pyllab.convolutional(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)
            else:
                self._cl = Pyllab.convolutional_without_learning_parameters(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)

            if self._cl is NULL:
                raise MemoryError()
            if is_only_for_feedforward:
                self.make_it_only_for_ff()
        else:
            self._cl = Pyllab.convolutional_without_arrays(channels,input_rows, input_cols,kernel_rows,kernel_cols,n_kernels,stride1_rows,stride1_cols,padding1_rows,padding1_cols,stride2_rows,stride2_cols,padding2_rows,padding2_cols,pooling_rows,pooling_cols,normalization_flag,activation_flag,pooling_flag,group_norm_channels,convolutional_flag,training_mode,feed_forward_flag,layer)
    
        
    def __dealloc__(self):
        if self._cl is not NULL and self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.free_convolutional(self._cl)
        elif self._cl is not NULL and self._does_have_arrays:
            Pyllab.free_convolutional_without_learning_parameters(self._cl)
        elif self._cl is not NULL:
            Pyllab.free_convolutional_without_arrays(self._cl)
    
    def save(self,number_of_file):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.save_cl(self._cl, number_of_file)
        
    def get_size(self):
        if not self._is_only_for_feedforward:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_cls(self._cl)
            else:
                return Pyllab.size_of_cls_without_learning_parameters(self._cl)
        return 0
    
    def make_it_only_for_ff(self):
        if not self._is_only_for_feedforward:
            self._is_only_for_feedforward = True
            Pyllab.make_the_cl_only_for_ff(self._cl)
    
    def reset(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_cl(self._cl)
            else:
                Pyllab.reset_cl_without_learning_parameters(self._cl)
                
    def clip(self, float threshold, float norm):
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_cls(&self._cl, 1, threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
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
        if self._does_have_arrays:
            return Pyllab.count_weights_cl(self._cl)


def paste_cl(cl c1, cl c2):
    if c1._does_have_arrays and c1._does_have_learning_parameters and c2._does_have_arrays and c2._does_have_learning_parameters:
        Pyllab.paste_cl(c1._cl,c2._cl)
    
def paste_cl_without_learning_parameters(cl c1, cl c2):
    if c1._does_have_arrays and c2._does_have_arrays:
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
    if c1._does_have_arrays and c1._does_have_learning_parameters and c2._does_have_arrays and c2._does_have_learning_parameters:
        Pyllab.slow_paste_cl(c1._cl, c2._cl,tau)

def copy_vector_to_params_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_cl(c._cl,vector.data.as_floats)

def copy_params_to_vector_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_cl(c._cl,vector.data.as_floats)

def copy_vector_to_weights_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_cl(c._cl,vector.data.as_floats)

def copy_weights_to_vector_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_cl(c._cl,vector.data.as_floats)

def copy_vector_to_scores_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_cl(c._cl,vector.data.as_floats)

def copy_scores_to_vector_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_cl(c._cl,vector.data.as_floats)

def copy_vector_to_derivative_params_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters and not c._is_only_for_feedforward:
        Pyllab.memcopy_vector_to_derivative_params_cl(c._cl,vector.data.as_floats)

def copy_derivative_params_to_vector_cl(cl c, array.array vector):
    if c._does_have_arrays and c._does_have_learning_parameters and not c._is_only_for_feedforward:
        Pyllab.memcopy_derivative_params_to_vector_cl(c._cl,vector.data.as_floats)

def compare_score_cl(cl c1, cl c2, cl c_output):
    if c1._does_have_arrays and c1._does_have_learning_parameters and c2._does_have_arrays and c2._does_have_learning_parameters and c_output._does_have_arrays and c_output._does_have_learning_parameters: 
        Pyllab.compare_score_cl(c1._cl, c2._cl,c_output._cl)
    
def compare_score_cl_with_vector(cl c1, array.array vector, cl c_output):
    if c1._does_have_arrays and c1._does_have_learning_parameters and c_output._does_have_arrays and c_output._does_have_learning_parameters:
        Pyllab.compare_score_cl_with_vector(c1._cl, vector.data.as_floats,c_output._cl)





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
            if(does_have_learning_parameters):
                self._fcl = Pyllab.fully_connected(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)
            else:
                self._fcl = Pyllab.fully_connected_without_learning_parameters(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)

            if self._fcl is NULL:
                raise MemoryError()
            if is_only_for_feedforward:
                self.make_it_only_for_ff()
        else:
            self._fcl = Pyllab.fully_connected_without_arrays(input, output, layer, dropout_flag, activation_flag, dropout_threshold, n_groups, normalization_flag, training_mode, feed_forward_flag)
    
        
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
        if not self._is_only_for_feedforward:
            self._is_only_for_feedforward = True
            Pyllab.make_the_fcl_only_for_ff(self._fcl)
    
    def reset(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_fcl(self._fcl)
            else:
                Pyllab.reset_fcl_without_learning_parameters(self._fcl)
                
    def clip(self, float threshold, float norm):
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_fcls(&self._fcl,1, threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
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
        if self._does_have_arrays:
            return Pyllab.count_weights_fcl(self._fcl)


def paste_fcl(fcl f1, fcl f2):
    if f1._does_have_arrays and f1._does_have_learning_parameters and f2._does_have_arrays and f2._does_have_learning_parameters:
        Pyllab.paste_fcl(f1._fcl,f2._fcl)
    
def paste_fcl_without_learning_parameters(fcl f1, fcl f2):
    if f1._does_have_arrays and f2._does_have_arrays:
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
    if f1._does_have_arrays and f1._does_have_learning_parameters and f2._does_have_arrays and f2._does_have_learning_parameters:
        Pyllab.slow_paste_fcl(f1._fcl, f2._fcl,tau)

def copy_vector_to_params_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params(f._fcl,vector.data.as_floats)

def copy_params_to_vector_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector(f._fcl,vector.data.as_floats)

def copy_vector_to_weights_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights(f._fcl,vector.data.as_floats)

def copy_weights_to_vector_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector(f._fcl,vector.data.as_floats)

def copy_vector_to_scores_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores(f._fcl,vector.data.as_floats)

def copy_scores_to_vector_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector(f._fcl,vector.data.as_floats)

def copy_vector_to_derivative_params_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters and not f._is_only_for_feedforward:
        Pyllab.memcopy_vector_to_derivative_params(f._fcl,vector.data.as_floats)

def copy_derivative_params_to_vector_fcl(fcl f, array.array vector):
    if f._does_have_arrays and f._does_have_learning_parameters and not f._is_only_for_feedforward:
        Pyllab.memcopy_derivative_params_to_vector(f._fcl,vector.data.as_floats)

def compare_score_fcl(fcl f1, fcl f2, fcl f_output):
    if f1._does_have_arrays and f1._does_have_learning_parameters and f2._does_have_arrays and f2._does_have_learning_parameters and f_output._does_have_arrays and f_output._does_have_learning_parameters: 
        Pyllab.compare_score_fcl(f1._fcl, f2._fcl,f_output._fcl)
    
def compare_score_fcl_with_vector(fcl f1, array.array vector, fcl f_output):
    if f1._does_have_arrays and f1._does_have_learning_parameters and f_output._does_have_arrays and f_output._does_have_learning_parameters:
        Pyllab.compare_score_fcl_with_vector(f1._fcl, vector.data.as_floats,f_output._fcl)



def dot1D(array.array input1, array.array input2, array.array output, int size):
    Pyllab.dot1D(input1.data.as_floats,input2.data.as_floats,output.data.as_floats,size)

def sum1D(array.array input1, array.array input2, array.array output, int size):
    Pyllab.sum1D(input1.data.as_floats,input2.data.as_floats,output.data.as_floats,size)


def mul_value(array.array input1,float value, array.array output, int size):
    Pyllab.mul_value(input1.data.as_floats,value,output.data.as_floats,size)

def clip_vector(array.array vector, float minimum, float maximum, int dimension):
    Pyllab.clip_vector(vector.data.as_floats,minimum,maximum,dimension)
    
    
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

def copy_vector_to_params_rl(rl r, array.array vector):
    Pyllab.memcopy_vector_to_params_rl(r._rl,vector.data.as_floats)

def copy_params_to_vector_rl(rl r, array.array vector):
    Pyllab.memcopy_params_to_vector_rl(r._rl,vector.data.as_floats)

def copy_vector_to_weights_rl(rl r, array.array vector):
    Pyllab.memcopy_vector_to_weights_rl(r._rl,vector.data.as_floats)

def copy_weights_to_vector_rl(rl r, array.array vector):
    Pyllab.memcopy_weights_to_vector_rl(r._rl,vector.data.as_floats)

def copy_vector_to_scores_rl(rl r, array.array vector):
    Pyllab.memcopy_vector_to_scores_rl(r._rl,vector.data.as_floats)

def copy_scores_to_vector_rl(rl r, array.array vector):
    Pyllab.memcopy_scores_to_vector_rl(r._rl,vector.data.as_floats)

def copy_vector_to_derivative_params_rl(rl r, array.array vector):
    Pyllab.memcopy_vector_to_derivative_params_rl(r._rl,vector.data.as_floats)

def copy_derivative_params_to_vector_rl(rl r, array.array vector):
    Pyllab.memcopy_derivative_params_to_vector_rl(r._rl,vector.data.as_floats)

def compare_score_rl(rl r1, rl r2, rl r_output):
    Pyllab.compare_score_rl(r1._rl, r2._rl,r_output._rl)
    
def compare_score_rl_with_vector(rl r1, array.array vector, rl r_output):
    Pyllab.compare_score_rl_with_vector(r1._rl, vector.data.as_floats,r_output._rl)  


cdef class model:
    cdef Pyllab.model* _model
    cdef fcl fcls
    cdef cl cls
    cdef rl rls
    cdef Pyllab.fcl** _fcls
    cdef Pyllab.cl** _cls
    cdef Pyllab.rl** _rls
    cdef int _n_fcl
    cdef int _n_cl
    cdef int _n_rl
    cdef int _layers
    cdef bint _does_have_learning_parameters
    cdef bint _does_have_arrays
    cdef bint _is_only_for_feedforward
    def __cinit__(self,int layers, int n_fcl, int n_cl, int n_rl, list fcls, list cls, list rls, bint does_have_learning_parameters = True, bint does_have_arrays = True, bint is_only_for_feedforward = False):
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
        for i in range(0,self._n_fcl):
            self.fcls[i].make_it_only_for_ff()
        for i in range(0,self._n_cl):
            self.cls[i].make_it_only_for_ff()
        for i in range(0,self._n_rl):
            self.rls[i].make_it_only_for_ff()
    
    def reset(self):
        for i in range(0,self._n_fcl):
            self.fcls[i].reset()
        for i in range(0,self._n_cl):
            self.cls[i].reset()
        for i in range(0,self._n_rl):
            self.rls[i].reset()
                
    def clip(self, float threshold):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.clipping_gradient(self._model, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
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
    
    def set_model_error(self, int error_flag, int output_dimension, float threshold1 = 0, float threshold2 = 0, float gamma = 0, array.array alpha = None):
        if alpha == None:
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,NULL,output_dimension)
        else:
            Pyllab.set_model_error(self._model,error_flag,threshold1,threshold2,gamma,alpha.data.as_floats,output_dimension)
    
    def feed_forward(self, int tensor_depth, int tensor_i, int tensor_j, array.array inputs):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.model_tensor_input_ff(self._model,tensor_depth,tensor_i,tensor_j,inputs.data.as_floats)
    
    def back_propagation(self, int tensor_depth, int tensor_i, int tensor_j, array.array inputs, array.array error, int error_dimension):
        cdef float* ret = Pyllab.model_tensor_input_bp(self._model,tensor_depth,tensor_i,tensor_j,inputs.data.as_floats,inputs.data.as_floats,error_dimension)
        return from_float_to_array(ret,tensor_depth*tensor_i*tensor_j)
        
    def ff_error_bp(self, int tensor_depth, int tensor_i, int tensor_j, array.array inputs, array.array outputs):
        cdef float* ret = Pyllab.ff_error_bp_model_once(self._model,tensor_depth,tensor_i,tensor_j,inputs.data.as_floats,outputs.data.as_floats)
        return from_float_to_array(ret,tensor_depth*tensor_i*tensor_j)
        
    
    def set_training_edge_popup(self, float k_percentage):
        Pyllab.set_model_training_edge_popup(self._model,k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        Pyllab.reinitialize_weights_according_to_scores_model(self._model,percentage,goodness)
    
    def set_training_gd(self):
        Pyllab.set_model_training_gd(self._model)
    
    def reset_edge_popup_d_params(self):
        Pyllab.reset_edge_popup_d_model(self._model)
    
def paste_model(model m1, model m2):
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters:
        Pyllab.paste_model(m1._model,m2._model)
    
def paste_model_without_learning_parameters(model m1, model m2):
    if m1._does_have_arrays and m2._does_have_arrays:
        Pyllab.paste_model_without_learning_parameters(m1._model,m2._model)

def copy_model(model m):
    l1 = []
    for i in range(0,len(m.fcls)):
        l1.append(copy_fcl(m.fcls[i]))
    l2 = []
    for i in range(0,len(m.cls)):
        l2.append(copy_cl(m.cls[i]))
    l3 = []
    for i in range(0,len(m.rls)):
        l3.append(copy_rl(m.rls[i]))
    cdef model mm = model(m._n_fcl,m._n_cl,m._n_rl,l1,l2,l3)
    paste_model(m,mm)
    return mm

def copy_model_without_learning_parameters(model m):
    l1 = []
    for i in range(0,len(m.fcls)):
        l1.append(copy_fcl_without_learning_parameters(m.fcls[i]))
    l2 = []
    for i in range(0,len(m.cls)):
        l2.append(copy_cl_without_learning_parameters(m.cls[i]))
    l3 = []
    for i in range(0,len(m.rls)):
        l3.append(copy_rl_without_learning_parameters(m.rls[i]))
    cdef model mm = model(m._n_fcl,m._n_cl,m._n_rl,l1,l2,l3)
    paste_model_without_learning_parameters(m,mm)
    return mm

def slow_paste_model(model m1, model m2, float tau):
    Pyllab.slow_paste_model(m1._model, m2._model,tau)

def copy_vector_to_params_model(model m, array.array vector):
    Pyllab.memcopy_vector_to_params_model(m._model,vector.data.as_floats)

def copy_params_to_vector_model(model m, array.array vector):
    Pyllab.memcopy_params_to_vector_model(m._model,vector.data.as_floats)

def copy_vector_to_weights_model(model m, array.array vector):
    Pyllab.memcopy_vector_to_weights_model(m._model,vector.data.as_floats)

def copy_weights_to_vector_model(model m, array.array vector):
    Pyllab.memcopy_weights_to_vector_model(m._model,vector.data.as_floats)

def copy_vector_to_scores_model(model m, array.array vector):
    Pyllab.memcopy_vector_to_scores_model(m._model,vector.data.as_floats)

def copy_scores_to_vector_model(model m, array.array vector):
    Pyllab.memcopy_scores_to_vector_model(m._model,vector.data.as_floats)

def copy_vector_to_derivative_params_model(model m, array.array vector):
    Pyllab.memcopy_vector_to_derivative_params_model(m._model,vector.data.as_floats)

def copy_derivative_params_to_vector_model(model m, array.array vector):
    Pyllab.memcopy_derivative_params_to_vector_model(m._model,vector.data.as_floats)

def compare_score_model(model m1, model m2, model m_output):
    Pyllab.compare_score_model(m1._model, m2._model,m_output._model)
    
def compare_score_model_with_vector(model m1, array.array vector, model m_output):
    Pyllab.compare_score_model_with_vector(m1._model, vector.data.as_floats,m_output._model)
    
def sum_score_model(model m1,model m2, model m_output):
    Pyllab.sum_score_model(m1._model,m2._model,m_output._model)


