cimport Pyllab
from libc.stdlib cimport free, malloc, srand
from cpython cimport array
from itertools import chain
cimport numpy as npc
import numpy as np
from libc cimport stdint
from libc.stdio cimport printf
ctypedef stdint.uint64_t uint64_t
from libc.time cimport time

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
AVERAGE_POOLING = Pyllab.AVERAGE_POOLING
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
LR_NO_DECAY = Pyllab.LR_NO_DECAY
LR_CONSTANT_DECAY = Pyllab.LR_CONSTANT_DECAY
LR_TIME_BASED_DECAY = Pyllab.LR_TIME_BASED_DECAY
LR_STEP_DECAY = Pyllab.LR_STEP_DECAY
LR_ANNEALING_DECAY = Pyllab.LR_ANNEALING_DECAY

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

cdef extern from "Python.h":
    bytes PyUnicode_Decode(char* c, Py_ssize_t size, const char *encoding, const char *errors)


def check_size(vector, dim):
    if type(vector) == np.ndarray:
        shape = vector.shape
        n = 1
        for i in shape:
            n*=i
        if n != dim:
            print("Error: the dimension of one of your arrays is not correct!")
            exit(1)
    elif type(vector) == list:
        v = vector
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        if len(v) != dim:
            print("Error: the dimension of one of your arrays is not correct!")
            exit(1)
    else:
        print("Error: one of your vectors is not a list, neither a np.array!")
        exit(1)

#if is not multi dimensional there is not good
def get_first_dimension_size(vector):
    v = vector
    if type(v) == list:
        v = np.array(v, dtype=np.float32)
    
    if type(v) == np.ndarray:
        if v.ndim != 2:
            print("your array must be multi dimensional! d 2")
            exit(1)
        else:
            return v.shape[0]
    else:
        print("the input is not a np.array!")
        exit(1)

def get_randomness():
    srand(time(NULL))     

def get_dict_from_model_setup_file(filename):
    try:
        d = {}
        f = open(filename,"r")
        l = f.read().split('\n')
        f.close()
        if len(l)<2:
            return None
        d['layers'] = l[0].strip().split(";")[:-1]
        data = []
        temp = {}
        for i in range(1,len(l),2):
            if l[i].strip().split(';')[0] != '':
                temp = {}
                temp[l[i].strip().split(';')[0]] = l[i+1].strip().split(';')[:-1]
                for j in range(len(temp[l[i].strip().split(';')[0]])):
                    t = temp[l[i].strip().split(';')[0]][j]
                    print(t)
                    if '.' in t:
                        temp[l[i].strip().split(';')[0]][j] = float(t)
                    else:
                        temp[l[i].strip().split(';')[0]][j] = int(t)
                    print(temp)
                data.append(temp)
        d['data'] = data
        return d
    except:
        return None

def get_dict_from_model_setup_string(s):
    try:
        d = {}
        l = s.split('\n')
        if len(l)<2:
            return None
        d['layers'] = l[0].strip().split(";")[:-1]
        data = []
        temp = {}
        for i in range(1,len(l),2):
            if l[i].strip().split(';')[0] != '':
                temp = {}
                temp[l[i].strip().split(';')[0]] = l[i+1].strip().split(';')[:-1]
                for j in range(len(temp[l[i].strip().split(';')[0]])):
                    temp[l[i].strip().split(';')[0]][j] = int(temp[l[i].strip().split(';')[0]][j])
                data.append(temp)
        d['data'] = data
        return d
    except:
        return None
        

def get_dict_from_dueling_categorical_dqn_setup_file(filename):
    try:
        d = {}
        f = open(filename,"r")
        l = f.read().split('\n')
        f.close()
        if len(l)<2:
            return None
        shared = 0
        v_hid = 0
        v_lin = 0
        a_hid = 0
        a_lin = 0
        for i in l:
            shared+=i.count('shared_hidden_layers')
            v_hid+=i.count('v_hidden_layers')
            v_lin+=i.count('v_linear_last_layer')
            a_hid+=i.count('a_hidden_layers')
            a_lin+=i.count('a_linear_last_layer')
        if shared != 1 or v_hid != 1 or v_lin != 1 or a_hid != 1 or a_lin != 1:
            return None
        d['shared_hidden_layers'] = ''
        d['v_hidden_layers'] = ''
        d['v_linear_last_layer'] = ''
        d['a_hidden_layers'] = ''
        d['a_linear_last_layer'] = ''
        for i in range(0,len(l)):
            if 'shared_hidden_layers' in l[i]:
                if i == len(l)-1:
                    return None
                for j in range(i+1,len(l)):
                    if j == len(l)-1:
                        return None
                    if 'v_linear_last_layer' in l[j] or 'a_hidden_layers' in l[j] or 'a_linear_last_layer' in l[j]:
                        return None
                    if 'v_hidden_layers' in l[j]:
                        break
                    d['shared_hidden_layers']+=l[j]+'\n'
            elif 'v_hidden_layers' in l[i]:
                if i == len(l)-1:
                    return None
                for j in range(i+1,len(l)):
                    if j == len(l)-1:
                        return None
                    if 'a_hidden_layers' in l[j] or 'a_linear_last_layer' in l[j]:
                        return None
                    if 'v_linear_last_layer' in l[j]:
                        break
                    d['v_hidden_layers']+=l[j]+'\n'
            elif 'v_linear_last_layer' in l[i]:
                if i == len(l)-1:
                    return None
                for j in range(i+1,len(l)):
                    if j == len(l)-1:
                        return None
                    if 'a_linear_last_layer' in l[j]:
                        return None
                    if 'a_hidden_layers' in l[j]:
                        break
                    d['v_linear_last_layer']+=l[j]+'\n'
            elif 'a_hidden_layers' in l[i]:
                if i == len(l)-1:
                    return None
                for j in range(i+1,len(l)):
                    if j == len(l)-1:
                        return None
                    if 'a_linear_last_layer' in l[j]:
                        break
                    d['a_hidden_layers']+=l[j]+'\n'
            elif 'a_linear_last_layer' in l[i]:
                if i == len(l)-1:
                    return None
                for j in range(i+1,len(l)):
                    d['a_linear_last_layer']+=l[j]+'\n'
        if d['shared_hidden_layers'] == '':
            return None
        if d['v_hidden_layers'] == '':
            return None
        if d['v_linear_last_layer'] == '':
            return None
        if d['a_hidden_layers'] == '':
            return None
        if d['a_linear_last_layer'] == '':
            return None
        d_model = get_dict_from_model_setup_string(d['shared_hidden_layers'])
        if d_model == None:
            return None
        d['shared_hidden_layers'] = d_model
        d_model = get_dict_from_model_setup_string(d['v_hidden_layers'])
        if d_model == None:
            return None
        d['v_hidden_layers'] = d_model
        d_model = get_dict_from_model_setup_string(d['v_linear_last_layer'])
        if d_model == None:
            return None
        d['v_linear_last_layer'] = d_model
        d_model = get_dict_from_model_setup_string(d['a_hidden_layers'])
        if d_model == None:
            return None
        d['a_hidden_layers'] = d_model
        d_model = get_dict_from_model_setup_string(d['a_linear_last_layer'])
        if d_model == None:
            return None
        d['a_linear_last_layer'] = d_model
        return d
        
    except:
        return None

def dict_to_pass_to_dueling_categorical_dqn_is_good(d):
    try:
        l = list(d.keys())
        if len(l) != 5:
            return False
        if 'shared_hidden_layers' not in l or 'v_hidden_layers' not in l or 'v_linear_last_layer' not in l or 'a_hidden_layers' not in l or 'a_linear_last_layer' not in l:
            return False
        for i in d:
            if not dict_to_pass_to_model_is_good(d[i]):
                return False
        return True
    except:
        return False            
def dict_to_pass_to_model_is_good(d):
    # we wrap the dict to avoid overflow or negatives values
    try:
        keys = list(d.keys())
        n_layers = 0
        convolutional = 0
        rconvolutional = 0
        fcl = 0
        residual = 0
        layers_list = []
        residuals_dict = {}
        kk = 0
        c_c = 0
        r_c = 0
        f_c = 0
        # the dictionary has 2 keys: layers and data
        for i in keys:
            if i=='layers':
                # layers has as values a list of [convolutional,fully-connected,rconvolutional,residual,...]
                # every word that no match the words written above gives an error
                if type(d[i]) != list or len(d[i]) == 0:
                    print("Error: the key of layers should be a list")
                    return False
                for j in d[i]:
                    if j == 'convolutional':
                        n_layers+=1
                        convolutional+=1
                    elif j == 'rconvolutional':
                        n_layers+=1
                        rconvolutional+=1
                    elif j == 'fully-connected':
                        n_layers+=1
                        fcl+=1
                    elif j == 'residual':
                        residual+=1
            elif i == 'data':
                #the data has as values a list of dict [{...:...},{...:...},...]
                for k in range(len(d[i])):
                    kk+=1
                    for ii in d[i][k].keys():
                        # each dict has just 1 key:
                        # either fully-connected, or convolutional, or rconvolutional
                        # all other words gives an error
                        # the values of fully connected is a list of 10 ints
                        # for convolutional 24 ints
                        # for rconvolutional 26 ints
                        #that's it
                        if ii == 'fully-connected':
                            if type(d[i][k][ii]) != list or len(d[i][k][ii]) != 10:
                                print("Error: either you value for a fully-connected is not a list, or its length is not 10!")
                                return False
                            for j in range(0,len(d[i][k][ii])):
                                if type(d[i][k][ii][j]) != int and j != 5:
                                    print("Error, some values in the list of the value of a fully-connected is not an int")
                                    return False
                                elif type(d[i][k][ii][j]) != float and type(d[i][k][ii][j]) != int and j == 5:
                                    print("Error, some value in the list of the value of a fully-connected is not a float neither a int")
                                    return False
                                if d[i][k][ii][j] < 0 or d[i][k][ii][j] >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False
                            if d[i][k][ii][2] in layers_list:
                                print("layer already exists!")
                                return False
                            f_c+=1
                            layers_list.append(d[i][k][ii][2])
                        elif ii == 'convolutional':
                            if type(d[i][k][ii]) != list or len(d[i][k][ii]) != 24:
                                print("Error: either you value for a convolutional is not a list, or its length is not 24!")
                                return False
                            for j in d[i][k][ii]:
                                if type(j) != int:
                                    print("Error, some values in the list of the value of a convolutional is not an int")
                                    return False
                                    
                                if j < 0 or j >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False
                            if d[i][k][ii][23] in layers_list:
                                print("layer already exists!")
                                return False
                            c_c+=1
                            layers_list.append(d[i][k][ii][23])
                        elif ii == 'rconvolutional':
                            if type(d[i][k][ii]) != list or len(d[i][k][ii]) != 26:
                                print("Error: either you value for a rconvolutional is not a list, or its length is not 26!")
                                return False
                            for j in d[i][k][ii]:
                                if type(j) != int:
                                    print("Error, some values in the list of the value of a rconvolutional is not an int")
                                    return False
                                    
                                if j < 0 or j >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False
                            if d[i][k][ii][23] in layers_list:
                                print("layer already exists!")
                                return False
                            r_c+=1
                            layers_list.append(d[i][k][ii][23])
                            if d[i][k][ii][25] in residuals_dict.keys():
                                if kk-1 not in residuals_dict[d[i][k][ii][25]]:
                                    print("convolutional layers inside the residual must be consecutive!")
                                    return False
                                residuals_dict[d[i][k][ii][25]].append(kk)
                            else:
                                residuals_dict[d[i][k][ii][25]] = [kk]
                            
                            if d[i][k][ii][24] > 6 and d[i][k][ii][24] < 0:
                                print("no recognized activation detected")
                                return False
                        else:
                            print("not recognized identifier")
                            return False
            else:
                print("not recognized identifier")
                return False
        if r_c != rconvolutional or f_c != fcl or c_c != convolutional:
            return False
        return True
    except:
        return False

def from_dict_to_str_model(dict d):
    if not dict_to_pass_to_model_is_good(d):
        return None
    s = ''
    first_line = d['layers']
    for i in first_line:
        s+=i+';'
    s+='\n'
    for i in d['data']:
        for j in i.keys():
            s+=j+';\n'
            for k in i[j]:
                s+=str(k)+';'
            s+='\n'
    return s

def from_dict_to_str_dueling_categorical_dqn(dict d):
    if not dict_to_pass_to_dueling_categorical_dqn_is_good(d):
        return None
    s = 'shared_hidden_layers;\n'
    d_model = from_dict_to_str_model(d['shared_hidden_layers'])
    if d_model == None:
        return None
    s+=d_model
    s += 'v_hidden_layers;\n'
    d_model=from_dict_to_str_model(d['v_hidden_layers'])
    if d_model == None:
        return None
    s+=d_model
    s += 'v_linear_last_layer;\n'
    d_model=from_dict_to_str_model(d['v_linear_last_layer'])
    if d_model == None:
        return None
    s+=d_model
    s += 'a_hidden_layers;\n'
    d_model=from_dict_to_str_model(d['a_hidden_layers'])
    if d_model == None:
        return None
    s+=d_model
    s += 'a_linear_last_layer;\n'
    d_model=from_dict_to_str_model(d['a_linear_last_layer'])
    if d_model == None:
        return None
    s+=d_model
    return s
    
    
cdef from_float_to_ndarray(float* ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return np.array(lst, dtype=np.float32)
    
cdef from_float_to_list(float *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

def vector_is_valid(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return np.ascontiguousarray(np.array(v,dtype=np.float32),dtype=np.float32)
    elif type(v) == np.ndarray:
        if v.ndim > 1:
            v = v.flatten()
        return np.ascontiguousarray(np.array(v,dtype=np.float32),dtype=np.float32) 
    
    print("Error: you vector is not a list neither a numpy.ndarray type")
    exit(1)

def vector_is_valid_int(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return np.ascontiguousarray(np.array(v,dtype=np.intc),dtype=np.intc)
    elif type(v) == np.ndarray:
        if v.ndim > 1:
            v = v.flatten()
        return np.ascontiguousarray(np.array(v,dtype=np.intc),dtype=np.intc) 
    
    print("Error: you vector is not a list neither a numpy.ndarray type")
    exit(1)

def check_int(int i):
    if i < 0 or i >= 2**31-1:
        print("Error, wrong input to pass")
        exit(1)
        
def check_float(float i):
    if i >= 2**31-1:
        print("Error, wrong input to pass")
        exit(1)


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

            if self._bn is NULL:
                raise MemoryError()
            if is_only_for_feedforward:
                Pyllab.make_the_bn_only_for_ff(<Pyllab.bn*>self._bn)
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
    cdef float[:] v = vector_is_valid(vector)
    if c1._does_have_arrays and c1._does_have_learning_parameters and c_output._does_have_arrays and c_output._does_have_learning_parameters:
        Pyllab.compare_score_cl_with_vector(c1._cl, <float*>&v[0],c_output._cl)


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
    cdef float[:] v = vector_is_valid(vector)
    if f1._does_have_arrays and f1._does_have_learning_parameters and f_output._does_have_arrays and f_output._does_have_learning_parameters:
        Pyllab.compare_score_fcl_with_vector(f1._fcl, <float*>&v[0],f_output._fcl)

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
        check_int(input_rows)
        check_int(input_cols)
        check_int(n_cl)
        self._channels = channels
        self._input_rows = input_rows
        self._input_cols = input_cols
        self._n_cl = n_cl
        self._cls = cls
        if len(cls) == 0:
            print("Error: you must pass a list of cls >= 1")
            exit(1)
        for i in range(len(cls)):
            if cls[i]._does_have_arrays != does_have_arrays or cls[i]._does_have_learning_parameters != does_have_learning_parameters or cls[i]._is_only_for_feedforward != is_only_for_feedforward:
                print("Error: your flags of the set of cls must be equal to the flag passed to this residual!")
                exit(1)
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
        check_float(threshold)
        check_float(norm)
        if self._does_have_arrays and self._does_have_learning_parameters and not self.is_only_for_feedforward:
            Pyllab.clip_rls(&self._rl,1,threshold, norm)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
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
    if r1.does_have_arrays and r1.does_have_learning_parameters and r2.does_have_arrays and r2.does_have_learning_parameters and not r1._is_only_for_feedforward and not r2._is_only_for_feedforward:
        Pyllab.paste_rl(r1._rl,r2._rl)
    
def paste_rl_without_learning_parameters(rl r1, rl r2):
    if r1.does_have_arrays and r2.does_have_arrays and not r1._is_only_for_feedforward and not r2._is_only_for_feedforward:
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
    Pyllab.slow_paste_rl(r1._rl,r2._rl,tau)

def copy_vector_to_params_rl(rl r, vector):
    check_size(vector,r.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_rl(r._rl,<float*>&v[0])

def copy_params_to_vector_rl(rl r):
    vector = np.arange(r.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_rl(r._rl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_rl(rl r, vector):
    check_size(vector,r.get_array_size_weights())
    cdef float[:] v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_rl(r._rl,<float*>&v[0])

def copy_weights_to_vector_rl(rl r):
    vector = np.arange(r.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_rl(r._rl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_rl(rl r, vector):
    check_size(vector,r.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_rl(r._rl,<float*>&v[0])

def copy_scores_to_vector_rl(rl r):
    vector = np.arange(r.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if r._does_have_arrays and r._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_rl(r._rl,<float*>&v[0])
        return vector
    return None

def copy_vector_to_derivative_params_rl(rl r, vector):
    check_size(vector,r.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if r._does_have_arrays and r._does_have_learning_parameters and not r._is_only_for_feedforward:
        Pyllab.memcopy_vector_to_derivative_params_rl(r._rl,<float*>&v[0])

def copy_derivative_params_to_vector_rl(rl r):
    vector = np.arange(r.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if r._does_have_arrays and r._does_have_learning_parameters and not r._is_only_for_feedforward:
        Pyllab.memcopy_derivative_params_to_vector_rl(r._rl,<float*>&v[0])
        return vector
    return None

def compare_score_rl(rl r1, rl r2, rl r_output):
    if r1._does_have_arrays and r1._does_have_learning_parameters and r2._does_have_arrays and r2._does_have_learning_parameters and r_output._does_have_arrays and r_output._does_have_learning_parameters:
        Pyllab.compare_score_rl(r1._rl, r2._rl,r_output._rl)
    
def compare_score_rl_with_vector(rl r1, vector, rl r_output):
    cdef float[:] v
    if r1._does_have_arrays and r1._does_have_learning_parameters and r_output._does_have_arrays and r_output._does_have_learning_parameters:
        v = vector_is_valid(vector)
        Pyllab.compare_score_rl_with_vector(r1._rl, <float*>&v[0],r_output._rl)  


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
        if filename != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            
            
            if mod == False:
                if not dict_to_pass_to_model_is_good(get_dict_from_model_setup_file(filename)):
                    print("Error: your setup file is not correct")
                    exit(1)
                if not self._does_have_arrays:
                    ss = <char*>PyUnicode_AsUTF8(filename)
                    self._model = Pyllab.parse_model_without_arrays_file(ss)
                    
                elif not self._does_have_learning_parameters:
                    ss = <char*>PyUnicode_AsUTF8(filename)
                    self._model = Pyllab.parse_model_without_learning_parameters_file(ss)
                    
                else:
                    ss = <char*>PyUnicode_AsUTF8(filename) 
                    self._model = Pyllab.parse_model_file(ss)
                    
            else:
                if not does_have_arrays or not does_have_learning_parameters:
                    print("Error: leave the default parameters for the flags!")
                    exit(1)
                ss = <char*>PyUnicode_AsUTF8(filename)
                self._model = Pyllab.load_model(ss)
                
                
            if self._model is NULL:
                raise MemoryError()
        
        elif d != None:
            self._does_have_arrays = does_have_arrays
            self._does_have_learning_parameters = does_have_learning_parameters
            self._is_only_for_feedforward = is_only_for_feedforward
            self._is_from_char = True
            s = from_dict_to_str_model(d)
            print(s)
            if s == None:
                print("Dict passed is not good, sorry")
                exit(1)
            if not self._does_have_arrays:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._model = Pyllab.parse_model_without_arrays_str(ss,len(s))
                
            elif not self._does_have_learning_parameters:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._model = Pyllab.parse_model_without_learning_parameters_str(ss,len(s))
            else:
                ss = <char*>PyUnicode_AsUTF8(s)
                self._model = Pyllab.parse_model_str(ss,len(s))
            
                
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
        
        if self._model is not NULL:
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
        if self._does_have_arrays and self._does_have_learning_parameters:
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
        if self._does_have_arrays:
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
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.reset_model_without_learning_parameters(self._models[i])
    def clip(self, float threshold):
        check_float(threshold)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.clipping_gradient(self._model, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_model(self._model,threshold,epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_model(self._model)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights_model(self._model)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_model(self._model)
    
    def set_biases_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_model_biases_to_zero(self._model)
    
    def set_unused_weights_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_model_unused_weights_to_zero(self._model)    
        
    def reset_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_score_model(self._model)
    def set_low_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_low_score_model(self._model)
    
    def get_number_of_weights(self):
        return Pyllab.count_weights(self._model)
    
    def set_model_error(self, int error_flag, int output_dimension, float threshold1 = 0, float threshold2 = 0, float gamma = 0, alpha = None):
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
            return from_float_to_ndarray(ret,tensor_depth*tensor_i*tensor_j)
        
    def ff_error_bp(self, int tensor_depth, int tensor_i, int tensor_j, inputs, outputs):
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
            return from_float_to_ndarray(ret,tensor_depth*tensor_i*tensor_j)
        
    
    def set_training_edge_popup(self, float k_percentage):
        check_float(k_percentage)
        if k_percentage > 1 or k_percentage <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_model_training_edge_popup(self._model,k_percentage)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_model_training_edge_popup(self._models[i],k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        check_float(percentage)
        check_float(goodness)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reinitialize_weights_according_to_scores_model(self._model,percentage,goodness)
    
    def set_training_gd(self):
        Pyllab.set_model_training_gd(self._model)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_model_training_gd(self._models[i])
    
    def reset_edge_popup_d_params(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_edge_popup_d_model(self._model)
    
    def output_layer(self):
        if self._does_have_arrays:
            return from_float_to_ndarray(Pyllab.get_output_layer_from_model(self._model),Pyllab.get_output_dimension_from_model(self._model))
    
    def get_output_dimension_from_model(self):
        n = Pyllab.get_output_dimension_from_model(self._model)
        return n
        
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
        b = Pyllab.get_beta3_from_model(self._model)
        return b
    
    def set_ith_layer_training_mode(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_model(self._model,ith,training_flag)
    
    def set_k_percentage_of_ith_layer(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_model(self._model, ith, k)

    def feed_forward_opt_multi_thread(self, int tensor_depth, int tensor_i, int tensor_j, inputs):
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
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(error_dimension)
        if error_dimension != self.get_output_dimension_from_model():
            print("Error: the error dimension doesn't match the output dimension of the model")
            exit(1)
        check_int(tensor_j*tensor_i*tensor_depth)
        if not self._is_multithread:
            return
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
        check_int(tensor_depth)
        check_int(tensor_i)
        check_int(tensor_j)
        check_int(error_dimension)
        if error_dimension != self.get_output_dimension_from_model():
            print("Error: the error dimension doesn't match the output dimension of the model")
            exit(1)
        check_int(tensor_j*tensor_i*tensor_depth)
        if not self._is_multithread:
            return
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
        check_int(index)
        if self._is_multithread:
            if index < self.threads:
                return from_float_to_ndarray(Pyllab.get_output_layer_from_model(self._models[index]),Pyllab.get_output_dimension_from_model(self._models[index]))
        return None
    
    def sum_models_partial_derivatives(self):
        if self._is_multithread:
            Pyllab.sum_models_partial_derivatives_multithread(self._models, self._model, self.threads, 0)
        
    def set_ith_layer_training_mode_models(self, int ith, int training_flag):
        check_int(training_flag)
        check_int(ith)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_ith_layer_training_mode_model(self._models[i],ith,training_flag)
    
    def set_k_percentage_of_ith_layer_models(self,int ith, float k):
        check_float(k)
        check_int(ith)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_k_percentage_of_ith_layer_model(self._models[i], ith, k)
            
            
            
       
def paste_model(model m1, model m2):
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.paste_model(m1._model,m2._model)
    
def paste_model_without_learning_parameters(model m1, model m2):
    if m1._does_have_arrays and m2._does_have_arrays and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.paste_model_without_learning_parameters(m1._model,m2._model)

def copy_model(model m):
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
        mod._is_from_char = True
        return mod
    
def copy_model_without_learning_parameters(model m):
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
        mod._is_from_char = True
        mod._does_have_learning_parameters = False
        return mod

def slow_paste_model(model m1, model m2, float tau):
    if m1._does_have_arrays and m1._does_have_learning_parameters and m2._does_have_arrays and m2._does_have_learning_parameters and not m1._is_only_for_feedforward and not m2._is_only_for_feedforward:
        Pyllab.slow_paste_model(m1._model, m2._model,tau)

def copy_vector_to_params_model(model m, vector):
    check_size(vector,m.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_model(m._model,<float*>&v[0])

def copy_params_to_vector_model(model m):
    vector = np.arange(m.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_model(model m, vector):
    check_size(vector,m.get_array_size_weights())
    cdef float[:]v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_model(m._model,<float*>&v[0])

def copy_weights_to_vector_model(model m):
    vector = np.arange(m.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_model(model m, vector):
    check_size(vector,m.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_model(m._model,<float*>&v[0])

def copy_scores_to_vector_model(model m):
    vector = np.arange(m.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def copy_vector_to_derivative_params_model(model m, vector):
    check_size(vector,m.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_derivative_params_model(m._model,<float*>&v[0])

def copy_derivative_params_to_vector_model(model m):
    vector = np.arange(m.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if m._does_have_arrays and m._does_have_learning_parameters:
        Pyllab.memcopy_derivative_params_to_vector_model(m._model,<float*>&v[0])
        return vector
    return None

def compare_score_model(model m1, model m2, model m_output):
    Pyllab.compare_score_model(m1._model, m2._model,m_output._model)
    
def compare_score_model_with_vector(model m1, vector, model m_output):
    cdef float[:] v = vector_is_valid(vector)
    Pyllab.compare_score_model_with_vector(m1._model, <float*>&v[0],m_output._model)
    
def sum_score_model(model m1,model m2, model m_output):
    Pyllab.sum_score_model(m1._model,m2._model,m_output._model)

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
        size = self.input_size
        return size
    def make_multi_thread(self, int threads):
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
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.free_dueling_categorical_dqn_without_learning_parameters(self._dqns[i])
            free(self._dqns)
        self._is_multithread = False
        self.threads = 1
    
    
    def save(self,int number_of_file, str directory):
        check_int(number_of_file)
        cdef char* ss
        if self._does_have_arrays and self._does_have_learning_parameters:
            ss = <char*>PyUnicode_AsUTF8(directory) 
            Pyllab.save_dueling_categorical_dqn_given_directory(self._dqn, number_of_file,ss)
    
    def get_size(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                return Pyllab.size_of_dueling_categorical_dqn(self._dqn)
            else:
                return Pyllab.size_of_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        return 0
    
    def make_it_only_for_ff(self):
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
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_dueling_categorical_dqn(self._dqn)
            else:
                Pyllab.reset_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        if self._is_multithread and self.reset_value == 1:
            self.reset_value = 0
            Pyllab.dueling_categorical_reset_without_learning_parameters_reset(self._dqns, self.threads)
    def reset_all(self):
        if self._does_have_arrays:
            if self._does_have_learning_parameters:
                Pyllab.reset_dueling_categorical_dqn(self._dqn)
            else:
                Pyllab.reset_dueling_categorical_dqn_without_learning_parameters(self._dqn)
        if self._is_multithread:
            self.reset_value = 0
            Pyllab.dueling_categorical_reset_without_learning_parameters_reset(self._dqns, self.threads)
    def clip(self, float threshold):
        check_float(threshold)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.dueling_categorical_dqn_clipping_gradient(self._dqn, threshold)
    
    def adaptive_clip(self, float threshold, float epsilon):
        check_float(threshold)
        check_float(epsilon)
        if self._does_have_arrays and self._does_have_learning_parameters and not self._is_only_for_feedforward:
            Pyllab.adaptive_gradient_clipping_dueling_categorical_dqn(self._dqn,threshold,epsilon)

    def get_array_size_params(self):
        return Pyllab.get_array_size_params_dueling_categorical_dqn(self._dqn)
    
    def get_array_size_weights(self):
        return Pyllab.get_array_size_weights_dueling_categorical_dqn(self._dqn)
        
    def get_array_size_scores(self):
        return Pyllab.get_array_size_scores_dueling_categorical_dqn(self._dqn)
    
    def set_biases_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_dueling_categorical_dqn_biases_to_zero(self._dqn)
    
    def set_unused_weights_to_zero(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_dueling_categorical_dqn_unused_weights_to_zero(self._dqn)    
        
    def reset_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reset_score_dueling_categorical_dqn(self._dqn)
    def set_low_scores(self):
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.set_low_score_dueling_categorical_dqn(self._dqn)
    
    def get_number_of_weights(self):
        return Pyllab.count_weights_dueling_categorical_dqn(self._dqn)
    
    def compute_probability_distribution(self, inputs , int input_size):
        check_int(input_size)
        cdef float[:] i
        if self._does_have_arrays and self._does_have_learning_parameters:
            check_size(inputs,input_size)
            i = vector_is_valid(inputs)
            Pyllab.compute_probability_distribution(<float*>&i[0], input_size, self._dqn)
    
    def compute_q_function(self, inputs, int input_size):
        self.compute_probability_distribution(inputs,input_size)
        if self._does_have_arrays and self._does_have_learning_parameters:
            return from_float_to_ndarray(Pyllab.compute_q_functions(self._dqn),self.action_size)
        return None
    def get_best_action(self,inputs,int input_size):
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
        check_float(k_percentage)
        if k_percentage > 1 or k_percentage <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_dueling_categorical_dqn_training_edge_popup(self._dqn,k_percentage)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_dueling_categorical_dqn_training_edge_popup(self._dqns[i],k_percentage)
    
    def reinitialize_weights_according_to_scores(self, float percentage, float goodness):
        check_float(percentage)
        check_float(goodness)
        if self._does_have_arrays and self._does_have_learning_parameters:
            Pyllab.reinitialize_weights_according_to_scores_dueling_categorical_dqn(self._dqn,percentage,goodness)
    
    def set_training_gd(self):
        Pyllab.set_dueling_categorical_dqn_training_gd(self._dqn)
        if self._is_multithread:
            for i in range(self.threads):
                Pyllab.set_dueling_categorical_dqn_training_gd(self._dqns[i])
        
    def set_beta1(self, float beta):
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta(self._dqn, beta, self.get_beta2())
    
    def set_beta2(self, float beta):
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta(self._dqn, self.get_beta1(), beta)
        
    def set_beta3(self, float beta):
        check_float(beta)
        Pyllab.set_dueling_categorical_dqn_beta_adamod(self._dqn, beta)
    
    def get_beta1(self):
        b = Pyllab.get_beta1_from_dueling_categorical_dqn(self._dqn)
        return b
    
    def get_beta2(self):
        b = Pyllab.get_beta2_from_dueling_categorical_dqn(self._dqn)
        return b
        
    def get_beta3(self):
        b = Pyllab.get_beta3_from_dueling_categorical_dqn(self._dqn)
        return b
    
    def set_ith_layer_training_mode_shared(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_shared(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_v_hid(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_v_hid(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_v_lin(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_v_lin(self._dqn,ith,training_flag)
        
    def set_ith_layer_training_mode_a_hid(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_a_hid(self._dqn,ith,training_flag)
    
    def set_ith_layer_training_mode_a_lin(self, int ith, int training_flag):
        check_int(ith)
        check_int(training_flag)
        Pyllab.set_ith_layer_training_mode_dueling_categorical_dqn_a_lin(self._dqn,ith,training_flag)
    
    def set_k_percentage_of_ith_layer_shared(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_shared(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_v_hid(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_hid(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_v_lin(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_v_lin(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_a_hid(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_hid(self._dqn, ith, k)

    def set_k_percentage_of_ith_layer_a_lin(self,int ith, float k):
        check_int(ith)
        check_float(k)
        if k > 1 or k <= 0:
            print("Error: the k percentage must be in (0,1]")
            exit(1)
        Pyllab.set_k_percentage_of_ith_layer_dueling_categorical_dqn_a_lin(self._dqn, ith, k)
        
    

    def train(self, states_t, states_t_1, rewards_t, actions_t, nonterminals_t_1, duelingCategoricalDQN dqn, float lambda_value):
        check_float(lambda_value)
        if dqn.threads != self.threads:
            return
        if not self._is_multithread:
            return
        if(get_first_dimension_size(states_t) != get_first_dimension_size(states_t_1)):
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
        if self._is_multithread:
            Pyllab.sum_dueling_categorical_dqn_partial_derivatives_multithread(self._dqns, self._dqn, self.threads, 0)
            
def paste_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2):
    if dqn1._does_have_arrays and dqn1._does_have_learning_parameters and dqn2._does_have_arrays and dqn2._does_have_learning_parameters and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.paste_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn)
    
def paste_dueling_categorical_dqn_without_learning_parameters(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2):
    if dqn1._does_have_arrays and dqn2._does_have_arrays and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.paste_dueling_categorical_dqn_without_learning_parameters(dqn1._dqn,dqn2._dqn)

def copy_dueling_categorical_dqn(duelingCategoricalDQN dqn):
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
    if dqn1._does_have_arrays and dqn1._does_have_learning_parameters and dqn2._does_have_arrays and dqn2._does_have_learning_parameters and not dqn1._is_only_for_feedforward and not dqn2._is_only_for_feedforward:
        Pyllab.slow_paste_dueling_categorical_dqn(dqn1._dqn, dqn2._dqn,tau)

def copy_vector_to_params_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    check_size(vector,dqn.get_array_size_params())
    cdef float[:] v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_params_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_params_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    vector = np.arange(dqn.get_array_size_params(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_params_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None

def copy_vector_to_weights_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    check_size(vector,dqn.get_array_size_weights())
    cdef float[:]v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_weights_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_weights_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    vector = np.arange(dqn.get_array_size_weights(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_weights_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None

def copy_vector_to_scores_dueling_categorical_dqn(duelingCategoricalDQN dqn, vector):
    check_size(vector,dqn.get_array_size_scores())
    cdef float[:] v = vector_is_valid(vector)
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_vector_to_scores_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])

def copy_scores_to_vector_dueling_categorical_dqn(duelingCategoricalDQN dqn):
    vector = np.arange(dqn.get_array_size_scores(),dtype=np.float32)
    cdef float[:] v = vector
    if dqn._does_have_arrays and dqn._does_have_learning_parameters:
        Pyllab.memcopy_scores_to_vector_dueling_categorical_dqn(dqn._dqn,<float*>&v[0])
        return vector
    return None
    
def compare_score_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2, duelingCategoricalDQN dqn_output):
    Pyllab.compare_score_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn,dqn_output._dqn)
    
def compare_score_dueling_categorical_dqn_with_vector(duelingCategoricalDQN dqn1, vector, duelingCategoricalDQN dqn_output):
    cdef float[:] v = vector_is_valid(vector)
    Pyllab.compare_score_dueling_categorical_dqn_with_vector(dqn1._dqn, <float*>&v[0],dqn_output._dqn)
    
def sum_score_dueling_categorical_dqn(duelingCategoricalDQN dqn1, duelingCategoricalDQN dqn2, duelingCategoricalDQN dqn_output):
    Pyllab.sum_score_dueling_categorical_dqn(dqn1._dqn,dqn2._dqn,dqn_output._dqn)
             
cdef class training:
    cdef float lr
    cdef float start_lr
    cdef float momentum
    cdef int batch_size
    cdef int gradient_descent_flag
    cdef float current_beta1
    cdef float current_beta2
    cdef int regularization
    cdef uint64_t total_number_weights
    cdef float lambda_value
    cdef unsigned long long int t
    cdef float start_beta1
    cdef float start_beta2
    cdef int lr_decay_flag
    cdef float lr_minimum
    cdef float lr_maximum
    cdef int timestep_threshold
    cdef float decay
    def __cinit__(self, float lr = 0.1, float momentum = 0.9, int batch_size = 0, int gradient_descent_flag = NESTEROV, float current_beta1 = BETA1_ADAM, float current_beta2 = BETA2_ADAM, int regularization = NO_REGULARIZATION, uint64_t total_number_weights = 0, float lambda_value = 0, int lr_decay_flag = LR_NO_DECAY, int timestep_threshold = 0, float lr_minimum = 0, float lr_maximum = 1, float decay = 0):
        check_float(lr)
        check_float(decay)
        check_float(momentum)
        check_float(lambda_value)
        check_float(current_beta1)
        check_float(current_beta2)
        check_float(lr_minimum)
        check_float(lr_maximum)
        check_int(batch_size)
        check_int(lr_decay_flag)
        check_int(timestep_threshold)
        check_int(gradient_descent_flag)
        check_int(regularization)
        if total_number_weights < 0:
            print("total_number weights can't be < 0")
            exit(1)
        
        self.decay = decay
        self.lr_minimum = lr_minimum
        self.lr_maximum = lr_maximum
        self.timestep_threshold = timestep_threshold
        self.lr = lr
        self.lr_decay_flag = lr_decay_flag
        self.start_lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.gradient_descent_flag = gradient_descent_flag
        self.start_beta1 = current_beta1
        self.current_beta1 = current_beta1
        self.current_beta2 = current_beta2
        self.start_beta2 = current_beta2
        self.regularization = regularization
        self.total_number_weights = total_number_weights
        self.lambda_value = lambda_value
        
    def set_lr(self, float lr):
        check_float(lr)
        if lr > 1 or lr < -1:
            print("Error lr must be in [-1,1]")
            exit(1)
        self.lr = lr
    def set_momentum(self, float momentum):
        check_float(momentum)
        if momentum > 1 or momentum < 0:
            print("Error: your momentum mmust be in [0,1]")
            exit(1)
        self.momentum = momentum
    def set_batch_size(self, model m):
        self.batch_size = m.threads
    
    def set_gradient_descent(self, int flag):
        check_int(flag)
        if flag != NESTEROV and flag != ADAMOD and flag != ADAM and flag != DIFF_GRAD and flag != RADAM:
            print("Error: flag not recognized")
            exit(1)
        self.gradient_descent_flag = flag
    
    def set_current_beta1(self, float beta):
        check_float(beta)
        if beta > 1 or beta < 0:
            print("Error: beta must be in [0,1]")
            exit(1)
        self.current_beta1 = beta
    
    def set_current_beta2(self, float beta):
        check_float(beta)
        if beta > 1 or beta < 0:
            print("Error: beta must be in [0,1]")
            exit(1)
        self.current_beta2 = beta
    
    def set_regularization(self, int regularization):
        check_int(regularization)
        if regularization != NO_REGULARIZATION and regularization != L2_REGULARIZATION:
            print("Error: either no regularization or l2 regularization must be applied")
            exit(1)
    
    def set_weights_number(self, model m):
        self.total_number_weights = m.get_number_of_weights()
        
    def set_lambda(self, float lambda_value):
        check_float(lambda_value)
        if lambda_value < 0 or lambda_value > 1:
            print("Error: lambda must be in [0,1]")
            exit(1)
        self.lambda_value = lambda_value
    
    def set_lr_decay(self,int lr_decay_flag, float lr_minimum, float lr_maximum, int timestep_threshold, float decay):
        check_int(lr_decay_flag) 
        check_int(timestep_threshold) 
        check_float(lr_minimum) 
        check_float(decay) 
        check_float(lr_maximum)
        if lr_decay_flag != LR_ANNEALING_DECAY and lr_decay_flag != LR_NO_DECAY and lr_decay_flag != LR_CONSTANT_DECAY and lr_decay_flag != LR_STEP_DECAY and lr_decay_flag != LR_TIME_BASED_DECAY:
            print("Error: no decay flag recognized")
            exit(1)
        if lr_minimum > 1 or lr_minimum < -1 or lr_maximum > 1 or lr_maximum < -1:
            print("Error: lr_minimum or lr_maximum not correct")
            exit(1)
        if decay < 0 or decay > 1:
            print("Error: decay must be set in [0,1]")
            exit(1)
        self.lr_decay_flag = lr_decay_flag
        self.lr_minimum = lr_minimum
        self.lr_maximum = lr_maximum
        self.timestep_threshold = timestep_threshold
        self.decay = decay
    def update_model(self, model m):
        Pyllab.update_model(m._model, self.lr, self.momentum,self.batch_size,self.gradient_descent_flag,&self.current_beta1,&self.current_beta2,self.regularization, self.total_number_weights,self.lambda_value,&self.t)
    
    def update_dqn_categorical_dqn(self, duelingCategoricalDQN dqn):
        Pyllab.update_dueling_categorical_dqn(dqn._dqn, self.lr, self.momentum,self.batch_size,self.gradient_descent_flag,&self.current_beta1,&self.current_beta2,self.regularization, self.total_number_weights,self.lambda_value,&self.t)
    
    def update_parameters(self):
        Pyllab.update_training_parameters(&self.current_beta1, &self.current_beta2, &self.t, self.start_beta1, self.start_beta2)
        if self.lr_decay_flag != Pyllab.LR_NO_DECAY:
            Pyllab.update_lr(&self.lr, self.lr_minimum, self.lr_maximum,self.start_lr, self.decay, <int>self.t, self.timestep_threshold, self.lr_decay_flag)


cdef class neat:
    cdef Pyllab.neat* _neat
    cdef bint _keep_parents
    cdef int _species_threshold
    cdef int _initial_population
    cdef int _generations
    cdef float _percentage_survivors_per_specie
    cdef float _connection_mutation_rate
    cdef float _new_connection_assignment_rate
    cdef float _add_connection_big_specie_rate
    cdef float _add_connection_small_specie_rate
    cdef float _add_node_specie_rate
    cdef float _activate_connection_rate
    cdef float _remove_connection_rate
    cdef int _children
    cdef float _crossover_rate
    cdef int _saving
    cdef int _limiting_species
    cdef int _limiting_threshold
    cdef int _max_population
    cdef int _same_fitness_limit
    cdef float _age_significance
    cdef int _inputs
    cdef int _outputs
    
    def __cinit__(self,int inputs, int outputs, bytes neat_as_byte = None, bint keep_parents = True, int species_threshold = Pyllab.SPECIES_THERESHOLD, int initial_population = Pyllab.INITIAL_POPULATION,
                  int generations = Pyllab.GENERATIONS, float percentage_survivors_per_specie = Pyllab.PERCENTAGE_SURVIVORS_PER_SPECIE, float connection_mutation_rate = Pyllab.CONNECTION_MUTATION_RATE,
                  float new_connection_assignment_rate = Pyllab.NEW_CONNECTION_ASSIGNMENT_RATE, float add_connection_big_specie_rate = Pyllab.ADD_CONNECTION_BIG_SPECIE_RATE,
                  float add_connection_small_specie_rate = Pyllab.ADD_CONNECTION_SMALL_SPECIE_RATE, float add_node_specie_rate = Pyllab.ADD_NODE_SPECIE_RATE, float activate_connection_rate = Pyllab.ACTIVATE_CONNECTION_RATE,
                  float remove_connection_rate = Pyllab.REMOVE_CONNECTION_RATE, int children = Pyllab.CHILDREN, float crossover_rate = Pyllab.CROSSOVER_RATE, int saving = Pyllab.SAVING,
                  int limiting_species = Pyllab.LIMITING_SPECIES, int limiting_threshold = Pyllab.LIMITING_THRESHOLD, int max_population = Pyllab.MAX_POPULATION, int same_fitness_limit = Pyllab.SAME_FITNESS_LIMIT,
                  float age_significance = Pyllab.AGE_SIGNIFICANCE):
        check_int(initial_population)
        check_int(generations)
        check_int(children)
        check_int(inputs)
        check_int(outputs)
        check_int(saving)
        check_int(limiting_species)
        check_int(limiting_threshold)
        check_int(max_population)
        check_int(same_fitness_limit)
        check_int(species_threshold)
        check_float(percentage_survivors_per_specie)
        check_float(connection_mutation_rate)
        check_float(new_connection_assignment_rate)
        check_float(add_connection_big_specie_rate)
        check_float(add_connection_small_specie_rate)
        check_float(add_node_specie_rate)
        check_float(activate_connection_rate)
        check_float(remove_connection_rate)
        check_float(crossover_rate)
        check_float(age_significance)
        
        self._keep_parents = keep_parents
        self._species_threshold = species_threshold
        self._initial_population = initial_population
        self._generations = generations
        self._percentage_survivors_per_specie = percentage_survivors_per_specie
        self._connection_mutation_rate = connection_mutation_rate
        self._new_connection_assignment_rate = new_connection_assignment_rate
        self._add_connection_big_specie_rate = add_connection_big_specie_rate
        self._add_connection_small_specie_rate = add_connection_small_specie_rate
        self._add_node_specie_rate = add_node_specie_rate
        self._activate_connection_rate = activate_connection_rate
        self._remove_connection_rate = remove_connection_rate
        self._children = children
        self._crossover_rate = crossover_rate
        self._saving = saving
        self._limiting_species = limiting_species
        self._limiting_threshold = limiting_threshold
        self._max_population = max_population
        self._same_fitness_limit = same_fitness_limit
        self._age_significance = age_significance
        self._inputs = inputs
        self._outputs = outputs
        cdef char* s
        
        if inputs == 0 or outputs == 0:
            print("Error: either you pass inputs and outputs > 0 or a neat as char charachters!")
            exit(1)
        
        if neat_as_byte != None:
            s = neat_as_byte
            
            self._neat = Pyllab.init_from_char(&s[0], inputs,outputs,initial_population,species_threshold,max_population,generations, 
                                               percentage_survivors_per_specie, connection_mutation_rate, new_connection_assignment_rate, add_connection_big_specie_rate, 
                                               add_connection_small_specie_rate, add_node_specie_rate, activate_connection_rate, remove_connection_rate, children, crossover_rate, 
                                               saving, limiting_species, limiting_threshold, same_fitness_limit, keep_parents, age_significance)
            
            
        else:
            self._neat = Pyllab.init(inputs,outputs,initial_population,species_threshold,max_population,generations, 
                                     percentage_survivors_per_specie, connection_mutation_rate, new_connection_assignment_rate, add_connection_big_specie_rate, 
                                     add_connection_small_specie_rate, add_node_specie_rate, activate_connection_rate, remove_connection_rate, children, crossover_rate, 
                                     saving, limiting_species, limiting_threshold, same_fitness_limit, keep_parents, age_significance)
        if self._neat is NULL:
            raise MemoryError()
            
        
    def __dealloc__(self):
        Pyllab.free_neat(self._neat)
    
    def get_best_fitness(self):
        n = Pyllab.best_fitness(self._neat)
        return n
        
    def generation_run(self):
        Pyllab.neat_generation_run(self._neat)
    
    def reset_fitnesses(self):
        Pyllab.reset_fitnesses(self._neat)
        
    
    def reset_fitness_ith_genome(self, int index):
        check_int(index)
        Pyllab.reset_fitness_ith_genome(self._neat, index)
        
    def ff_ith_genomes(self, inputs, indices, int n_genomes):
        value_i = None
        value_e = None
        check_int(n_genomes)
        v_indices = vector_is_valid_int(indices)
        
        if v_indices.ndim != 1:
            print("Error: your dimension must be 1")
            exit(1)
        check_size(v_indices, n_genomes)
        n = get_first_dimension_size(inputs)
        if n < n_genomes:
            n_genomes = n
        if n_genomes > self.get_number_of_genomes():
            print("Error: there are not so many genomes")
            exit(1)
        
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] i_buff
        cdef npc.ndarray[npc.npy_int, ndim=1, mode = 'c'] e_buff
        cdef float** ret = NULL
        for j in range(n_genomes):
            check_size(inputs[j], self._inputs)
            if j == 0:
                value_i = np.array([vector_is_valid(inputs[j])])
            else:
                value_i = np.append(value_i, np.array([vector_is_valid(inputs[j])]),axis=0)
        value_e = vector_is_valid_int(v_indices)
        e_buff = np.ascontiguousarray(value_e,dtype=np.intc)
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        cdef float** i = <float**>malloc(sizeof(float*)*n_genomes)
        cdef int* e = <int*>&e_buff[0]
        for j in range(n_genomes):
            i[j] = &i_buff[j, 0]
        ret = Pyllab.feed_forward_iths_genome(self._neat, &i[0], &e[0], n_genomes)
        l = []
        for j in range(n_genomes):
            l.append(from_float_to_list(ret[j], self._outputs))
            free(ret[j])
        free(ret)
        free(i)
        return np.array(l, dtype='float')
        
    def ff_ith_genome(self,inputs, int index):
        check_int(index)
        value_i = vector_is_valid(inputs)
        check_size(value_i,self._inputs)
        cdef npc.ndarray[npc.npy_float32, ndim=1, mode = 'c'] i_buff
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        cdef float* i = &i_buff[0]
        cdef float* output = Pyllab.feed_forward_ith_genome(self._neat, &i_buff[0], index)
        nd_output = from_float_to_ndarray(output,self._outputs)
        free(output)
        return nd_output
   
    def increment_fitness_of_genome_ith(self,int index, float increment):
        check_int(index)
        check_float(increment)
        Pyllab.increment_fitness_of_genome_ith(self._neat,index,increment)
    
    def get_global_innovation_number_nodes(self):
        return Pyllab.get_global_innovation_number_nodes(self._neat)
   
    def get_global_innovation_number_connections(self):
        return Pyllab.get_global_innovation_number_connections(self._neat)
   
    def get_fitness_of_ith_genome(self, int index):
        check_int(index)
        return Pyllab.get_fitness_of_ith_genome(self._neat, index)
   
    def get_lenght_of_char_neat(self):
        return Pyllab.get_lenght_of_char_neat(self._neat)
   
    def get_neat_in_char(self):
        cdef char* c = Pyllab.get_neat_in_char_vector(self._neat)
        s = c[:self.get_lenght_of_char_neat()]
        free(c)
        return s
   
    def get_number_of_genomes(self):
        return Pyllab.get_number_of_genomes(self._neat)
   
    def save_ith_genome(self, int index, int n):
        check_int(n)
        check_int(index)
        Pyllab.save_ith_genome(self._neat, index, n)

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

