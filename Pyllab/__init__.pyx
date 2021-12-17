cimport Pyllab
from libc.stdlib cimport free, malloc
from cpython cimport array
from itertools import chain
import numpy as np

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
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        if len(v) != dim:
            print("Error: the dimension of one of your arrays is not correct!")
            exit(1)
    else:
        print("Error: one of your vectors is not a list, neither a np.ndarray!")
        exit(1)

#if is not multi dimensional there is not good
def get_first_dimension_size(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        else:
            print("your array must be multi dimensional! d > 1")
            exit(1)
        return len(v)
    elif type(v) == np.ndarray:
        if v.ndim < 2:
            print("your array must be multi dimensional! d > 1")
            exit(1)
        else:
            return v.shape[0]
    else:
        print("the input is not a list neither a np.darray!")
        exit(1)
        

def get_dict_from_model_setup_file(filename):
    try:
        d = {}
        f = open(filename,"r")
        l = f.read().split('\n')
        f.close()
        if len(l)<2 or len(l)%2 == 0:
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
                            for j in d[i][k][ii]:
                                if type(j) != int:
                                    print("Error, some values in the list of the value of a fully-connected is not an int")
                                    return False
                                    
                                if j < 0 or j >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False 
                            if d[i][k][ii][2] in layers_list:
                                print("layer already exists!")
                                return False
                            layers_list.append(d[i][k][ii][2])
                        elif ii == 'convolutional':
                            if type(d[i][k][ii]) != list or len(d[i][k][ii]) != 24:
                                print("Error: either you value for a convolutional is not a list, or its length is not 24!")
                                return False
                            for j in d[i][k][ii]:
                                if type(j) != int:
                                    print("Error, some values in the list of the value of a fully-connected is not an int")
                                    return False
                                    
                                if j < 0 or j >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False
                            if d[i][k][ii][23] in layers_list:
                                print("layer already exists!")
                                return False
                            layers_list.append(d[i][k][ii][23])
                        elif ii == 'rconvolutional':
                            if type(d[i][k][ii]) != list or len(d[i][k][ii]) != 26:
                                print("Error: either you value for a rconvolutional is not a list, or its length is not 26!")
                                return False
                            for j in d[i][k][ii]:
                                if type(j) != int:
                                    print("Error, some values in the list of the value of a fully-connected is not an int")
                                    return False
                                    
                                if j < 0 or j >= 2**31-1:
                                    print("Error: no negative values, neither overflow are permitted!")
                                    return False
                            if d[i][k][ii][23] in layers_list:
                                print("layer already exists!")
                                return False
                            layers_list.append(d[i][k][ii][23])
                            if d[i][k][ii][25] in residuals_dict.keys():
                                if kk-1 not in residuals_dict[d[i][k][ii][25]]:
                                    print(kk-1)
                                    print(residuals_dict[d[i][k][ii][25]])
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
        return True
    except:
        return False


cdef from_float_to_ndarray(float* ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return np.ndarray(lst)
     
cdef from_float_to_array(float *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return array.array('f',lst)
    
cdef from_float_to_list(float *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef assign_float_pointer_from_array(float** handler, float[:] a, int index):
    handler[index] = <float*>(&a[0])

def vector_is_valid(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return array.array('f',v)
    elif type(v) == np.ndarray:
        v = v.flatten()
        return array.array('f',v)
    elif type(v) != array.array:
        print("Error: you vector is not a list neither an array.array type")
        exit(1)
    return v

def dot1D(input1, input2, output, int size):
    cdef array.array i1 = vector_is_valid(input1)
    cdef array.array i2 = vector_is_valid(input2)
    cdef array.array o = vector_is_valid(output)
    Pyllab.dot1D(<float*>i1.data.as_floats,<float*>i2.data.as_floats,<float*>o.data.as_floats,size)

def sum1D(input1, input2, output, int size):
    cdef array.array i1 = vector_is_valid(input1)
    cdef array.array i2 = vector_is_valid(input2)
    cdef array.array o = vector_is_valid(output)
    Pyllab.sum1D(<float*>i1.data.as_floats,<float*>i2.data.as_floats,<float*>o.data.as_floats,size)


def mul_value(input1,float value, output, int size):
    cdef array.array i1 = vector_is_valid(input1)
    cdef array.array o = vector_is_valid(output)
    Pyllab.mul_value(<float*>i1.data.as_floats,value,<float*>o.data.as_floats,size)

def clip_vector(vector, float minimum, float maximum, int dimension):
    cdef array.array v = vector_is_valid(vector)
    Pyllab.clip_vector(<float*>v.data.as_floats,minimum,maximum,dimension)
    
    




