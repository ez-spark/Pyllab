from libc.stdio cimport FILE
from libc cimport stdint
ctypedef stdint.uint64_t uint64_t



cdef extern from "../src/llab.h":
    ctypedef struct bn:
        pass
    ctypedef struct fcl:
        pass
    ctypedef struct cl:
        pass
    ctypedef struct rl:
        pass
    ctypedef struct lstm:
        pass
    ctypedef struct model:
        pass
    ctypedef struct rmodel:
        pass
    ctypedef struct recurrent_enc_dec:
        pass
    ctypedef struct vaemodel:
        pass
    ctypedef struct scaled_l2_norm:
        pass
    ctypedef struct transformer_encoder:
        pass
    ctypedef struct transformer_decoder:
        pass
    ctypedef struct transformer:
        pass
    ctypedef struct thread_args_model:
        pass
    ctypedef struct thread_args_rmodel:
        pass
    ctypedef struct thread_args_enc_dec_model:
        pass
    ctypedef struct thread_args_vae_model:
        pass
    ctypedef struct thread_args_transformer_encoder:
        pass
    ctypedef struct thread_args_transformer_decoder:
        pass
    ctypedef struct thread_args_transformer:
        pass
    ctypedef struct server:
        pass
    ctypedef struct ddpg:
        pass
    ctypedef struct oustrategy:
        pass
    ctypedef struct mystruct:
        pass
    ctypedef struct training:
        pass
    ctypedef struct vector_struct:
        pass
    ctypedef struct struct_conn:
        pass
    ctypedef struct struct_conn_handler:
        pass
    ctypedef struct error_handler:
        pass
    ctypedef struct error_super_struct:
        pass
    cdef enum:
        N_NORMALIZATION
    cdef enum:
        BETA_NORMALIZATION
    cdef enum:
        ALPHA_NORMALIZATION
    cdef enum:
        K_NORMALIZATION
    cdef enum:
        NESTEROV
    cdef enum:
        ADAM
    cdef enum:
        RADAM
    cdef enum:
        DIFF_GRAD
    cdef enum:
        ADAMOD
    cdef enum:
        FCLS
    cdef enum:
        CLS
    cdef enum:
        RLS
    cdef enum:
        BNS
    cdef enum:
        LSTMS
    cdef enum:
        TRANSFORMER_ENCODER
    cdef enum:
        TRANSFORMER_DECODER
    cdef enum:
        TRANSFORMER
    cdef enum:
        MODEL
    cdef enum:
        RMODEL
    cdef enum:
        ATTENTION
    cdef enum:
        MULTI_HEAD_ATTENTION
    cdef enum:
        L2_NORM_CONN
    cdef enum:
        VECTOR
    cdef enum:
        NOTHING
    cdef enum:
        TEMPORAL_ENCODING_MODEL
    cdef enum:
        NO_ACTIVATION
    cdef enum:
        SIGMOID
    cdef enum:
        RELU
    cdef enum:
        SOFTMAX
    cdef enum:
        TANH
    cdef enum:
        LEAKY_RELU
    cdef enum:
        ELU
    cdef enum:
        NO_POOLING
    cdef enum:
        MAX_POOLING
    cdef enum:
        AVARAGE_POOLING
    cdef enum:
        NO_DROPOUT
    cdef enum:
        DROPOUT
    cdef enum:
        DROPOUT_TEST
    cdef enum:
        NO_NORMALIZATION
    cdef enum:
        LOCAL_RESPONSE_NORMALIZATION
    cdef enum:
        BATCH_NORMALIZATION
    cdef enum:
        GROUP_NORMALIZATION
    cdef enum:
        LAYER_NORMALIZATION
    cdef enum:
        SCALED_L2_NORMALIZATION
    cdef enum:
        COSINE_NORMALIZATION
    cdef enum:
        BETA1_ADAM
    cdef enum:
        BETA2_ADAM
    cdef enum:
        BETA3_ADAMOD
    cdef enum:
        EPSILON_ADAM
    cdef enum:
        EPSILON
    cdef enum:
        RADAM_THRESHOLD
    cdef enum:
        NO_REGULARIZATION
    cdef enum:
        L2_REGULARIZATION
    cdef enum:
        NO_CONVOLUTION
    cdef enum:
        CONVOLUTION 
    cdef enum:
        TRANSPOSED_CONVOLUTION 
    cdef enum:
        BATCH_NORMALIZATION_TRAINING_MODE 
    cdef enum:
        BATCH_NORMALIZATION_FINAL_MODE 
    cdef enum:
        STATEFUL
    cdef enum:
        STATELESS
    cdef enum:
        LEAKY_RELU_THRESHOLD 
    cdef enum:
        ELU_THRESHOLD
    cdef enum:
        LSTM_RESIDUAL 
    cdef enum:
        LSTM_NO_RESIDUAL
    cdef enum:
        TRANSFORMER_RESIDUAL
    cdef enum:
        TRANSFORMER_NO_RESIDUAL 
    cdef enum:
        NO_SET 
    cdef enum:    
        NO_LOSS
    cdef enum:
        CROSS_ENTROPY_LOSS
    cdef enum:
        FOCAL_LOSS
    cdef enum:
        HUBER1_LOSS
    cdef enum:
        HUBER2_LOSS
    cdef enum:
        MSE_LOSS
    cdef enum:
        KL_DIVERGENCE_LOSS
    cdef enum:
        ENTROPY_LOSS
    cdef enum:
        TOTAL_VARIATION_LOSS_2D
    cdef enum:
        CONTRASTIVE_2D_LOSS
    cdef enum:
        LOOK_AHEAD_ALPHA
    cdef enum:
        LOOK_AHEAD_K
    cdef enum:
        GRADIENT_DESCENT 
    cdef enum:
        EDGE_POPUP
    cdef enum:
        FULLY_FEED_FORWARD
    cdef enum:
        FREEZE_TRAINING
    cdef enum:
        FREEZE_BIASES
    cdef enum:
        ONLY_FF
    cdef enum:
        ONLY_DROPOUT
    cdef enum:
        STANDARD_ATTENTION
    cdef enum:
        MASKED_ATTENTION
    cdef enum:
        RUN_ONLY_DECODER
    cdef enum:
        RUN_ONLY_ENCODER
    cdef enum:
        RUN_ALL_TRANSF
    cdef enum:
        SORT_SWITCH_THRESHOLD
    cdef enum:
        NO_ACTION 
    cdef enum:
        ADDITION
    cdef enum:
        SUBTRACTION 
    cdef enum:
        MULTIPLICATION
    cdef enum:
        RESIZE
    cdef enum:
        CONCATENATE
    cdef enum:
        DIVISION
    cdef enum:
        INVERSE
    cdef enum:
        CHANGE_SIGN
    cdef enum:
        GET_MAX
    cdef enum:
        NO_CONCATENATE
    cdef enum:
        POSITIONAL_ENCODING
    cdef enum:
        SPECIES_THERESHOLD
    cdef enum:
        INITIAL_POPULATION
    cdef enum:
        GENERATIONS
    cdef enum:
        PERCENTAGE_SURVIVORS_PER_SPECIE
    cdef enum:
        CONNECTION_MUTATION_RATE
    cdef enum:
        NEW_CONNECTION_ASSIGNMENT_RATE
    cdef enum:
        ADD_CONNECTION_BIG_SPECIE_RATE
    cdef enum:
        ADD_CONNECTION_SMALL_SPECIE_RATE
    cdef enum:
        ADD_NODE_SPECIE_RATE
    cdef enum:
        ACTIVATE_CONNECTION_RATE
    cdef enum:
        REMOVE_CONNECTION_RATE
    cdef enum:
        CHILDREN
    cdef enum:
        CROSSOVER_RATE
    cdef enum:
        SAVING
    cdef enum:
        LIMITING_SPECIES
    cdef enum:
        LIMITING_THRESHOLD
    cdef enum:
        MAX_POPULATION
    cdef enum:
        SAME_FITNESS_LIMIT
    cdef enum:
        AGE_SIGNIFICANCE
        
        
cdef extern from "../src/attention.h":
    void self_attention_ff(float* query, float* key, float* value, float* score_matrix,float* score_matrix_softmax,float* output, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension)
    void self_attention_bp(float* query, float* key, float* value, float* query_error, float* key_error, float* value_error, float* score_matrix,float* score_matrix_softmax,float* score_matrix_error,float* score_matrix_softmax_error,float* output_error, int dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension)
    void multi_head_attention_ff(model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output, int dimension, int n_heads, int output_dimension, int attention_flag, int k_embedding_dimension, int v_embedding_dimension)
    void multi_head_attention_bp(float* queries_error, float* keys_error, float* values_error, float* score_matrices_error, float* score_matrices_softmax_error, model** queries, model** keys, model** values,float* score_matrices, float* score_matrices_softmax, float* output_error, int dimension, int n_heads, int output_dimension, int attention_flag,int k_embedding_dimension,int v_embedding_dimension)

cdef extern from "../src/batch_norm_layers.h":
    bn* batch_normalization(int batch_size, int vector_input_dimension)
    void free_batch_normalization(bn* b)
    void save_bn(bn* b, int n)
    bn* load_bn(FILE* fr)
    bn* copy_bn(bn* b)
    bn* reset_bn(bn* b)
    uint64_t size_of_bn(bn* b)
    void paste_bn(bn* b1, bn* b2)
    void slow_paste_bn(bn* f, bn* copy,float tau)
    bn* batch_normalization_without_learning_parameters(int batch_size, int vector_input_dimension)
    bn* copy_bn_without_learning_parameters(bn* b)
    uint64_t size_of_bn_without_learning_parameters(bn* b)
    void paste_bn_without_learning_parameters(bn* b1, bn* b2)
    bn* batch_normalization_without_arrays(int batch_size, int vector_input_dimension)
    void make_the_bn_only_for_ff(bn* b)
    void free_the_bn_only_for_ff(bn* b)
    void paste_w_bn(bn* b1, bn* b2)
    bn* reset_bn_except_partial_derivatives(bn* b)
    
cdef extern from "../src/client.h":
    bint run_client(int port, char* server_address, int buffer_size, int reading_pipe, int writing_pipe)
    void contact_server(int sockfd, int buffer_size, int reading_pipe, int writing_pipe)

cdef extern from "../src/clipping_gradient.h":
    void clipping_gradient(model* m, float threshold)
    void clip_rls(rl** rls, int n, float threshold,float norm)
    void clip_cls(cl** cls, int n, float threshold, float norm)
    void clip_fcls(fcl** fcls, int n, float threshold, float norm)
    float sum_all_quadratic_derivative_weights_rls(rl** rls, int n)
    float sum_all_quadratic_derivative_weights_cls(cl** cls, int n)
    float sum_all_quadratic_derivative_weights_fcls(fcl** fcls, int n)
    void clip_lstms(lstm** lstms, int n, float threshold, float norm)
    float sum_all_quadratic_derivative_weights_lstms(lstm** lstms, int n)
    void clipping_gradient_rmodel(rmodel* m, float threshold)
    float sum_all_quadratic_derivative_weights_bns(bn** bns, int n)
    void clip_bns(bn** bns, int n, float threshold, float norm)
    void clipping_gradient_vae_model(vaemodel* m, float threshold)
    void general_clipping_gradient(model** m, rmodel** r,transformer** t, transformer_encoder** e, transformer_decoder** d, int n_m, int n_r, int n_t,int n_e, int n_d, float threshold)
    float sum_all_quadratic_derivative_weights_scaled_l2_norm(scaled_l2_norm** l, int n)
    void clip_scaled_l2(scaled_l2_norm** l, int n, float threshold, float norm)
    float sum_all_quadratic_derivative_weights_m(model* m)
    void clipping_gradient_transf_encoder(transformer_encoder* t, float threshold)
    void clipping_gradient_transf_decoder(transformer_decoder* t, float threshold)
    void clipping_gradient_transf(transformer* t, float threshold)
    void adaptive_gradient_clipping_lstm(lstm* f ,float threshold, float epsilon)
    void adaptive_gradient_clipping_fcl(fcl* f ,float threshold, float epsilon)
    void adaptive_gradient_clipping_cl(cl* f ,float threshold, float epsilon)
    void adaptive_gradient_clipping_rl(rl* f ,float threshold, float epsilon)
    void adaptive_gradient_clipping_model(model* m, float threshold, float epsilon)
    void adaptive_gradient_clipping_rmodel(rmodel* r, float threshold, float epsilon)
    void adaptive_gradient_clipping_encoder_transformer(transformer_encoder* e, float threshold, float epsilon)
    void adaptive_gradient_clipping_decoder_transformer(transformer_decoder* t, float threshold, float epsilon)
    void adaptive_gradient_clipping_transformer(transformer* t, float threshold, float epsilon)

cdef extern from "../src/convolutional.h":
    void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1, int stride2, int padding)
    void convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding)
    void max_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding)
    void max_pooling_back_prop(float* input, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding, float* input_error)
    void avarage_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding)
    void avarage_pooling_back_prop(float* input_error, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding)
    void convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n)
    void convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error)
    void convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n)
    void convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error)
    void transposed_convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1, int stride2, int padding)
    void transposed_convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding)
    void transposed_convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n)
    void transposed_convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error)
    void transposed_convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output_error, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error)
    void transposed_convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n)

cdef extern from "../src/convolutional_layers.h":
    cl* convolutional(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer)
    bint exists_d_kernels_cl(cl* c)
    bint exists_d_biases_cl(cl* c)
    bint exists_kernels_cl(cl* c)
    bint exists_biases_cl(cl* c)
    bint exists_pre_activation_cl(cl* c)
    bint exists_post_activation_cl(cl* c)
    bint exists_normalization_cl(cl* c)
    bint exists_pooling(cl* c)
    bint exists_edge_popup_stuff_cl(cl * c)
    bint exists_edge_popup_stuff_with_only_training_mode_cl(cl * c)
    bint exists_bp_handler_arrays(cl* c)
    void free_convolutional(cl* c)
    void save_cl(cl* f, int n)
    void copy_cl_params(cl* f, float** kernels, float* biases)
    cl* load_cl(FILE* fr)
    cl* copy_cl(cl* f)
    cl* reset_cl(cl* f)
    cl* reset_cl_except_partial_derivatives(cl* f)
    cl* reset_cl_without_dwdb(cl* f)
    cl* reset_cl_for_edge_popup(cl* f)
    uint64_t size_of_cls(cl* f)
    void paste_cl(cl* f, cl* copy)
    void paste_w_cl(cl* f, cl* copy)
    void slow_paste_cl(cl* f, cl* copy,float tau)
    uint64_t get_array_size_params_cl(cl* f)
    uint64_t get_array_size_weights_cl(cl* f)
    uint64_t get_array_size_scores_cl(cl* f)
    void memcopy_vector_to_params_cl(cl* f, float* vector)
    void memcopy_params_to_vector_cl(cl* f, float* vector)
    void memcopy_vector_to_weights_cl(cl* f, float* vector)
    void memcopy_weights_to_vector_cl(cl* f, float* vector)
    void memcopy_scores_to_vector_cl(cl* f, float* vector)
    void memcopy_vector_to_scores_cl(cl* f, float* vector)
    void memcopy_vector_to_derivative_params_cl(cl* f, float* vector)
    void memcopy_derivative_params_to_vector_cl(cl* f, float* vector)
    void set_convolutional_biases_to_zero(cl* c)
    void set_convolutional_unused_weights_to_zero(cl* c)
    void sum_score_cl(cl* input1, cl* input2, cl* output)
    void compare_score_cl(cl* input1, cl* input2, cl* output)
    void compare_score_cl_with_vector(cl* input1, float* input2, cl* output)
    void dividing_score_cl(cl* c,float value)
    void reset_score_cl(cl* f)
    void reinitialize_weights_according_to_scores_cl(cl* f, float percentage, float goodness)
    void reinitialize_w_cl(cl* f)
    cl* reset_edge_popup_d_cl(cl* f)
    void set_low_score_cl(cl* f)
    cl* convolutional_without_learning_parameters(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer)
    void free_convolutional_without_learning_parameters(cl* c)
    cl* copy_cl_without_learning_parameters(cl* f)
    cl* reset_cl_without_learning_parameters(cl* f)
    cl* reset_cl_without_dwdb_without_learning_parameters(cl* f)
    uint64_t size_of_cls_without_learning_parameters(cl* f)
    void paste_cl_without_learning_parameters(cl* f, cl* copy)
    uint64_t count_weights_cl(cl* c)
    void make_the_cl_only_for_ff(cl* c)
    void set_feed_forward_flag(cl* c, int feed_forward_flag)
    cl* convolutional_without_arrays(int channels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int n_kernels, int stride1_rows, int stride1_cols, int padding1_rows, int padding1_cols, int stride2_rows, int stride2_cols, int padding2_rows, int padding2_cols, int pooling_rows, int pooling_cols, int normalization_flag, int activation_flag, int pooling_flag, int group_norm_channels, int convolutional_flag,int training_mode, int feed_forward_flag, int layer)
    void free_convolutional_without_arrays(cl* c)

cdef extern from "../src/dictionary.h":
    bint check_int_array(int* array, mystruct** ms, int size, int index)
    void free_my_struct(mystruct** ms)


cdef extern from "../src/fully_connected.h":
    void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size)
    void fully_connected_back_prop(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size, int training_flag)
    void fully_connected_back_prop_edge_popup(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n)
    void fully_connected_feed_forward_edge_popup(float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, int last_n)
    void fully_connected_back_prop_edge_popup_ff_gd_bp(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n)
    void paste_w_fcl(fcl* f,fcl* copy)
    
cdef extern from "../src/fully_connected_layers.h":
    fcl* fully_connected(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag)
    bint exists_params_fcl(fcl* f)
    bint exists_d_params_fcl(fcl* f)
    bint exists_dropout_stuff_fcl(fcl* f)
    bint exists_edge_popup_stuff_fcl(fcl* f)
    bint exists_activation_fcl(fcl* f)
    bint exists_normalization_fcl(fcl* f)
    void free_fully_connected(fcl* f)
    void free_fully_connected_for_edge_popup(fcl* f)
    void free_fully_connected_complementary_edge_popup(fcl* f)
    void save_fcl(fcl* f, int n)
    void copy_fcl_params(fcl* f, float* weights, float* biases)
    fcl* load_fcl(FILE* fr)
    fcl* copy_fcl(fcl* f)
    fcl* copy_light_fcl(fcl* f)
    fcl* reset_fcl(fcl* f)
    fcl* reset_fcl_except_partial_derivatives(fcl* f)
    fcl* reset_fcl_without_dwdb(fcl* f)
    fcl* reset_fcl_for_edge_popup(fcl* f)
    uint64_t size_of_fcls(fcl* f)
    void paste_fcl(fcl* f,fcl* copy)
    void paste_w_fcl(fcl* f,fcl* copy)
    void slow_paste_fcl(fcl* f,fcl* copy, float tau)
    uint64_t get_array_size_params(fcl* f)
    uint64_t get_array_size_scores_fcl(fcl* f)
    uint64_t get_array_size_weights(fcl* f)
    void memcopy_vector_to_params(fcl* f, float* vector)
    void memcopy_vector_to_scores(fcl* f, float* vector)
    void memcopy_params_to_vector(fcl* f, float* vector)
    void memcopy_weights_to_vector(fcl* f, float* vector)
    void memcopy_vector_to_weights(fcl* f, float* vector)
    void memcopy_scores_to_vector(fcl* f, float* vector)
    void memcopy_vector_to_derivative_params(fcl* f, float* vector)
    void memcopy_derivative_params_to_vector(fcl* f, float* vector)
    void set_fully_connected_biases_to_zero(fcl* f)
    void set_fully_connected_unused_weights_to_zero(fcl* f)
    void sum_score_fcl(fcl* input1, fcl* input2, fcl* output)
    void compare_score_fcl(fcl* input1, fcl* input2, fcl* output)
    void compare_score_fcl_with_vector(fcl* input1, float* input2, fcl* output)
    void dividing_score_fcl(fcl* f, float value)
    void set_fcl_only_dropout(fcl* f)
    void reset_score_fcl(fcl* f)
    void reinitialize_weights_according_to_scores_fcl(fcl* f, float percentage, float goodness)
    void reinitialize_w_fcl(fcl* f)
    fcl* reset_edge_popup_d_fcl(fcl* f)
    void set_low_score_fcl(fcl* f)
    int* get_used_outputs(fcl* f, int* used_output, int flag, int output_size)
    fcl* copy_fcl_without_learning_parameters(fcl* f)
    fcl* fully_connected_without_learning_parameters(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag)
    fcl* reset_fcl_without_learning_parameters(fcl* f)
    uint64_t size_of_fcls_without_learning_parameters(fcl* f)
    void paste_fcl_without_learning_parameters(fcl* f,fcl* copy)
    fcl* reset_fcl_without_dwdb_without_learning_parameters(fcl* f)
    uint64_t count_weights_fcl(fcl* f)
    void make_the_fcl_only_for_ff(fcl* f)
    fcl* fully_connected_without_arrays(int input, int output, int layer, int dropout_flag, int activation_flag, float dropout_threshold, int n_groups, int normalization_flag, int training_mode, int feed_forward_flag)
    void free_fully_connected_without_arrays(fcl* f)

cdef extern from "../src/gd.h":
    void nesterov_momentum(float* p, float lr, float m, int mini_batch_size, float dp, float* delta)
    void adam_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size)
    void radam_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size, unsigned long long int t)
    void adam_diff_grad_algorithm(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size, float* ex_d)
    void adamod(float* p,float* delta1, float* delta2, float dp, float lr, float b1, float b2, float bb1, float bb2, float epsilon, int mini_batch_size, float b3, float* delta3)

cdef extern from "../src/initialization.h":
    float r2()
    float generate_from_random_distribution(float lo, float hi)
    float drand ()
    float random_normal ()
    float random_general_gaussian(float mean, float std)
    float random_general_gaussian_xavier_init(float n)
    float random_general_gaussian_xavier_init2(float n1,float n2)
    float random_general_gaussian_kaiming_init(float n)
    float signed_r2(float n)
    float signed_kaiming_constant(float n)

cdef extern from "../src/math_functions.h":
    void softmax(float* input, float* output, int size)
    void derivative_softmax_array(int* input, float* output,float* softmax_arr,float* error, int size)
    float sigmoid(float x)
    void sigmoid_array(float* input, float* output, int size)
    float derivative_sigmoid(float x)
    void derivative_sigmoid_array(float* input, float* output, int size)
    float relu(float x)
    void relu_array(float* input, float* output, int size)
    float derivative_relu(float x)
    void derivative_relu_array(float* input, float* output, int size)
    float leaky_relu(float x)
    void leaky_relu_array(float* input, float* output, int size)
    float derivative_leaky_relu(float x)
    void derivative_leaky_relu_array(float* input, float* output, int size)
    float tanhh(float x)
    void tanhh_array(float* input, float* output, int size)
    float derivative_tanhh(float x)
    void derivative_tanhh_array(float* input, float* output, int size)
    float mse(float y_hat, float y)
    float derivative_mse(float y_hat, float y)
    float cross_entropy(float y_hat, float y)
    float derivative_cross_entropy(float y_hat, float y)
    float cross_entropy_reduced_form(float y_hat, float y)
    float derivative_cross_entropy_reduced_form_with_softmax(float y_hat, float y)
    void derivative_cross_entropy_reduced_form_with_softmax_array(float* y_hat, float* y,float* output, int size)
    float huber_loss(float y_hat, float y, float threshold)
    float derivative_huber_loss(float y_hat, float y, float threshold)
    void derivative_huber_loss_array(float* y_hat, float* y,float* output, float threshold, int size)
    float modified_huber_loss(float y_hat, float y, float threshold1, float threshold2)
    float derivative_modified_huber_loss(float y_hat, float y, float threshold1, float threshold2)
    void derivative_modified_huber_loss_array(float* y_hat, float* y, float threshold1, float* output, float threshold2, int size)
    float focal_loss(float y_hat, float y, float gamma)
    void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size)
    float derivative_focal_loss(float y_hat, float y, float gamma)
    void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size)
    void mse_array(float* y_hat, float* y, float* output, int size)
    void derivative_mse_array(float* y_hat, float* y, float* output, int size)
    void cross_entropy_array(float* y_hat, float* y, float* output, int size)
    void derivative_cross_entropy_array(float* y_hat, float* y, float* output, int size)
    void kl_divergence(float* input1, float* input2, float* output, int size)
    void derivative_kl_divergence(float* y_hat, float* y, float* output, int size)
    float entropy(float y_hat)
    void entropy_array(float* y_hat, float* output, int size)
    float derivative_entropy(float y_hat)
    void derivative_entropy_array(float* y_hat, float* output, int size)
    float abs_sigmoid(float x)
    void abs_sigmoid_array(float* input, float* output, int size)
    void softmax_array_not_complete(float* input, float* output,int* mask, int size)
    float elu(float z, float a)
    void elu_array(float* input, float* output, int size, float a)
    float derivative_elu(float z, float a)
    void derivative_elu_array(float* input, float* output, int size, float a)
    void derivative_softmax(float* output,float* softmax_arr,float* error, int size)
    void dot1D(float* input1, float* input2, float* output, int size)
    void sum1D(float* input1, float* input2, float* output, int size)
    void mul_value(float* input, float value, float* output, int dimension)
    void sum_residual_layers_partial_derivatives(model* m, model* m2, model* m3)
    void sum_convolutional_layers_partial_derivatives(model* m, model* m2, model* m3)
    void sum_fully_connected_layers_partial_derivatives(model* m, model* m2, model* m3)
    void sum_lstm_layers_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3)
    float float_abs(float a)
    void float_abs_array(float* a, int n)
    float* get_float_abs_array(float* a, int n)
    void dot_float_input(float* input1, int* input2, float* output, int size)
    void sum_model_partial_derivatives(model* m, model* m2, model* m3)
    void sum_models_partial_derivatives(model* sum_m, model** models, int n_models)
    void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3)
    void sum_rmodels_partial_derivatives(rmodel* m, rmodel** m2, int n_models)
    void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3)
    int min(int x, int y)
    int max(int x, int y)
    double sum_over_input(float* inputs, int dimension)
    float derivative_sigmoid_given_the_sigmoid(float x)
    void derivative_sigmoid_array_given_the_sigmoid(float* input, float* output, int size)
    float total_variation_loss_2d(float* y, int rows, int cols)
    void derivative_total_variation_loss_2d(float* y, float* output, int rows, int cols)
    void div1D(float* input1, float* input2, float* output, int size)
    void sub1D(float* input1, float* input2, float* output, int size)
    void inverse(float* input, float* output, int size)
    float min_float(float x, float y)
    float max_float(float x, float y)
    float constrantive_loss(float y_hat, float y, float margin)
    float derivative_constrantive_loss(float y_hat, float y, float margin)
    void constrantive_loss_array(float* y_hat, float* y,float* output, float margin, int size)
    void derivative_constrantive_loss_array(float* y_hat, float* y,float* output, float margin, int size)
    float dotProduct1D(float* input1, float* input2, int size)
    void additional_mul_value(float* input, float value, float* output, int dimension)
    void copy_clipped_vector(float* vector, float* output, float maximum, float minimum, int dimension)
    void clip_vector(float* vector, float minimum, float maximum, int dimension)

cdef extern from "../src/model.h":
    model* network(int layers, int n_rl, int n_cl, int n_fcl, rl** rls, cl** cls, fcl** fcls)
    void free_model(model* m)
    model* copy_model(model* m)
    void paste_model(model* m, model* copy)
    void paste_w_model(model* m, model* copy)
    void slow_paste_model(model* m, model* copy, float tau)
    model* reset_model(model* m)
    model* reset_model_except_partial_derivatives(model* m)
    model* reset_model_without_dwdb(model* m)
    model* reset_model_for_edge_popup(model* m)
    uint64_t size_of_model(model* m)
    void save_model(model* m, int n)
    void save_model_given_directory(model* m, int n, char* directory)
    model* load_model(char* file)
    model* load_model_with_file_already_opened(FILE* fr)
    void ff_fcl_fcl(fcl* f1, fcl* f2)
    void ff_fcl_cl(fcl* f1, cl* f2)
    void ff_cl_fcl(cl* f1, fcl* f2)
    void ff_cl_cl(cl* f1, cl* f2)
    float* bp_fcl_fcl(fcl* f1, fcl* f2, float* error)
    float* bp_fcl_cl(fcl* f1, cl* f2, float* error)
    float* bp_cl_cl(cl* f1, cl* f2, float* error)
    float* bp_cl_fcl(cl* f1, fcl* f2, float* error)
    void model_tensor_input_ff(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input)
    float* model_tensor_input_bp(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension)
    uint64_t count_weights(model* m)
    uint64_t get_array_size_params_model(model* f)
    uint64_t get_array_size_weights_model(model* f)
    uint64_t get_array_size_scores_model(model* f)
    void memcopy_vector_to_params_model(model* f, float* vector)
    void memcopy_vector_to_weights_model(model* f, float* vector)
    void memcopy_vector_to_scores_model(model* f, float* vector)
    void memcopy_params_to_vector_model(model* f, float* vector)
    void memcopy_weights_to_vector_model(model* f, float* vector)
    void memcopy_scores_to_vector_model(model* f, float* vector)
    void memcopy_vector_to_derivative_params_model(model* f, float* vector)
    void memcopy_derivative_params_to_vector_model(model* f, float* vector)
    void set_model_error(model* m, int error_flag, float threshold1, float threshold2, float gamma, float* alpha, int output_dimension)
    void mse_model_error(model* m, float* output)
    void cross_entropy_model_error(model* m, float* output)
    void focal_model_error(model* m, float* output)
    void huber_one_model_error(model* m, float* output)
    void huber_two_model_error(model* m, float* output)
    void kl_model_error(model* m, float* output)
    void entropy_model_error(model* m, float* output)
    void compute_model_error(model* m, float* output)
    float* ff_error_bp_model_once(model* m, int tensor_depth, int tensor_i, int tensor_j, float* input, float* output)
    void set_model_biases_to_zero(model* m)
    void set_model_unused_weights_to_zero(model* m)
    void set_model_training_edge_popup(model* m, float k_percentage)
    void set_model_training_gd(model* m)
    void sum_score_model(model* input1, model* input2, model* output)
    void compare_score_model(model* input1, model* input2, model* output)
    void compare_score_model_with_vector(model* input1, float* input2, model* output)
    void dividing_score_model(model* m, float value)
    void avaraging_score_model(model* avarage, model** m, int n_model)
    void reset_score_model(model* f)
    void reinitialize_weights_according_to_scores_model(model* m, float percentage, float goodness)
    void reinitialize_w_model(model* m)
    model* reset_edge_popup_d_model(model* m)
    int check_model_last_layer(model* m)
    void set_low_score_model(model* f)
    void free_model_without_learning_parameters(model* m)
    model* copy_model_without_learning_parameters(model* m)
    void paste_model_without_learning_parameters(model* m, model* copy)
    model* reset_model_without_learning_parameters(model* m)
    model* reset_model_without_dwdb_without_learning_parameters(model* m)
    uint64_t size_of_model_without_learning_parameters(model* m)
    void ff_fcl_fcl_without_learning_parameters(fcl* f1, fcl* f2, fcl* f3)
    void ff_fcl_cl_without_learning_parameters(fcl* f1, cl* f2, cl* f3)
    void ff_cl_fcl_without_learning_parameters(cl* f1, fcl* f2, fcl* f3)
    void ff_cl_cl_without_learning_parameters(cl* f1, cl* f2, cl* f3)
    float* bp_fcl_fcl_without_learning_parameters(fcl* f1, fcl* f2, fcl* f3, float* error)
    float* bp_fcl_cl_without_learning_parameters(fcl* f1, cl* f2,cl* f3, float* error)
    float* bp_cl_cl_without_learning_parameters(cl* f1, cl* f2,cl* f3, float* error)
    float* bp_cl_fcl_without_learning_parameters(cl* f1, fcl* f2,fcl* f3, float* error)
    void model_tensor_input_ff_without_learning_parameters(model* m, model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input)
    float* model_tensor_input_bp_without_learning_parameters(model* m, model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension)
    float* ff_error_bp_model_once_opt(model* m,model* m2, int tensor_depth, int tensor_i, int tensor_j, float* input, float* output)
    void free_model_without_arrays(model* m)
    
cdef extern from "../src/multi_core_model.h":
    void* model_thread_ff(void* _args)
    void* model_thread_bp(void* _args)
    void model_tensor_input_ff_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads)
    void model_tensor_input_bp_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error)
    void ff_error_bp_model_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error)
    void* model_thread_ff_bp(void* _args)
    void* model_thread_ff_opt(void* _args)
    void model_tensor_input_ff_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads)
    void model_tensor_input_bp_multicore_opt(model** m,model*m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error)
    void* model_thread_bp_opt(void* _args)
    void ff_error_bp_model_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error)
    void* model_thread_ff_bp_opt(void* _args)

cdef extern from "../src/multi_core_rmodel.h":
    void* rmodel_thread_ff(void* _args)
    void* rmodel_thread_bp(void* _args)
    void ff_rmodel_lstm_multicore(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, int mini_batch_size, int threads)
    void bp_rmodel_lstm_multicore(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, float*** error_model, int mini_batch_size, int threads, float**** returning_error, float*** returning_input_error)
    void* rmodel_thread_ff_opt(void* _args)
    void* rmodel_thread_bp_opt(void* _args)
    void ff_rmodel_lstm_multicore_opt(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, int mini_batch_size, int threads, rmodel* m2)
    void bp_rmodel_lstm_multicore_opt(float*** hidden_states, float*** cell_states, float*** input_model, rmodel** m, float*** error_model, int mini_batch_size, int threads, float**** returning_error, float*** returning_input_error, rmodel* m2)

cdef extern from "../src/multi_core_vae_model.h":
    void* vae_model_thread_ff(void* _args)
    void* vae_model_thread_bp(void* _args)
    void vae_model_tensor_input_ff_multicore(vaemodel** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads)
    void vae_model_tensor_input_bp_multicore(vaemodel** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error)

cdef extern from "../src/genome.h":
    ctypedef struct node:
        pass
    ctypedef struct connection:
        pass
    ctypedef struct genome:
        pass
cdef extern from "../src/feed_structure.h":
    ctypedef struct ff:
        pass
cdef extern from "../src/species.h":
    ctypedef struct species:
        pass
cdef extern from "../src/neat_structure.h":
    ctypedef struct neat:
        pass
cdef extern from "../src/neat_functions.h":
    float modified_sigmoid(float x)
    genome* init_genome(int input, int output)
    void print_genome(genome* g)
    genome* copy_genome(genome* g)
    int random_number(int min, int max)
    void init_global_params(int input, int output, int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections)
    void free_genome(genome* g,int global_inn_numb_connections)
    connection** get_connections(genome* g, int global_inn_numb_connections)
    int get_numb_connections(genome* g, int global_inn_numb_connections)
    int shuffle_node_set(node** m,int n)
    float random_float_number(float a)
    int shuffle_connection_set(connection** m,int n)
    int shuffle_genome_set(genome** m,int n)
    int save_genome(genome* g, int global_inn_numb_connections, int numb)
    genome* load_genome(int global_inn_numb_connections)
    int round_up(float num)


    void connections_mutation(genome* g, int global_inn_numb_connections, float first_thereshold, float second_thereshold)
    int split_random_connection(genome* g,int* global_inn_numb_nodes,int* global_inn_numb_connections, int** dict_connections, int*** matrix_nodes, int*** matrix_connections)
    int add_random_connection(genome* g,int* global_inn_numb_connections, int*** matrix_connections, int** dict_connections)
    int remove_random_connection(genome* g, int global_inn_numb_connections)
    genome* crossover(genome* g, genome* g2, int global_inn_numb_connections,int global_inn_numb_nodes)
    int activate_connections(genome* g, int global_inn_numb_connections,float thereshold)
    void activate_bias(genome* g)



    float* feed_forward(genome* g1, float* inputs, int global_inn_numb_nodes, int global_inn_numb_connections)
    int ff_reconstruction(genome* g, int** array, node* head, int len, ff** lists,int* size, int* global_j)
    int recursive_computation(int** array, node* head, genome* g, connection* c,float* actual_value)


    float compute_species_distance(genome* g1, genome* g2, int global_inn_numb_connections)
    species* create_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species)
    void free_species(species* s, int total_species, int global_inn_numb_connections)
    species* put_genome_in_species(genome** g, int numb_genomes, int global_inn_numb_connections, float species_thereshold, int* total_species, species** s)
    void free_species_except_for_rapresentatives(species* s, int total_species, int global_inn_numb_connections)
    int get_oldest_age(species* s, int total_species)

    float get_mean_fitness(species* s, int n_species, int oldest_age, float age_significance)
    float get_mean_specie_fitness(species* s, int i,int oldest_age, float age_significance)
    genome** sort_genomes_by_fitness(genome** g, int size)

    neat* init(int max_buffer, int input, int output)
    void neat_generation_run(neat* nes, genome** gg)
    void free_neat(neat* nes)

cdef extern from "../src/noise.h":
    oustrategy* init_oustrategy(int action_dim, float* act_max, float* act_min)
    void free_oustrategy(oustrategy* ou)
    void reset_oustrategy(oustrategy* ou)
    void evolve_state(oustrategy* ou)
    void get_action(oustrategy* ou, long long unsigned int t, float* actions)
    
cdef extern from "../src/normalization.h":
    void local_response_normalization_feed_forward(float* tensor,float* output, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k, int* used_kernels)
    void local_response_normalization_back_prop(float* tensor,float* tensor_error,float* output_error, int index_ac,int index_ai,int index_aj, int tensor_depth, int tensor_i, int tensor_j, float n_constant, float beta, float alpha, float k, int* used_kernels)
    void batch_normalization_feed_forward(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon)
    void batch_normalization_back_prop(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon)
    void channel_normalization_feed_forward(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs,float epsilon, int rows_pad, int cols_pad, int rows, int cols, int* used_kernels)
    void channel_normalization_back_prop(int batch_size, float* input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float* outputs_error, float* gamma_error, float* beta_error, float* input_error, float** temp_vectors_error,float* temp_array, float epsilon, int rows_pad, int cols_pad, int rows, int cols, int* used_kernels)
    void batch_normalization_final_mean_variance(float** input_vectors, int n_vectors, int vector_size, int mini_batch_size, bn* bn_layer)
    void group_normalization_feed_forward(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, int pad_i, int pad_j, float* post_normalization, int* used_kernels)
    void group_normalization_back_propagation(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, float* ret_error,int pad_i, int pad_j, float* input_error, int* used_kernels)
    void batch_normalization_feed_forward_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon)
    void batch_normalization_feed_forward_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs,float epsilon, int i)
    void batch_normalization_back_prop_first_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon)
    void batch_normalization_back_prop_second_step(int batch_size, float** input_vectors,float** temp_vectors, int size_vectors, float* gamma, float* beta, float* mean, float* var, float** outputs_error, float* gamma_error, float* beta_error, float** input_error, float** temp_vectors_error,float* temp_array, float epsilon, int j)
    void normalize_scores_among_fcl_layers(fcl* f)
    void normalize_scores_among_cl_layers(cl* f)
    void normalize_scores_among_all_internal_layers(model* m)
    void given_max_min_normalize_fcl(fcl* f, float max, float min)
    void given_max_min_normalize_cl(cl* f, float max, float min)
    void min_max_normalize_scores_among_all_leyers(model* m)
    void feed_forward_scaled_l2_norm(int input_dimension, float learned_g, float* norm, float* input, float* output)
    void back_propagation_scaled_l2_norm(int input_dimension,float learned_g, float* d_learned_g, float norm,float* input, float* output_error, float* input_error)
    void local_response_normalization_feed_forward_fcl(float* input,float* output,int size, float n_constant, float beta, float alpha, float k, int* used_outputs)
    void local_response_normalization_back_prop_fcl(float* input,float* input_error,float* output_error, int size, float n_constant, float beta, float alpha, float k, int* used_kernels)
    void group_normalization_feed_forward_without_learning_parameters(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, int pad_i, int pad_j, float* post_normalization, int* used_kernels, bn** bns2)
    void group_normalization_back_propagation_without_learning_parameters(float* tensor,int tensor_c, int tensor_i, int tensor_j,int n_channels, int stride, bn** bns, float* ret_error,int pad_i, int pad_j, float* input_error, int* used_kernels, bn** bns2)

cdef extern from "../src/parser.h":
    bint single_instance_single_csv_file_parser(float* input, float* output,char* filename,int input_size)
    bint single_instance_multiple_csv_file_parser(float** input, float** output,char** filename,int input_size, int n_files)
    bint multiple_instance_single_csv_file_parser(float** input, float** output,char* filename,int input_size)
    bint single_instance_single_file_parser(float* input, float* output,char* filename,int input_size)
    bint single_instance_multiple_file_parser(float** input, float** output,char** filename,int input_size, int n_files)
    bint multiple_instance_single_file_parser(float** input, float** output,char* filename,int input_size)

cdef extern from "../src/recurrent.h":
    void lstm_ff(float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int input_size, int output_size)
    void lstm_ff_edge_popup(int** w_active_output_neurons, int** u_active_output_neurons, int** w_indices,int** u_indices, float* x, float* h, float* c, float* cell_state, float* hidden_state, float** w, float** u, float** b, float** z, int input_size, int output_size, float k_percentage)
    float** lstm_bp(int flag, int input_size, int output_size, int output_size_up, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus)
    float** lstm_bp_edge_popup(int flag, int input_size, int output_size, int output_size_up, float** dw,float** du, float** db, float** w, float** u, float** z, float* dy, float* x_t, float* c_t, float* h_minus, float* c_minus, float** z_up, float** dfioc_up, float** z_plus, float** dfioc_plus, float** w_up, float* dropout_mask,float* dropout_mask_plus, int** w_active_output_neurons, int** u_active_output_neurons, int** w_indices_up, int** u_indices, float** d_w_scores, float** d_u_scores, float k_percentage, int** w_active_output_neurons_up, int** u_active_output_neurons_up)

cdef extern from "../src/recurrent_layers.h":    
    lstm* recurrent_lstm(int input_size, int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag)
    void free_recurrent_lstm(lstm* rlstm)
    void save_lstm(lstm* rlstm, int n)
    lstm* load_lstm(FILE* fr)
    lstm* copy_lstm(lstm* l)
    void paste_lstm(lstm* l,lstm* copy)
    void slow_paste_lstm(lstm* l,lstm* copy, float tau)
    lstm* reset_lstm(lstm* f)
    uint64_t get_array_size_params_lstm(lstm* f)
    void memcopy_vector_to_params_lstm(lstm* f, float* vector)
    void memcopy_params_to_vector_lstm(lstm* f, float* vector)
    void memcopy_vector_to_derivative_params_lstm(lstm* f, float* vector)
    void memcopy_derivative_params_to_vector_lstm(lstm* f, float* vector)
    void paste_w_lstm(lstm* l,lstm* copy)
    void heavy_save_lstm(lstm* rlstm, int n)
    lstm* heavy_load_lstm(FILE* fr)
    void get_used_outputs_lstm(int* arr, int input, int output, int* indices, float k_percentage)
    lstm* recurrent_lstm_without_learning_parameters (int input_size,int output_size, int dropout_flag1, float dropout_threshold1, int dropout_flag2, float dropout_threshold2, int layer, int window, int residual_flag, int norm_flag, int n_grouped_cell, int training_mode, int feed_forward_flag)
    void free_recurrent_lstm_without_learning_parameters(lstm* rlstm)
    lstm* copy_lstm_without_learning_parameters(lstm* l)
    lstm* reset_lstm_without_learning_parameters(lstm* f)
    lstm* reset_lstm_except_partial_derivatives(lstm* f)
    lstm* reset_lstm_without_dwdb(lstm* f)
    lstm* reset_lstm_without_dwdb_without_learning_parameters(lstm* f)
    uint64_t size_of_lstm(lstm* l)
    uint64_t size_of_lstm_without_learning_parameters(lstm* l)
    void paste_lstm_without_learning_parameters(lstm* l,lstm* copy)
    uint64_t count_weights_lstm(lstm* l)
    uint64_t get_array_size_params_lstm(lstm* f)
    uint64_t get_array_size_scores_lstm(lstm* f)
    uint64_t get_array_size_weights_lstm(lstm* f)
    void memcopy_params_to_vector_lstm(lstm* f, float* vector)
    void memcopy_scores_to_vector_lstm(lstm* f, float* vector)
    void memcopy_vector_to_params_lstm(lstm* f, float* vector)
    void memcopy_vector_to_weights_lstm(lstm* f, float* vector)
    void memcopy_weights_to_vector_lstm(lstm* f, float* vector)
    void memcopy_vector_to_scores_lstm(lstm* f, float* vector)
    
cdef extern from "../src/regularization.h":
    void add_l2_residual_layer(model* m,double total_number_weights,float lambda_value)
    void add_l2_convolutional_layer(model* m,double total_number_weights,float lambda_value)
    void add_l2_fully_connected_layer(model* m,double total_number_weights,float lambda_value)
    void add_l2_lstm_layer(rmodel* m,double total_number_weights,float lambda_value)

cdef extern from "../src/residual_layers.h":
    rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls)
    void free_residual(rl* r)
    void save_rl(rl* f, int n)
    void heavy_save_rl(rl* f, int n)
    rl* load_rl(FILE* fr)
    rl* heavy_load_rl(FILE* fr)
    rl* copy_rl(rl* f)
    void paste_rl(rl* f, rl* copy)
    rl* reset_rl(rl* f)
    uint64_t size_of_rls(rl* f)
    void slow_paste_rl(rl* f, rl* copy,float tau)
    uint64_t get_array_size_params_rl(rl* f)
    void memcopy_vector_to_params_rl(rl* f, float* vector)
    void memcopy_params_to_vector_rl(rl* f, float* vector)
    void memcopy_vector_to_derivative_params_rl(rl* f, float* vector)
    void memcopy_derivative_params_to_vector_rl(rl* f, float* vector)
    void set_residual_biases_to_zero(rl* r)
    int rl_adjusting_weights_after_edge_popup(rl* c, int* used_input, int* used_output)
    int* get_used_kernels_rl(rl* c, int* used_input)
    int* get_used_channels_rl(rl* c, int* used_output)
    void paste_w_rl(rl* f, rl* copy)
    void sum_score_rl(rl* input1, rl* input2, rl* output)
    void dividing_score_rl(rl* f, float value)
    void reset_score_rl(rl* f)
    void reinitialize_weights_according_to_scores_rl(rl* f, float percentage, float goodness)
    void free_residual_for_edge_popup(rl* r)
    rl* light_load_rl(FILE* fr)
    rl* light_reset_rl(rl* f)
    uint64_t get_array_size_weights_rl(rl* f)
    void memcopy_vector_to_scores_rl(rl* f, float* vector)
    void memcopy_scores_to_vector_rl(rl* f, float* vector)
    rl* copy_light_rl(rl* f)
    rl* reset_rl_for_edge_popup(rl* f)
    rl* reset_rl_without_dwdb(rl* f)
    void paste_rl_for_edge_popup(rl* f, rl* copy)
    void free_residual_complementary_edge_popup(rl* r)
    void memcopy_weights_to_vector_rl(rl* f, float* vector)
    void memcopy_vector_to_weights_rl(rl* f, float* vector)
    void compare_score_rl(rl* input1, rl* input2, rl* output)
    uint64_t get_array_size_scores_rl(rl* f)
    rl* reset_edge_popup_d_rl(rl* f)
    void set_low_score_rl(rl* f)
    rl* reset_rl_except_partial_derivatives(rl* f)
    void reinitialize_w_rl(rl* f)
    void compare_score_rl_with_vector(rl* input1, float* input2, rl* output)
    rl* copy_rl_without_learning_parameters(rl* f)
    rl* reset_rl_without_learning_parameters(rl* f)
    uint64_t size_of_rls_without_learning_parameters(rl* f)
    void paste_rl_without_learning_parameters(rl* f, rl* copy)
    void free_residual_without_learning_parameters(rl* r)
    rl* reset_rl_without_dwdb_without_learning_patameters(rl* f)
    uint64_t count_weights_rl(rl* r)
    void free_residual_without_arrays(rl* r)
    
cdef extern from "../src/rmodel.h":
    rmodel* recurrent_network(int layers, int n_lstm, lstm** lstms, int window, int hidden_state_mode)
    void free_rmodel(rmodel* m)
    rmodel* copy_rmodel(rmodel* m)
    void paste_rmodel(rmodel* m, rmodel* copy)
    void slow_paste_rmodel(rmodel* m, rmodel* copy, float tau)
    rmodel* reset_rmodel(rmodel* m)
    void save_rmodel(rmodel* m, int n)
    void heavy_save_rmodel(rmodel* m, int n)
    rmodel* load_rmodel(char* file)
    rmodel* heavy_load_rmodel(char* file)
    void ff_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, int window, int layers, lstm** lstms)
    float*** bp_rmodel_lstm(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window,int layers,lstm** lstms, float** input_error)
    uint64_t count_weights_rmodel(rmodel* m)
    void sum_rmodel_partial_derivatives(rmodel* m, rmodel* m2, rmodel* m3)
    float* lstm_dinput(int index, int output, float** returning_error, lstm* lstms)
    float* lstm_dh(int index, int output, float** returning_error, lstm* lstms)
    void ff_rmodel(float** hidden_states, float** cell_states, float** input_model, rmodel* m)
    float*** bp_rmodel(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error)
    void paste_w_rmodel(rmodel* m, rmodel* copy)
    void sum_rmodels_partial_derivatives(rmodel* m, rmodel** m2, int n_models)
    void free_rmodel_without_learning_parameters(rmodel* m)
    rmodel* copy_rmodel_without_learning_parameters(rmodel* m)
    void paste_rmodel_without_learning_parameters(rmodel* m, rmodel* copy)
    rmodel* reset_rmodel_without_learning_parameters(rmodel* m)
    void ff_rmodel_lstm_opt(float** hidden_states, float** cell_states, float** input_model, int window, int layers, lstm** lstms, lstm** lstms2)
    float*** bp_rmodel_lstm_opt(float** hidden_states, float** cell_states, float** input_model, float** error_model, int window,int layers,lstm** lstms, float** input_error, lstm** lstms2)
    float* lstm_dinput_opt(int index, int output, float** returning_error, lstm* lstms, lstm* lstms2)
    float* lstm_dh_opt(int index, int output, float** returning_error, lstm* lstms, lstm* lstms2)
    void ff_rmodel_opt(float** hidden_states, float** cell_states, float** input_model, rmodel* m, rmodel* m2)
    float*** bp_rmodel_opt(float** hidden_states, float** cell_states, float** input_model, float** error_model, rmodel* m, float** input_error, rmodel* m2)
    uint64_t size_of_rmodel(rmodel* r)
    uint64_t size_of_rmodel_without_learning_parameters(rmodel* r)
    uint64_t get_array_size_params_rmodel(rmodel* f)
    uint64_t get_array_size_weights_rmodel(rmodel* f)
    uint64_t get_array_size_scores_rmodel(rmodel* f)
    void memcopy_vector_to_params_rmodel(rmodel* f, float* vector)
    void memcopy_vector_to_weights_rmodel(rmodel* f, float* vector)
    void memcopy_vector_to_scores_rmodel(rmodel* f, float* vector)
    void memcopy_params_to_vector_rmodel(rmodel* f, float* vector)
    void memcopy_weights_to_vector_rmodel(rmodel* f, float* vector)
    void memcopy_scores_to_vector_rmodel(rmodel* f, float* vector)
    rmodel* load_rmodel_with_file_already_opened(FILE* fr)
    float* get_ith_output_cell(rmodel* r, int ith)

cdef extern from "../src/positional_encoding.h":
    float* sin_cos_positional_encoding_vector(int embedding_dimension, int sequence_length)

cdef extern from "../src/scaled_l2_norm_layers.h":
    scaled_l2_norm* scaled_l2_normalization_layer(int vector_dimension)
    void free_scaled_l2_normalization_layer(scaled_l2_norm* l2)
    void save_scaled_l2_norm(scaled_l2_norm* f, int n)
    scaled_l2_norm* load_scaled_l2_norm(FILE* fr)
    scaled_l2_norm* copy_scaled_l2_norm(scaled_l2_norm* f)
    scaled_l2_norm* reset_scaled_l2_norm(scaled_l2_norm* f)
    unsigned long long int size_of_scaled_l2_norm(scaled_l2_norm* f)
    void paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy)
    void slow_paste_scaled_l2_norm(scaled_l2_norm* f,scaled_l2_norm* copy, float tau)
    scaled_l2_norm* reset_scaled_l2_norm_except_partial_derivatives(scaled_l2_norm* f)

cdef extern from "../src/server.h":
    int run_server(int port, int max_num_conn, int* reading_pipes, int* writing_pipes, int buffer_size, char* ip)
    void* server_thread(void* _args)
    
cdef extern from "../src/struct_conn.h":
    struct_conn* structure_connection(int id, model* m1, model* m2, rmodel* r1, rmodel* r2, transformer_encoder* e1, transformer_encoder* e2, transformer_decoder* d1, transformer_decoder* d2, transformer* t1, transformer* t2, scaled_l2_norm* l1, scaled_l2_norm* l2, vector_struct* v1, vector_struct* v2, vector_struct* v3, int input1_type, int input2_type, int output_type, int* input_temporal_index,int* input_encoder_indeces, int* input_decoder_indeces_left, int* input_decoder_indeces_down, int* input_transf_encoder_indeces, int* input_transf_decoder_indeces, int* rmodel_input_left, int* rmodel_input_down, int decoder_left_input, int decoder_down_input, int transf_dec_input, int transf_enc_input, int concatenate_flag, int input_size, int model_input_index, int temporal_encoding_model_size, int vector_index)
    void reset_struct_conn(struct_conn* s)
    void struct_connection_input_arrays(struct_conn* s)
    void ff_struc_conn(struct_conn* s, int transformer_flag)
    error_super_struct* bp_struc_conn(struct_conn* s, int transformer_flag, error_super_struct* e, error_super_struct* es)
    error_super_struct* bp_struc_conn_opt(struct_conn* real_s, struct_conn* s, int transformer_flag, error_super_struct* e, error_super_struct* es)
    void ff_struc_conn_opt(struct_conn* real_s, struct_conn* s, int transformer_flag)
    void paste_struct_conn(struct_conn* s, struct_conn* copy)
    void free_struct_conn(struct_conn* s)
    
cdef extern from "../src/struct_conn_handler.h":
    struct_conn_handler* init_mother_of_all_structs(int n_inputs, int n_models,int n_rmodels,int n_encoders,int n_decoders,int n_transformers,int n_l2s,int n_vectors,int n_total_structures,int n_struct_conn,int n_targets, model** m, rmodel** r, transformer_encoder** e,transformer_decoder** d,transformer** t,scaled_l2_norm** l2,vector_struct** v,struct_conn** s, int** models, int** rmodels,int** encoders,int** decoders,int** transformers,int** l2s,int** vectors,float** targets,int* targets_index,int* targets_error_flag,float** targets_weights,float* targets_threshold1,float* targets_threshold2,float* targets_gamma, int* targets_size)
    void free_struct_conn_handler(struct_conn_handler* s)
    void free_struct_conn_handler_without_learning_parameters(struct_conn_handler* s)
    struct_conn_handler* copy_struct_conn_handler(struct_conn_handler* s)
    struct_conn_handler* copy_struct_conn_handler_without_learning_parameters(struct_conn_handler* s)
    void paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy)
    void paste_struct_conn_handler_without_learning_parameters(struct_conn_handler* s, struct_conn_handler* copy)
    void slow_paste_struct_conn_handler(struct_conn_handler* s, struct_conn_handler* copy, float tau)
    void reset_struct_conn_handler(struct_conn_handler* s)
    void reset_struct_conn_handler_without_learning_parameters(struct_conn_handler* s)
    uint64_t size_of_struct_conn_handler(struct_conn_handler* s)
    uint64_t size_of_struct_conn_handler_without_learning_parameters(struct_conn_handler* s)


cdef extern from "../src/training.h":
    training* get_training(char** chars, int** ints, float** floats, model** m, rmodel** r,int epochs, int n_char_size, int n_float_size, int n_int_size, int instance, int n_m, int m_r, int n_char, int n_float, int n_int)
    void save_training(training* t,int n)
    training* load_training(int n, int n_files)
    void standard_save_training(training* t, int n)
    training* light_load_training(int n, int n_files)

cdef extern from "../src/transformer.h":
    transformer* transf(int n_te, int n_td, transformer_encoder** te, transformer_decoder** td, int** encoder_decoder_connections)
    void free_transf(transformer* t)
    transformer* copy_transf(transformer* t)
    void paste_transformer(transformer* t, transformer* copy)
    void slow_paste_transformer(transformer* t, transformer* copy, float tau)
    void save_transf(transformer* t, int n)
    transformer* load_transf(FILE* fr)
    void reset_transf(transformer* t)
    void reset_transf_for_edge_popup(transformer* t)
    uint64_t size_of_transformer(transformer* t)
    float* get_output_layer_from_encoder_transf(transformer_encoder* t)
    void transf_ff(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag)
    float* transf_bp(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag)
    void reset_transf_decoders(transformer* t)
    void free_transf_without_learning_parameters(transformer* t)
    transformer* copy_transf_without_learning_parameters(transformer* t)
    void reset_transf_without_learning_parameters(transformer* t)
    uint64_t size_of_transformer_without_learning_parameters(transformer* t)
    void transf_ff_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag, transformer* t2)
    float* transf_bp_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag, transformer* t2)
    void paste_transformer_without_learning_parameters(transformer* t, transformer* copy)

cdef extern from "../src/transformer_decoder.h":
    transformer_decoder* transformer_decoder_layer(int input_dimension, int left_dimension, int n_head1, int n_head2, int residual_flag1, int normalization_flag1, int residual_flag2, int normalization_flag2, int residual_flag3, int normalization_flag3, int attention_flag1, int attention_flag2, int encoder_input_dimension, model* m,model* linear_after_attention1,model* linear_after_attention2, model** q,model** k, model** v, scaled_l2_norm** l2, int decoder_k_embedding, int decoder_v_embedding, int encoder_k_embedding, int encoder_v_embedding)
    void free_transformer_decoder_layer(transformer_decoder* d)
    void free_transformer_decoder_layer_without_learning_parameters(transformer_decoder* d)
    void save_transformer_decoder(transformer_decoder* t, int n)
    transformer_decoder* load_transformer_decoder(FILE* fr)
    transformer_decoder* copy_transformer_decoder(transformer_decoder* t)
    transformer_decoder* copy_transformer_decoder_without_learning_parameters(transformer_decoder* t)
    void reset_transformer_decoder(transformer_decoder* t)
    void reset_transformer_decoder_without_learning_parameters(transformer_decoder* t)
    void reset_transformer_decoder_except_partial_derivatives_and_left_input(transformer_decoder* t)
    void reset_transformer_decoder_for_edge_popup(transformer_decoder* t)
    uint64_t size_of_transformer_decoder(transformer_decoder* t)
    uint64_t size_of_transformer_decoder_without_learning_parameters(transformer_decoder* t)
    void paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy)
    void paste_transformer_decoder_without_learning_parameters(transformer_decoder* t, transformer_decoder* copy)
    void slow_paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy, float tau)
    void decoder_transformer_ff(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension)
    void decoder_transformer_ff_opt(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension, transformer_decoder* t2)
    float* decoder_transformer_bp(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error)
    float* decoder_transformer_bp_opt(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error,transformer_decoder* t2)
    void wrapped_encoder_transformer_decoder_ff(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1)
    void wrapped_encoder_transformer_decoder_ff_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1, transformer_encoder* t2)
    float* wrapped_encoder_transformer_decoder_bp(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error)
    float* wrapped_encoder_transformer_decoder_bp_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error, transformer_encoder* t2)

cdef extern from "../src/transformer_encoder.h":
    transformer_encoder* transformer_encoder_layer(model** q, model** k, model** v, model* m, model* linear_after_attention, scaled_l2_norm** l2, int input_dimension, int n_head,int residual_flag1,int normalization_flag1,int residual_flag2,int normalization_flag2, int attention_flag, int k_embedding_dimension, int v_embedding_dimension)
    void free_transformer_encoder_layer(transformer_encoder* t)
    void free_transformer_encoder_layer_without_learning_parameters(transformer_encoder* t)
    void free_transformer_wrapped_encoder_layer(transformer_encoder* t)
    void free_transformer_wrapped_encoder_layer_without_learning_parameters(transformer_encoder* t)
    void save_transformer_encoder(transformer_encoder* t, int n)
    transformer_encoder* load_transformer_encoder(FILE* fr)
    transformer_encoder* copy_transformer_encoder(transformer_encoder* t)
    transformer_encoder* copy_transformer_encoder_without_learning_parameters(transformer_encoder* t)
    void reset_transformer_encoder(transformer_encoder* t)
    void reset_transformer_encoder_without_learning_parameters(transformer_encoder* t)
    void reset_transformer_encoder_except_partial_derivatives(transformer_encoder* t) 
    void reset_transformer_encoder_for_edge_popup(transformer_encoder* t)
    uint64_t size_of_transformer_encoder(transformer_encoder* t)
    uint64_t size_of_transformer_encoder_without_learning_parameters(transformer_encoder* t)
    void paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy)
    void paste_transformer_encoder_without_learning_parameters(transformer_encoder* t, transformer_encoder* copy)
    void slow_paste_transformer_encoder(transformer_encoder* t, transformer_encoder* copy, float tau)
    void encoder_transformer_ff(float* inputs, transformer_encoder* t, int input_dimension)
    void encoder_transformer_ff_opt(float* inputs, transformer_encoder* t, int input_dimension, transformer_encoder* t2)
    float* encoder_transformer_bp(float* inputs, transformer_encoder* t, int input_dimension,float* output_error)
    float* encoder_transformer_bp_opt(float* inputs, transformer_encoder* t, int input_dimension,float* output_error, transformer_encoder* t2)

cdef extern from "../src/update.h":
    void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size)
    void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam)
    void update_residual_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size)
    void update_convolutional_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size)
    void update_fully_connected_layer_nesterov(model* m, float lr, float momentum, int mini_batch_size)
    void update_residual_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_residual_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_convolutional_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_convolutional_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_fully_connected_layer_adam(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_fully_connected_layer_adam_diff_grad(model* m, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_batch_normalized_layer_nesterov(bn** bns,int n_bn, float lr, float momentum, int mini_batch_size)
    void update_batch_normalized_layer_adam(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_batch_normalized_layer_adam_diff_grad(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2,float beta1_adam,float beta2_adam)
    void update_lstm_layer_nesterov(rmodel* m, float lr, float momentum, int mini_batch_size)
    void update_lstm_layer_adam(rmodel* m,float lr,int mini_batch_size,float b1, float b2,float beta1_adam,float beta2_adam)
    void update_lstm_layer_adam_diff_grad(rmodel* m,float lr,int mini_batch_size,float b1, float b2,float beta1_adam,float beta2_adam)
    void update_residual_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long t,float beta1_adam,float beta2_adam)
    void update_residual_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_convolutional_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam)
    void update_convolutional_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_fully_connected_layer_radam(model* m, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam)
    void update_fully_connected_layer_adamod(model* m, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_batch_normalized_layer_radam(bn** bns, int n_bn, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam)
    void update_batch_normalized_layer_adamod(bn** bns,int n_bn, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_lstm_layer_radam(rmodel* m,float lr,int mini_batch_size,float b1, float b2, unsigned long long int t,float beta1_adam,float beta2_adam)
    void update_lstm_layer_adamod(rmodel* m,float lr,int mini_batch_size,float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_scaled_l2_norm_nesterov(scaled_l2_norm* l, float lr, float momentum, int mini_batch_size)
    void update_scaled_l2_norm_adam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam)
    void update_scaled_l2_norm_adamod(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam, float beta3_adamod)
    void update_scaled_l2_norm_adam_diff_grad(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, float beta1_adam, float beta2_adam)
    void update_scaled_l2_norm_radam(scaled_l2_norm* l, float lr, int mini_batch_size, float b1, float b2, unsigned long long int t, float beta1_adam, float beta2_adam)
    void update_model(model* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* t)
    void update_rmodel(rmodel* m, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* t)
    void update_transformer(transformer* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* time)
    void update_transformer_decoder(transformer_decoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* time)
    void update_transformer_encoder(transformer_encoder* t, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* time)
    void update_vae_model(vaemodel* vm, float lr, float momentum, int mini_batch_size, int gradient_descent_flag, float* b1, float* b2, int regularization, uint64_t total_number_weights, float lambda_value, unsigned long long int* t)
    void update_training_parameters(float* beta1, float* beta2, long long unsigned int* time_step, float start_beta1, float start_beta2)

cdef extern from "../src/utils.h":
    char* get_full_path(char* directory, char* filename)
    void get_dropout_array(int size, float* mask, float* input, float* output) 
    void set_dropout_mask(int size, float* mask, float threshold) 
    void ridge_regression(float *dw, float w, float lambda_value, int n)
    int read_files(char** name, char* directory)
    char* itoa(int i, char b[])
    int shuffle_char_matrix(char** m,int n)
    int bool_is_real(float d)
    int shuffle_float_matrix(float** m,int n)
    int shuffle_int_matrix(int** m,int n)
    int shuffle_char_matrices(char** m,char** m1,int n)
    int shuffle_float_matrices(float** m,float** m1,int n)
    int shuffle_int_matrices(int** m,int** m1,int n)
    int read_file_in_char_vector(char** ksource, char* fname, int* size)
    void copy_array(float* input, float* output, int size)
    int shuffle_char_matrices_float_int_vectors(char** m,char** m1,float* f, int* v,int n)
    void copy_char_array(char* input, char* output, int size)
    int shuffle_char_matrices_float_int_int_vectors(char** m,char** m1,float* f, int* v, int* v2, int n)
    void free_matrix(void** m, int n)
    long long unsigned int** confusion_matrix(float* model_output, float* real_output, long long unsigned int** cm, int size, float threshold)
    double* accuracy_array(long long unsigned int** cm, int size)
    int shuffle_float_matrices_float_int_int_vectors(float** m,float** m1,float* f, int* v, int* v2, int n)
    int shuffle_float_matrices_float_int_vectors(float** m,float** m1,float* f, int* v,int n)
    double* precision_array(long long unsigned int** cm, int size)
    double* sensitivity_array(long long unsigned int** cm, int size)
    double* specificity_array(long long unsigned int** cm, int size)
    void print_accuracy(long long unsigned int** cm, int size)
    void print_precision(long long unsigned int** cm, int size)
    void print_sensitivity(long long unsigned int** cm, int size)
    void print_specificity(long long unsigned int** cm, int size)
    void quick_sort(float A[], int I[], int lo, int hi)
    void copy_int_array(int* input, int* output, int size)
    int shuffle_int_array(int* m,int n)
    char** get_files(int index1, int n_files)
    int check_nans_matrix(float** m, int rows, int cols)
    void merge(float* values, int* indices, int temp[], int from_index, int mid, int to, int length)
    void mergesort(float* values, int* indices, int low, int high)
    void sort(float* values, int* indices, int low, int high)
    void free_tensor(float*** t, int dim1, int dim2)
    int shuffle_float_matrix_float_tensor(float** m,float*** t,int n)
    void set_vector_with_value(float value, float* v, int dimension)
    char* read_files_from_file(char* file, int package_size)
    void set_files_free_from_file(char* file_to_free, char* file)
    void remove_occupied_sets(char* file)
    int msleep(long msec)
    int* get_new_copy_int_array(int* array, int size)
    void set_int_vector_with_value(int value, int* v, int dimension)

cdef extern from "../src/vae_model.h":
    vaemodel* variational_auto_encoder_model(model* encoder, model* decoder, int latent_size)
    void free_vae_model(vaemodel* vm)
    vaemodel* copy_vae_model(vaemodel* vm)
    void paste_vae_model(vaemodel* vm1, vaemodel* vm2)
    void slow_paste_vae_model(vaemodel* vm1, vaemodel* vm2, float tau)
    void reset_vae_model(vaemodel* vm)
    unsigned long long int size_of_vae_model(vaemodel* vm)
    void save_vae_model(vaemodel* vm, int n, int m)
    vaemodel* load_vae_model(char* file1, char* file2)
    void vae_model_tensor_input_ff(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j,float* input)
    float* vae_model_tensor_input_bp(vaemodel* vm, int tensor_depth, int tensor_i, int tensor_j, float* input, float* error, int error_dimension)
    int count_weights_vae_model(vaemodel* vm)
    void sum_vae_model_partial_derivatives(vaemodel* vm, vaemodel* vm2, vaemodel* vm3)

cdef extern from "../src/vector.h":
    vector_struct* create_vector(float* v, int v_size, int output_size, int action, int activation_flag, int dropout_flag, int index, float dropout_threshold, int input_size)
    void free_vector(vector_struct* v)
    void reset_vector(vector_struct* v)
    vector_struct* copy_vector(vector_struct* v)
    void save_vector(vector_struct* v, int n)
    vector_struct* load_vector(FILE* fr)
    void ff_vector(float* input1,float* input2, vector_struct* v)
    float* bp_vector(float* input1,float* input2, vector_struct* v, float* output_error)
    void paste_vector(vector_struct* v, vector_struct* copy)
    uint64_t size_of_vector(vector_struct* v)


cdef extern from "../src/drl.h":
    ddpg* init_ddpg(model* m1, model* m2, model* m3, model* m4, int batch_size, int threads, int regularization1,int regularization2, int m1_input,int m1_output,int m2_output,int m3_output,int gradient_descent_flag1,int gradient_descent_flag2, int buff_size, int max_frames, float lr1, float lr2, float momentum1, float momentum2, float lambda1, float lambda2, float tau,float epsilon_greedy, float lambda_value)
    void free_ddpg(ddpg* d)
    void ddpg_train(ddpg* d)
