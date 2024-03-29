/*
MIT License

Copyright (c) 2018 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files ((the "LICENSE")), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "llab.h"

/* This function builds a residual layer according to the rl structure defined in layers.h
 * 
 * Input:
 * 
 *             @ int channels:= n. channels of the current layer
 *             @ int input_rows:= n. rows per channel of the current layer
 *             @ int input_cols:= n. columns per channel of the current layer
 *             @ int n_cl:= number of cls structure in this residual layer
 *             @ cl** cls:= the cls structures of the layer
 * 
 * PAY ATTENTION IF YOU ARE USING EDGE POPUP ALGORITHM, YOU CAN'T USE SIGMOID FUNCTION (IN RESIDUAL NO LINEAR FUNCTION)
 * AFTER THE SUM IN THE RESIDUAL LAYER, BECAUSE RESIDUAL LAYER DOESN'T KEEP THE UNUSED
 * NEURONS FROM INPUT AND LAST CONVOLUTIONAL LAYER. SO IF THE NEURON I IS NOT USED BY LAST CONVOLUTIONAL LAYER
 * AND BY THE INPUT OF THE RESIDUAL LAYER, ITS OUTPUT IS GONNA BE 0, BUT IN THE RESIDUAL LAYER IF YOU APPLY
 * SIGMOID ITS OUTPUT IS NOT GONNA BE 0 ANYMORE. INSTEAD YOU CAN USE TANH LEAKY RELU RELU BECAUSE
 * ITS OUTPUT IS GONNA BE 0 ANYHOW IF ITS INPUT IS 0
 * 
 * */
rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls){
    if(channels<=0 || input_rows<=0 || input_cols<=0 || n_cl<=0 || cls == NULL){
        fprintf(stderr,"Error: channels, input rows, input cols params must be > 0 and or n_cl or n_fcl must be > 0\n");
        return NULL;
    }
    rl* r = (rl*)malloc(sizeof(rl));
    r->channels = channels;
    r->input_rows = input_rows;
    r->input_cols = input_cols;
    r->n_cl = n_cl;
    r->cls =cls;
    r->cl_output = convolutional_without_learning_parameters(channels,input_rows,input_cols,1,1,channels,1,1,0,0,1,1,0,0,0,0,0,RELU,0,0,CONVOLUTION,FREEZE_TRAINING,FULLY_FEED_FORWARD,cls[n_cl-1]->layer);
    return r;
    
}


/* Given a rl* structure this function frees the space allocated by this structure*/
void free_residual(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_convolutional(r->cls[i]);
    }
    
    free(r->cls);
    free_convolutional_without_learning_parameters(r->cl_output);
    free(r);
}
/* Given a rl* structure this function frees the space allocated by this structure*/
void free_residual_without_learning_parameters(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_convolutional_without_learning_parameters(r->cls[i]);
    }
    
    free(r->cls);
    free_convolutional_without_learning_parameters(r->cl_output);
    free(r);
}
/* Given a rl* structure this function frees the space allocated by this structure*/
void free_residual_without_arrays(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_convolutional_without_arrays(r->cls[i]);
    }
    
    free(r->cls);
    free_convolutional_without_learning_parameters(r->cl_output);
    free(r);
}

void free_scores_rl(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_scores_cl(r->cls[i]);
    }
}

void free_indices_rl(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        free_indices_cl(r->cls[i]);
    }
}

void set_null_scores_rl(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        set_null_scores_cl(r->cls[i]);
    }
}

void set_null_indices_rl(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        set_null_indices_cl(r->cls[i]);
    }
}


/* This function saves a residual layer on a .bin file with name n.bin
 * 
 * Input:
 * 
 *             @ rl* f:= the actual layer that must be saved
 *             @ int n:= the name of the bin file where the layer is saved
 * 
 * 
 * */
void save_rl(rl* f, int n){
    if(f == NULL || n < 0)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    
    fw = fopen(s,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s);
        exit(1);
    }
    convert_data(&f->cl_output->activation_flag,sizeof(int),1);
    i = fwrite(&f->cl_output->activation_flag,sizeof(int),1,fw);
    convert_data(&f->cl_output->activation_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a rl layer\n");
        exit(1);
    }
    convert_data(&f->channels,sizeof(int),1);
    i = fwrite(&f->channels,sizeof(int),1,fw);
    convert_data(&f->channels,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a rl layer\n");
        exit(1);
    }
    convert_data(&f->input_rows,sizeof(int),1);
    i = fwrite(&f->input_rows,sizeof(int),1,fw);
    convert_data(&f->input_rows,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a rl layer\n");
        exit(1);
    }
    convert_data(&f->input_cols,sizeof(int),1);
    i = fwrite(&f->input_cols,sizeof(int),1,fw);
    convert_data(&f->input_cols,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a rl layer\n");
        exit(1);
    }
    convert_data(&f->n_cl,sizeof(int),1);
    i = fwrite(&f->n_cl,sizeof(int),1,fw);
    convert_data(&f->n_cl,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving a rl layer\n");
        exit(1);
    }
    
    i = fclose(fw);
    if(i!=0){
        fprintf(stderr,"Error: an error occurred closing the file %s\n",s);
        exit(1);
    }
    
    for(i = 0; i < f->n_cl; i++){
        save_cl(f->cls[i],n);
    }
    
    free(s);
}


/* This function loads a residual layer from a .bin file from fr
 * 
 * Input:
 * 
 *             @ FILE* fr:= a pointer to a file already opened
 * 
 * */
rl* load_rl(FILE* fr){
    if(fr == NULL)
        return NULL;
    int i;
    
    int channels = 0,input_rows = 0,input_cols = 0,n_cl = 0, act_flag = 0;
    cl** cls;
    
    i = fread(&act_flag,sizeof(int),1,fr);
    convert_data(&act_flag,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&channels,sizeof(int),1,fr);
    convert_data(&channels,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&input_rows,sizeof(int),1,fr);
    convert_data(&input_rows,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&input_cols,sizeof(int),1,fr);
    convert_data(&input_cols,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    i = fread(&n_cl,sizeof(int),1,fr);
    convert_data(&n_cl,sizeof(int),1);
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading a rl layer\n");
        exit(1);
    }
    
    cls = (cl**)malloc(sizeof(cl*)*n_cl);
    for(i = 0; i < n_cl; i++){
        cls[i] = load_cl(fr);
    }
    
    rl* f = residual(channels,input_rows,input_cols,n_cl,cls);
    f->cl_output->activation_flag = act_flag;
    return f;
}



/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * You have a rl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 * 
 * */
rl* copy_rl(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    cl** cls = (cl**)malloc(sizeof(cl*)*f->n_cl);
    for(i = 0; i < f->n_cl; i++){
        cls[i] = copy_cl(f->cls[i]);
    }
    
    rl* copy = residual(f->channels, f->input_rows, f->input_cols, f->n_cl, cls);
    copy->cl_output->activation_flag = f->cl_output->activation_flag;
    return copy;
}
/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * You have a rl* f structure, this function creates an identical structure
 * with all the arrays used for the feed forward and back propagation
 * with all the initial states. and the same weights and derivatives in f are copied
 * into the new structure. d1 and d2 weights are used by nesterov and adam algorithms
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 * 
 * */
rl* copy_rl_without_learning_parameters(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    cl** cls = (cl**)malloc(sizeof(cl*)*f->n_cl);
    for(i = 0; i < f->n_cl; i++){
        cls[i] = copy_cl_without_learning_parameters(f->cls[i]);
    }
    
    rl* copy = residual(f->channels, f->input_rows, f->input_cols, f->n_cl, cls);
    copy->cl_output->activation_flag = f->cl_output->activation_flag;
    return copy;
}




/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl(f->cls[i]);
    }
    
    reset_cl_without_learning_parameters(f->cl_output);

    return f;
}
/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_only_for_ff(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_only_for_ff(f->cls[i]);
    }
    
    reset_cl_only_for_ff(f->cl_output);

    return f;
}
/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_without_learning_parameters(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_without_learning_parameters(f->cls[i]);
    }
    
    reset_cl_without_learning_parameters(f->cl_output);

    return f;
}
/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_except_partial_derivatives(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_except_partial_derivatives(f->cls[i]);
    }
    
    reset_cl_without_learning_parameters(f->cl_output);

    return f;
}

/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_without_dwdb(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_without_dwdb(f->cls[i]);
    }
    
    reset_cl_without_dwdb_without_learning_parameters(f->cl_output);

    return f;
}
/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_without_dwdb_without_learning_patameters(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_without_dwdb_without_learning_parameters(f->cls[i]);
    }
    
    reset_cl_without_dwdb_without_learning_parameters(f->cl_output);

    return f;
}

/* this function reset all the arrays of a residual layer
 * used during the feed forward and backpropagation
 * You have a rl* f structure, this function resets all the arrays used
 * for the feed forward and back propagation with partial derivatives D inculded
 * but the weights and D1 and D2 don't change
 * 
 * Input:
 * 
 *             @ rl* f:= a rl* f layer
 * 
 * */
rl* reset_rl_for_edge_popup(rl* f){
    if(f == NULL)
        return NULL;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_cl_for_edge_popup(f->cls[i]);
    }
    
    reset_cl_without_learning_parameters(f->cl_output);

    return f;
}



/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             rl* f:= the residual layer f
 * 
 * */
uint64_t size_of_rls(rl* f){
    if(f == NULL)
        return 0;
    int i;
    uint64_t sum = 0;
    for(i = 0; i < f->n_cl; i++){
        sum+= size_of_cls(f->cls[i]);
    }
    
    sum+= ((f->channels*f->input_cols*f->input_rows*sizeof(float)));
    sum+= size_of_cls_without_learning_parameters(f->cl_output);
    return sum;
    
}
/* this function compute the space allocated by the arrays of f
 * 
 * Input:
 * 
 *             rl* f:= the residual layer f
 * 
 * */
uint64_t size_of_rls_without_learning_parameters(rl* f){
    if (f == NULL)
        return 0;
    int i;
    uint64_t sum = 0;
    for(i = 0; i < f->n_cl; i++){
        sum+= size_of_cls_without_learning_parameters(f->cls[i]);
    }
    
    sum+= ((f->channels*f->input_cols*f->input_rows*sizeof(float)));
    sum+= size_of_cls_without_learning_parameters(f->cl_output);
    return sum;
    
}


uint64_t count_weights_rl(rl* r){
    if(r == NULL)
        return 0;
    int i;
    uint64_t sum = 0;
    for(i = 0; i < r->n_cl; i++){
        sum+=count_weights_cl(r->cls[i]);
    }
    return sum;
}
/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 * 
 * */
void paste_rl(rl* f, rl* copy){
    if(f == NULL || copy == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        paste_cl(f->cls[i],copy->cls[i]);
    }
    
    return;
}
/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 * 
 * */
void paste_rl_without_learning_parameters(rl* f, rl* copy){
    if(f == NULL || copy == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        paste_cl_without_learning_parameters(f->cls[i],copy->cls[i]);
    }
    
    return;
}



/* This function returns a rl* layer that is the same copy of the input f
 * except for the input array
 * This functions copies the weights and D and D1 and D2 into a another structure
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 * 
 * */
void paste_w_rl(rl* f, rl* copy){
    if(f == NULL || copy == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        paste_w_cl(f->cls[i],copy->cls[i]);
    }
    
    return;
}
/* This function returns a rl* layer that is the same copy for the weights and biases
 * of the layer f with the rule teta_i = tau*teta_j + (1-tau)*teta_i
 * 
 * Input:
 * 
 *             @ rl* f:= the residual layer that must be copied
 *             @ rl* copy:= the residual layer where f is copied
 *                @ float tau:= the tau param
 * */
void slow_paste_rl(rl* f, rl* copy,float tau){
    if(f == NULL || copy == NULL)
        return;
    
    int i;
    for(i = 0; i < f->n_cl; i++){
        slow_paste_cl(f->cls[i],copy->cls[i],tau);
    }
    
    return;
}



/* this function gives the number of float params for biases and weights in a rl
 * 
 * Input:
 * 
 * 
 *                 @ rl* f:= the residual layer
 * */
uint64_t get_array_size_params_rl(rl* f){
    if(f == NULL)
        return 0;
    uint64_t sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        sum+=(uint64_t)get_array_size_params_cl(f->cls[i]);
    }
    
    return sum;
}

/* this function gives the number of float params for biases and weights in a rl
 * 
 * Input:
 * 
 * 
 *                 @ rl* f:= the residual layer
 * */
uint64_t get_array_size_scores_rl(rl* f){
    if(f == NULL)
        return 0;
    uint64_t sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        sum+=(uint64_t)get_array_size_scores_cl(f->cls[i]);
    }
    
    return sum;
}
/* this function gives the number of float params for biases and weights in a rl
 * 
 * Input:
 * 
 * 
 *                 @ rl* f:= the residual layer
 * */
uint64_t get_array_size_weights_rl(rl* f){
    if(f == NULL)
        return 0;
    uint64_t sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        sum+=(uint64_t)get_array_size_weights_cl(f->cls[i]);
    }
    
    return sum;
}

void make_the_rl_only_for_ff(rl* r){
	int i;
	for(i = 0; i < r->n_cl; i++){
		make_the_cl_only_for_ff(r->cls[i]);
	}
}
/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_params_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_params_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
}

/* this function paste the weights in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_weights_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_weights_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_weights_cl(f->cls[i]);
    }
}

/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_scores_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_scores_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}
/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_indices_rl2(rl* f, int* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_indices_cl2(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}
/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void assign_vector_to_scores_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        assign_vector_to_scores_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}

/* this function paste the weights and biases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ int* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_indices_rl(rl* f, int* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_indices_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}
/* this function paste the vector in the weights and biases of the rl
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_params_to_vector_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_params_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
}

/* this function paste the vector in the weights of the rl
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_weights_to_vector_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_weights_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_weights_cl(f->cls[i]);
    }
}

/* this function paste the vector in the weights and biases of the rl
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_scores_to_vector_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_scores_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}

/* this function paste the vector in the weights and biases of the rl
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_indices_to_vector_rl(rl* f, int* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_indices_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_scores_cl(f->cls[i]);
    }
}

/* this function paste the dweights and dbiases in a single vector
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_vector_to_derivative_params_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_vector_to_derivative_params_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
}


/* this function paste the vector in the dweights and dbiases of the rl
 * 
 * Inputs:
 * 
 * 
 *                 @ rl* f:= the residual layer
 *                 @ float* vector:= the vector where is copyed everything
 * */
void memcopy_derivative_params_to_vector_rl(rl* f, float* vector){
    if(f == NULL || vector == NULL)
        return;
    int sum = 0,i;
    for(i = 0; i < f->n_cl; i++){
        memcopy_derivative_params_to_vector_cl(f->cls[i],&vector[sum]);
        sum += get_array_size_params_cl(f->cls[i]);
    }
}

/* setting the biases of convolutional layers inside residual ones to 0
 * 
 * Input:
 *             @ rl* r:= the residual layer
 * */
void set_residual_biases_to_zero(rl* r){
    if(r == NULL)
        return;
    int i;
    for(i = 0; i < r->n_cl; i++){
        set_convolutional_biases_to_zero(r->cls[i]);
    }
}


/* this function returns an array of the used kernels (output of rl layer) from the input and output layer
 * (we must consider also the end of the layer cause there is an addition)
 * 
 * Inputs:
 *             
 *             @ rl* c:= the residual layer
 *             @ int* used_input:= the used input at the end of the rl layer must be != NULL
 * */
int* get_used_kernels_rl(rl* c, int* used_input){
    if(c == NULL || used_input == NULL)
        return NULL;
    int* ch = (int*)calloc(c->channels,sizeof(int));
    copy_int_array(c->cls[c->n_cl-1]->used_kernels,ch,c->channels);
    int i;
    for(i = 0; i < c->channels; i++){
        if(used_input[i])
            ch[i] = 1;
    }
    
    return ch;
}



/* this function sum up all the scores of input1 and input2 of the convolutional layer isnide them
 * to output
 * 
 * 
 * Input:
 * 
 *                 @ rl* input1:= the first input residual layer
 *                 @ rl* input2:= the second input residual layer
 *                 @ rl* output:= the output residual layer
 * */
void sum_score_rl(rl* input1, rl* input2, rl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < input1->n_cl; i++){
        sum_score_cl(input1->cls[i],input2->cls[i],output->cls[i]);
    }
}
/* this function sum up all the scores of input1 and input2 of the convolutional layer isnide them
 * to output
 * 
 * 
 * Input:
 * 
 *                 @ rl* input1:= the first input residual layer
 *                 @ rl* input2:= the second input residual layer
 *                 @ rl* output:= the output residual layer
 * */
void compare_score_rl(rl* input1, rl* input2, rl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < input1->n_cl; i++){
        compare_score_cl(input1->cls[i],input2->cls[i],output->cls[i]);
    }
}

/* this function sum up all the scores of input1 and input2 of the convolutional layer isnide them
 * to output
 * 
 * 
 * Input:
 * 
 *                 @ rl* input1:= the first input residual layer
 *                 @ float* input2:= the vector
 *                 @ rl* output:= the output residual layer
 * */
void compare_score_rl_with_vector(rl* input1, float* input2, rl* output){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    uint64_t sum = 0;
    for(i = 0; i < input1->n_cl; i++){
        compare_score_cl_with_vector(input1->cls[i],&input2[sum],output->cls[i]);
        sum+=get_array_size_scores_cl(input1->cls[i]);
    }
}

/* this function divides all the scores of the convolutional layer inside the reisidual one
 * with vale
 * 
 * Input:
 *             
 *                 @ rl* f:= the residual layer
 *                 @ float value:= the value
 * */
void dividing_score_rl(rl* f, float value){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        dividing_score_cl(f->cls[i],value);
    }
}


/* this function sets all the scores of the convolutional layer inside the residual one to 0
 * 
 * Input:
 * 
 *                 @ rl* f:= the reisdual layer
 * */
void reset_score_rl(rl* f){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_score_cl(f->cls[i]);
    }
}

/* this function reinitialize the worst weights inside each cl layer
 * layer inside the residual one look at reinitialize_scores_cl function in convolutional_layers.c
 * for more details
 * */
void reinitialize_weights_according_to_scores_rl(rl* f, float percentage, float goodness){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reinitialize_weights_according_to_scores_cl(f->cls[i],percentage,goodness);
    }
}

/* this function reinitialize the worst weights inside each cl layer
 * layer inside the residual one look at reinitialize_scores_cl function in convolutional_layers.c
 * for more details
 * */
void reinitialize_weights_according_to_scores_rl_only_percentage(rl* f, float percentage){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reinitialize_weights_according_to_scores_cl_only_percentage(f->cls[i],percentage);
    }
}

/* this function reinitialize the worst weights inside each cl layer
 * layer inside the residual one look at reinitialize_scores_cl function in convolutional_layers.c
 * for more details
 * */
void reinitialize_weights_according_to_scores_and_inner_info_rl(rl* f){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reinitialize_weights_according_to_scores_and_inner_info_cl(f->cls[i]);
    }
}

/* this function re initialize the weights of a residual layer
 * 
 * Inputs:
 * 
 * 
 *             @ rl* f:= the residual layer
 * */
void reinitialize_w_rl(rl* f){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reinitialize_w_cl(f->cls[i]);
    }
}

rl* reset_edge_popup_d_rl(rl* f){
    if (f == NULL)
        return NULL;
    int i;
    for(i = 0; i < f->n_cl; i++){
        reset_edge_popup_d_cl(f->cls[i]);
    }
    return f;
}

/* this function sets all the scores of the convolutional layer inside the residual one to 0
 * 
 * Input:
 * 
 *                 @ rl* f:= the reisdual layer
 * */
void set_low_score_rl(rl* f){
    if(f == NULL)
        return;
    int i;
    for(i = 0; i < f->n_cl; i++){
        set_low_score_cl(f->cls[i]);
    }
}
