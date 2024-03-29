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

#ifndef __RESIDUAL_LAYERS_H__
#define __RESIDUAL_LAYERS_H__

rl* residual(int channels, int input_rows, int input_cols, int n_cl, cl** cls);
void free_residual(rl* r);
void save_rl(rl* f, int n);
void heavy_save_rl(rl* f, int n);
rl* load_rl(FILE* fr);
rl* heavy_load_rl(FILE* fr);
rl* copy_rl(rl* f);
void paste_rl(rl* f, rl* copy);
rl* reset_rl(rl* f);
uint64_t size_of_rls(rl* f);
void slow_paste_rl(rl* f, rl* copy,float tau);
uint64_t get_array_size_params_rl(rl* f);
void memcopy_vector_to_params_rl(rl* f, float* vector);
void memcopy_params_to_vector_rl(rl* f, float* vector);
void memcopy_vector_to_derivative_params_rl(rl* f, float* vector);
void memcopy_derivative_params_to_vector_rl(rl* f, float* vector);
void set_residual_biases_to_zero(rl* r);
int rl_adjusting_weights_after_edge_popup(rl* c, int* used_input, int* used_output);
int* get_used_kernels_rl(rl* c, int* used_input);
int* get_used_channels_rl(rl* c, int* used_output);
void paste_w_rl(rl* f, rl* copy);
void sum_score_rl(rl* input1, rl* input2, rl* output);
void dividing_score_rl(rl* f, float value);
void reset_score_rl(rl* f);
void reinitialize_weights_according_to_scores_rl(rl* f, float percentage, float goodness);
void free_residual_for_edge_popup(rl* r);
rl* light_load_rl(FILE* fr);
rl* light_reset_rl(rl* f);
uint64_t get_array_size_weights_rl(rl* f);
void memcopy_vector_to_scores_rl(rl* f, float* vector);
void memcopy_scores_to_vector_rl(rl* f, float* vector);
rl* copy_light_rl(rl* f);
rl* reset_rl_for_edge_popup(rl* f);
rl* reset_rl_without_dwdb(rl* f);
void paste_rl_for_edge_popup(rl* f, rl* copy);
void free_residual_complementary_edge_popup(rl* r);
void memcopy_weights_to_vector_rl(rl* f, float* vector);
void memcopy_vector_to_weights_rl(rl* f, float* vector);
void compare_score_rl(rl* input1, rl* input2, rl* output);
uint64_t get_array_size_scores_rl(rl* f);
rl* reset_edge_popup_d_rl(rl* f);
void set_low_score_rl(rl* f);
rl* reset_rl_except_partial_derivatives(rl* f);
void reinitialize_w_rl(rl* f);
void compare_score_rl_with_vector(rl* input1, float* input2, rl* output);
rl* copy_rl_without_learning_parameters(rl* f);
rl* reset_rl_without_learning_parameters(rl* f);
uint64_t size_of_rls_without_learning_parameters(rl* f);
void paste_rl_without_learning_parameters(rl* f, rl* copy);
void free_residual_without_learning_parameters(rl* r);
rl* reset_rl_without_dwdb_without_learning_patameters(rl* f);
uint64_t count_weights_rl(rl* r);
void free_residual_without_arrays(rl* r);
void reinitialize_weights_according_to_scores_and_inner_info_rl(rl* f);
void memcopy_vector_to_indices_rl(rl* f, int* vector);
void memcopy_indices_to_vector_rl(rl* f, int* vector);
void free_scores_rl(rl* r);
void free_indices_rl(rl* r);
void assign_vector_to_scores_rl(rl* f, float* vector);
void set_null_scores_rl(rl* r);
void set_null_indices_rl(rl* r);
void reinitialize_weights_according_to_scores_rl_only_percentage(rl* f, float percentage);
void memcopy_vector_to_indices_rl2(rl* f, int* vector);
void make_the_rl_only_for_ff(rl* r);
rl* reset_rl_only_for_ff(rl* f);

#endif
