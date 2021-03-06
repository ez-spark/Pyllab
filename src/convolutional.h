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

#ifndef __CONVOLUTIONAL_H__
#define __CONVOLUTIONAL_H__

void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1, int stride2, int padding);
void convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding);
void max_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding);
void max_pooling_back_prop(float* input, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding, float* input_error);
void average_pooling_feed_forward(float* input, float* output, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding);
void average_pooling_back_prop(float* input_error, float* output_error, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding);
void convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n);
void convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error);
void convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n);
void convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error);
void transposed_convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output, int stride1, int stride2, int padding);
void transposed_convolutional_back_prop(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding);
void transposed_convolutional_feed_forward_edge_popup(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n);
void transposed_convolutional_back_prop_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error);
void transposed_convolutional_back_prop_edge_popup_ff_gd_bp(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* bias, int channels, float* output_error, int stride1, int stride2, int padding, int* indices, int n_kernels, int last_n, float* bias_error, float** kernel_error);
void transposed_convolutional_back_prop_edge_popup_for_input(float* input, float** kernel, int input_i, int input_j, int kernel_i, int kernel_j, float bias, int channels, float* output_error,float* input_error, float* kernel_error, float* bias_error, int stride1, int stride2, int padding, float* score_error, int* indices, int n_kernels, int last_n);

#endif
