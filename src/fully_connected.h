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

#ifndef __FULLY_CONNECTED_H__
#define __FULLY_CONNECTED_H__

void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size);
void fully_connected_back_prop(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size, int training_flag);
void fully_connected_back_prop_edge_popup(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n);
void fully_connected_feed_forward_edge_popup(float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, int last_n);
void fully_connected_back_prop_edge_popup_ff_gd_bp(float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n);
void paste_w_fcl(fcl* f,fcl* copy);
void noisy_fully_connected_feed_forward(float* noise_biases, float* new_biases,float* noisy_biases, float* noise, float* new_weights,float* noisy_weights, float* input, float* output, float* weight,float* bias, int input_size, int output_size);
void noisy_fully_connected_feed_forward_edge_popup(float* noise, float* new_weights,float* noisy_weights, float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, int last_n);
void noisy_fully_connected_back_prop(float* noise_biases, float* new_biases,float* noisy_biases,float* noisy_biases_error, float* noise, float* new_weights,float* noisy_weights, float* noisy_weights_error, float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,int training_flag);
void noisy_fully_connected_back_prop_edge_popup(float* new_weights,float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n);
void noisy_fully_connected_back_prop_edge_popup_ff_gd_bp(float* noise, float* new_weights,float* noisy_weights, float* noisy_weights_error,float* input, float* output_error, float* weight,float* input_error, float* weight_error,float* bias_error, int input_size, int output_size,float* score_error, int* indices, int last_n);

#endif
