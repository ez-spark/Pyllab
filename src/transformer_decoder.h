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

#ifndef __TRANSFORMER_DECODER_H__
#define __TRANSFORMER_DECODER_H__


transformer_decoder* transformer_decoder_layer(int input_dimension, int left_dimension, int n_head1, int n_head2, int residual_flag1, int normalization_flag1, int residual_flag2, int normalization_flag2, int residual_flag3, int normalization_flag3, int attention_flag1, int attention_flag2, int encoder_input_dimension, model* m,model* linear_after_attention1,model* linear_after_attention2, model** q,model** k, model** v, scaled_l2_norm** l2, int decoder_k_embedding, int decoder_v_embedding, int encoder_k_embedding, int encoder_v_embedding);
void free_transformer_decoder_layer(transformer_decoder* d);
void free_transformer_decoder_layer_without_learning_parameters(transformer_decoder* d);
void save_transformer_decoder(transformer_decoder* t, int n);
transformer_decoder* load_transformer_decoder(FILE* fr);
transformer_decoder* copy_transformer_decoder(transformer_decoder* t);
transformer_decoder* copy_transformer_decoder_without_learning_parameters(transformer_decoder* t);
void reset_transformer_decoder(transformer_decoder* t);
void reset_transformer_decoder_without_learning_parameters(transformer_decoder* t);
void reset_transformer_decoder_except_partial_derivatives_and_left_input(transformer_decoder* t);
void reset_transformer_decoder_for_edge_popup(transformer_decoder* t);
uint64_t size_of_transformer_decoder(transformer_decoder* t);
uint64_t size_of_transformer_decoder_without_learning_parameters(transformer_decoder* t);
void paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy);
void paste_transformer_decoder_without_learning_parameters(transformer_decoder* t, transformer_decoder* copy);
void slow_paste_transformer_decoder(transformer_decoder* t, transformer_decoder* copy, float tau);
void decoder_transformer_ff(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension);
void decoder_transformer_ff_opt(float* inputs1, float* inputs2, transformer_decoder* t,int input1_dimension, int input2_dimension, transformer_decoder* t2);
float* decoder_transformer_bp(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error);
float* decoder_transformer_bp_opt(float* inputs1, float* inputs2, transformer_decoder* t, int input1_dimension, int input2_dimension, float* output_error, float* inputs2_error,transformer_decoder* t2);
void wrapped_encoder_transformer_decoder_ff(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1);
void wrapped_encoder_transformer_decoder_ff_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1, transformer_encoder* t2);
float* wrapped_encoder_transformer_decoder_bp(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error);
float* wrapped_encoder_transformer_decoder_bp_opt(float* inputs1, float* inputs2, transformer_encoder* t, int input_dimension2,int input_dimension1,float* output_error,float* encoder_error, transformer_encoder* t2);


#endif
