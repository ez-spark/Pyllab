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

#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

transformer* transf(int n_te, int n_td, transformer_encoder** te, transformer_decoder** td, int** encoder_decoder_connections);
void free_transf(transformer* t);
transformer* copy_transf(transformer* t);
void paste_transformer(transformer* t, transformer* copy);
void slow_paste_transformer(transformer* t, transformer* copy, float tau);
void save_transf(transformer* t, int n);
transformer* load_transf(FILE* fr);
void reset_transf(transformer* t);
void reset_transf_for_edge_popup(transformer* t);
uint64_t size_of_transformer(transformer* t);
float* get_output_layer_from_encoder_transf(transformer_encoder* t);
void transf_ff(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag);
float* transf_bp(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag);
void reset_transf_decoders(transformer* t);
void free_transf_without_learning_parameters(transformer* t);
transformer* copy_transf_without_learning_parameters(transformer* t);
void reset_transf_without_learning_parameters(transformer* t);
uint64_t size_of_transformer_without_learning_parameters(transformer* t);
void transf_ff_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, int flag, transformer* t2);
float* transf_bp_opt(transformer* t, float* inputs_encoder, int input_dimension1, float* inputs_decoder, int input_dimension2, float* output_error, int flag, transformer* t2);
void paste_transformer_without_learning_parameters(transformer* t, transformer* copy);

#endif
