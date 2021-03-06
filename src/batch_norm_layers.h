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

#ifndef __BATCH_NORM_LAYERS_H__
#define __BATCH_NORM_LAYERS_H__

bn* batch_normalization(int batch_size, int vector_input_dimension);
void free_batch_normalization(bn* b);
void save_bn(bn* b, int n);
bn* load_bn(FILE* fr);
bn* copy_bn(bn* b);
bn* reset_bn(bn* b);
uint64_t size_of_bn(bn* b);
void paste_bn(bn* b1, bn* b2);
void slow_paste_bn(bn* f, bn* copy,float tau);
bn* batch_normalization_without_learning_parameters(int batch_size, int vector_input_dimension);
bn* copy_bn_without_learning_parameters(bn* b);
uint64_t size_of_bn_without_learning_parameters(bn* b);
void paste_bn_without_learning_parameters(bn* b1, bn* b2);
bn* batch_normalization_without_arrays(int batch_size, int vector_input_dimension);
void make_the_bn_only_for_ff(bn* b);
void free_the_bn_only_for_ff(bn* b);
void paste_w_bn(bn* b1, bn* b2);
bn* reset_bn_except_partial_derivatives(bn* b);

#endif
