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

#ifndef __MULTI_CORE_MODEL_H__
#define __MULTI_CORE_MODEL_H__

void* model_thread_ff(void* _args);
void* model_thread_bp(void* _args);
void model_tensor_input_ff_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads);
void model_tensor_input_bp_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error);
void ff_error_bp_model_multicore(model** m, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error);
void* model_thread_ff_bp(void* _args);
void* model_thread_ff_opt(void* _args);
void model_tensor_input_ff_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads);
void model_tensor_input_bp_multicore_opt(model** m,model*m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** errors, int error_dimension, float** returning_error);
void* model_thread_bp_opt(void* _args);
void ff_error_bp_model_multicore_opt(model** m, model* m2, int depth, int rows, int cols, float** inputs, int mini_batch_size, int threads,float** outputs, float** returning_error);
void* model_thread_ff_bp_opt(void* _args);
model* sum_models_partial_derivatives_multithread(model** batch_m, model* m, int n, int depth);
#endif
