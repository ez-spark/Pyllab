#include "../src/llab.h"

int main(){
    int size = 0;
    char* ksource = NULL;
    read_file_in_char_vector(&ksource,"./wrong_model/model_002.txt",&size);
    model* m = parse_model_without_arrays_str(ksource,size);
    free_model_without_arrays(m);
    free(ksource);
    size = 0;
    read_file_in_char_vector(&ksource,"./wrong_model/model_003.txt",&size);
    m = parse_model_without_arrays_str(ksource,size);
    if(m == NULL)
        printf("is null\n");
    int input_size = get_input_layer_size(m);
    printf("%d \n",model_tensor_input_ff_without_arrays(m, input_size, 1, 1, NULL));
    free_model_without_arrays(m);
    free(ksource);
}
