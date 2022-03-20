#include "../src/llab.h"

int main(){
    int size = 0;
    char* ksource = NULL;
    read_file_in_char_vector(&ksource,"./wrong_model/model_002.txt",&size);
    model* m = parse_model_without_arrays_str(ksource,size);
    free_model_without_arrays(m);
    free(ksource);
}
