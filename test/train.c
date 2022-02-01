#include "../src/llab.h"

int main( int argc,char** argv){
    //./train ./model/model_test_001.txt ADAM(1) MSE_LOSS(2) k_percentage
    if (argc < 3){
        fprintf(stderr,"Error, you should pass the model file and the optimizer (NESTEROV 1,ADAM 2,RADAM 3,DIFF_GRAD 4,ADAMOD 5\n");
        exit(1);
    }
    
    char* model_filename = argv[1];
    int optimizer = atoi(argv[2]);
    float edge = 0;
    
    
    int value = 0;
    if (argc >= 4){
        value = atoi(argv[3]);
        if(value != CROSS_ENTROPY_LOSS && value != FOCAL_LOSS && value != HUBER1_LOSS && value != HUBER2_LOSS && value != MSE_LOSS && value != ENTROPY_LOSS){
            fprintf(stderr, "Error: no loss found\n");
            exit(1);
        }
    }
    
    if (argc == 5){
        edge = atof(argv[4]);
    }
    
    if(optimizer != NESTEROV && optimizer != ADAM && optimizer != RADAM && optimizer != DIFF_GRAD && optimizer != ADAMOD){
        fprintf(stderr,"Error: optimizer not recognized!\n");
        exit(1);
    }
    srand(time(NULL));
    // Initializing Training resources
    int i,j,k,z,training_instances = 50000,input_dimension = 784,output_dimension = 10;
    int batch_size = 10,threads = batch_size;
    int epochs = 2;
    unsigned long long int t = 1;
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "./data/train.bin";
    int size = 0;
    char temp[2];
    temp[1] = '\0';
    float** errors = (float**)malloc(sizeof(float*)*batch_size);
    
    for(i = 0; i < batch_size; i++){
        errors[i] = (float*)calloc(output_dimension,sizeof(float));
    }
    // Model Architecture
    model* m = parse_model_file(model_filename);
    set_model_error(m,value,1,4,2,NULL,output_dimension);
    if(edge > 0)
        set_model_training_edge_popup(m,edge);
    //m->fcls[m->n_fcl-1]->k_percentage = 1;
    //m->fcls[m->n_fcl-2]->k_percentage = 1;
    model** batch_m = (model**)malloc(sizeof(model*)*batch_size);
    float** ret_err = (float**)malloc(sizeof(float*)*batch_size);
    for(i = 0; i < batch_size; i++){
        batch_m[i] = copy_model_without_learning_parameters(m);
    }
    int ws = count_weights(m);
    float lr = 0.0003, momentum = 0.9, lambda = 0.0003;
    if(edge > 0)
        lr = 0.01;
    // Reading the data in a char** vector
    read_file_in_char_vector(ksource,filename,&size);
    float** inputs = (float**)malloc(sizeof(float*)*training_instances);
    float** outputs = (float**)malloc(sizeof(float*)*training_instances);
    // Putting the data in float** vectors
    for(i = 0; i < training_instances; i++){
        inputs[i] = (float*)malloc(sizeof(float)*input_dimension);
        outputs[i] = (float*)calloc(output_dimension,sizeof(float));
        for(j = 0; j < input_dimension+1; j++){
            temp[0] = ksource[0][i*(input_dimension+1)+j];
            if(j == input_dimension)
                outputs[i][atoi(temp)] = 1;
            else
                inputs[i][j] = atof(temp);
        }
    }
    
    float b1 = m->beta1_adam;
    float b2 = m->beta2_adam;
    
    printf("Training phase!\n");
    save_model(m,0);
    // Training
    for(k = 0; k < epochs; k++){
        printf("Starting epoch %d/%d\n",k+1,epochs);
        // Shuffling before each epoch
        shuffle_float_matrices(inputs,outputs,training_instances);
        for(i = 0; i < training_instances; i+=batch_size){
            ff_error_bp_model_multicore_opt(batch_m,m,1,28,28,inputs+i,batch_size,batch_size,outputs+i,NULL);
            sum_models_partial_derivatives_multithread(batch_m,m,batch_size,0);
            update_model(m,lr,momentum,batch_size,optimizer,&b1,&b2,NO_REGULARIZATION,0,0,&t);
            update_training_parameters(&b1,&b2,&t,m->beta1_adam,m->beta2_adam);
            reset_model(m);
            for(j = 0; j < batch_size; j++){
                reset_model_without_learning_parameters(batch_m[j]);
            }
        }
        // Saving the model
        save_model(m,k+1);
    }
    
    // Deallocating Training resources
    free(ksource[0]);
    free(ksource);
    free_model(m);
    for(i = 0; i < batch_size; i++){
        free_model_without_learning_parameters(batch_m[i]);
        free(errors[i]);
    }
    free(errors);
    free(batch_m);
    free(ret_err);
    for(i = 0; i < training_instances; i++){
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    
    // Initializing Testing resources
    model* test_m;
    char** ksource2 = (char**)malloc(sizeof(char*));
    char* filename2 = "./data/train.bin";
    int size2 = 0;
    int testing_instances = 50000;
    char temp2[256];
    read_file_in_char_vector(ksource2,filename2,&size);
    float** inputs_test = (float**)malloc(sizeof(float*)*testing_instances);
    float** outputs_test = (float**)malloc(sizeof(float*)*testing_instances);
    // Putting the data in float** vectors
    for(i = 0; i < testing_instances; i++){
        inputs_test[i] = (float*)malloc(sizeof(float)*input_dimension);
        outputs_test[i] = (float*)calloc(output_dimension,sizeof(float));
        for(j = 0; j < input_dimension+1; j++){
            temp[0] = ksource2[0][i*(input_dimension+1)+j];
            if(j == input_dimension)
                outputs_test[i][atoi(temp)] = 1;
            else
                inputs_test[i][j] = atof(temp);
        }
    }
    
    
    long long unsigned int** cm;
    
    printf("Testing phase!\n");
    double error = 0;
    // Testing
    for(k = 0; k < epochs+1; k++){
        printf("Model N. %d/%d\n",k+1,epochs);
        // Loading the model
        char temp3[5];
        temp3[0] = '.';
        temp3[1] = 'b';
        temp3[2] = 'i';
        temp3[3] = 'n';
        temp3[4] = '\0';
        itoa(k,temp2);
        strcat(temp2,temp3);
        test_m = load_model(temp2);
        if(edge > 0){
            set_model_training_edge_popup(test_m,edge);
        }
        for(i = 0; i < test_m->n_fcl; i++){
            if(test_m->fcls[i]->dropout_flag == DROPOUT){
                test_m->fcls[i]->dropout_flag = DROPOUT_TEST;
                test_m->fcls[i]->dropout_threshold = 1-test_m->fcls[i]->dropout_threshold;
            }
        }
        reset_model(test_m);
        
        make_the_model_only_for_ff(test_m);
        for(i = 0; i < testing_instances; i++){
            // Feed forward
            
            model_tensor_input_ff(test_m,input_dimension,1,1,inputs_test[i]);
            for(j = 0; j < output_dimension; j++){
                if(value == FOCAL_LOSS)
                    error+=focal_loss(test_m->output_layer[j],outputs_test[i][j],2);
                else if(value == CROSS_ENTROPY_LOSS)
                    error+=cross_entropy(test_m->output_layer[j],outputs_test[i][j]);
                else if(value == MSE_LOSS)
                    error+=mse(test_m->output_layer[j],outputs_test[i][j]);
                else if(value == HUBER1_LOSS)
                    error+=huber_loss(test_m->output_layer[j],outputs_test[i][j],1);
                else if(value == HUBER2_LOSS)
                    error+=modified_huber_loss(test_m->output_layer[j],outputs_test[i][j],1,4);
            }
              
            if(!i)
                cm = confusion_matrix(test_m->output_layer, outputs_test[i],NULL, 10,0.5);
            else
                cm = confusion_matrix(test_m->output_layer, outputs_test[i],cm, 10,0.5);
            reset_model(test_m);
        }
        printf("Error: %lf\n",error);
        printf("Accuracy, Precision, Sensitivity, Specificity:\n");
        print_accuracy(cm,output_dimension);
        print_precision(cm,output_dimension);
        print_sensitivity(cm,output_dimension);
        print_specificity(cm,output_dimension);
        for(i = 0; i < output_dimension*2; i++){
            free(cm[i]);
        }
        free(cm);
        error = 0;
        free_model(test_m);
    }
    // Deallocating testing resources
    free(ksource2[0]);
    free(ksource2);
    for(i = 0; i < testing_instances; i++){
        free(inputs_test[i]);
        free(outputs_test[i]);
    }
    free(inputs_test);
    free(outputs_test);
}
