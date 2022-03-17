#include "../src/llab.h"

int main(){
    dueling_categorical_dqn* dqn = parse_dueling_categorical_dqn_file("./model/model_027.txt");
    save_dueling_categorical_dqn_given_directory(dqn,0,"./");
    //dueling_categorical_dqn* dqn2 = load_dueling_categorical_dqn("2.bin");
    int batch_size = 2;
    float** states1 = (float**)malloc(sizeof(float*)*batch_size);
    float** states2 = (float**)malloc(sizeof(float*)*batch_size);
    float* rewards = (float*)calloc(batch_size,sizeof(float));
    int* actions = (int*)calloc(batch_size, sizeof(int));
    int* nonterminals = (int*)calloc(batch_size,sizeof(int));
    dueling_categorical_dqn* target = copy_dueling_categorical_dqn(dqn);
    
    dueling_categorical_dqn** dqns = (dueling_categorical_dqn**)malloc(sizeof(dueling_categorical_dqn*)*batch_size);
    dueling_categorical_dqn** targets = (dueling_categorical_dqn**)malloc(sizeof(dueling_categorical_dqn*)*batch_size);
    
    int i;
    for(i = 0; i < batch_size; i++){
        dqns[i] = copy_dueling_categorical_dqn_without_learning_parameters(dqn);
        targets[i] = copy_dueling_categorical_dqn_without_learning_parameters(target);
        states1[i] = (float*)calloc(4,sizeof(float));
        states2[i] = (float*)calloc(4,sizeof(float));
    }
    
    reset_dueling_categorical_dqn(dqn);
    dueling_categorical_dqn_train(batch_size,dqn,target,dqns,targets,states1,rewards,actions,states2,nonterminals,0.8,4);
    compute_q_functions(dqn);
    slow_paste_dueling_categorical_dqn(dqn,target,0.8);
    reset_dueling_categorical_dqn(dqn);
    sum_dueling_categorical_dqn_partial_derivatives_multithread(dqns,dqn,batch_size,0);
    for(i = 0; i < batch_size; i++){
        reset_dueling_categorical_dqn_without_learning_parameters(dqns[i]);
        free_dueling_categorical_dqn_without_learning_parameters(dqns[i]);
        free_dueling_categorical_dqn_without_learning_parameters(targets[i]);
    }
    free_dueling_categorical_dqn(dqn);
    free_dueling_categorical_dqn(target);
    free(dqns);
    free(targets);
    free_matrix((void*)states1,batch_size);
    free_matrix((void*)states2,batch_size);
    free(rewards);
    free(actions);
    free(nonterminals);
}
