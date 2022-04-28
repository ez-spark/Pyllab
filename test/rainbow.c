#include "../src/llab.h"

int main(){
	//char* model_string = "shared_hidden_layers;\nfully-connected;fully-connected;\nfully-connected;\n4;200;0;0;5;0;0;0;1;3;\nfully-connected;\n200;200;1;0;5;0;0;0;1;3;\nv_hidden_layers;\nfully-connected;\nfully-connected;\n200;200;0;0;5;0;0;0;1;3;\nv_linear_last_layer;\nfully-connected;\nfully-connected;\n200;51;0;0;0;0;0;0;1;3;\na_hidden_layers;\nfully-connected;\nfully-connected;\n200;200;0;0;5;0;0;0;1;3;\na_linear_last_layer;\nfully-connected;\nfully-connected;\n200;102;0;0;0;0;0;0;1;3;"
    dueling_categorical_dqn* online_net = parse_dueling_categorical_dqn_file("./model/model_027.txt");
    dueling_categorical_dqn* target_net = copy_dueling_categorical_dqn(online_net);
    int batch_size = 10,i,j;
    dueling_categorical_dqn** online_net_wlp = (dueling_categorical_dqn**)malloc(sizeof(dueling_categorical_dqn*)*batch_size);
    dueling_categorical_dqn** target_net_wlp = (dueling_categorical_dqn**)malloc(sizeof(dueling_categorical_dqn*)*batch_size);
    for(i = 0; i < batch_size; i++){
        online_net_wlp[i] = copy_dueling_categorical_dqn_without_learning_parameters(online_net);
        target_net_wlp[i] = copy_dueling_categorical_dqn_without_learning_parameters(target_net);
    } 
    
    int gd_flag = RADAM;
    int lr_decay_flag = LR_ANNEALING_DECAY;
    int feed_forward_flag = FULLY_FEED_FORWARD;
    int training_mode = GRADIENT_DESCENT;
    int clipping_flag = 0;
    int adaptive_clipping_flag = 1;
    int mini_batch_size = 20;
    int threads = batch_size;
    uint64_t diversity_driven_q_functions = 20;
    uint64_t epochs_to_copy_target = 10;
    uint64_t max_buffer_size = 40;
    uint64_t n_step_rewards = 2;
    uint64_t stop_epsilon_greedy = 10;
    uint64_t past_errors = 5;
    uint64_t lr_epoch_threshold = 1;
    float max_epsilon = 1;
    float min_epsilon = 0.1;
    float epsilon_decay = 0.0001;
    float epsilon = 0.3;
    float alpha_priorization = 0.6;
    float beta_priorization = 0.4;
    float lambda = 0.5;
    float gamma = 0.5;
    float tau = 0.8;
    float beta1 = BETA1_ADAM;
    float beta2 = BETA2_ADAM;
    float beta3 = BETA3_ADAMOD;
    float k_percentage = 1;
    float clipping_gradient_value = 0.01;
    float adaptive_clipping_gradient_value = 0.01;
    float lr = 0.001;
    float lr_minimum = 0.0001;
    float lr_maximum = 0.1;
    float lr_decay = 0.0001;
    float momentum = 0.9;
    float ddl_threshold = 1;
    srand(9);
    rainbow* r = init_rainbow(REWARD_SAMPLING,gd_flag,lr_decay_flag,feed_forward_flag,training_mode,clipping_flag,adaptive_clipping_flag,
                              mini_batch_size,threads,diversity_driven_q_functions,epochs_to_copy_target,max_buffer_size,n_step_rewards,
                              stop_epsilon_greedy,past_errors,lr_epoch_threshold,max_epsilon,min_epsilon,epsilon_decay,epsilon,alpha_priorization,
                              beta_priorization,lambda,gamma,tau,beta1,beta2,beta3,k_percentage,clipping_gradient_value,
                              adaptive_clipping_gradient_value,lr,lr_minimum,lr_maximum,lr_decay,momentum,ddl_threshold,0.05,0.05,0.05,0.05,online_net,target_net,
                              online_net_wlp,target_net_wlp);
    
    
    
    for(i = 0; i <max_buffer_size*3; i++){
        float* state = (float*)calloc(get_input_layer_size_dueling_categorical_dqn(online_net),sizeof(float));
        float* state2 = (float*)calloc(get_input_layer_size_dueling_categorical_dqn(online_net),sizeof(float));
        float* q = (float*)calloc(get_output_dimension_from_model(online_net->a_linear_last_layer),sizeof(float));
        for(j = 0; j < get_input_layer_size_dueling_categorical_dqn(online_net); j++){
            state[j] = r2();
            state2[j] = r2();
        }
        for(j = 0; j < get_output_dimension_from_model(online_net->a_linear_last_layer); j++){
            if(r2() < 0.5)
            q[j] = r2();
            else
            q[j] = r2();
        }
        
        get_action_rainbow(r,state,get_input_layer_size_dueling_categorical_dqn(online_net),1);
        add_state_plus_q_functions_to_diversity_driven_buffer(r,state2,q);
        free(q);
    }
    for(i = 0; i <max_buffer_size*3; i++){
        float* state = (float*)calloc(get_input_layer_size_dueling_categorical_dqn(online_net),sizeof(float));
        float* state2 = (float*)calloc(get_input_layer_size_dueling_categorical_dqn(online_net),sizeof(float));
        
        float reward = r2();
        if(r2() < 0.5)
            reward = reward;
        int action = rand()%2;
        int nonterminal = 1;
        int terminal = 0;
        if(r2() < 0.5){
            nonterminal = 0;
            terminal = 1;
        }
        for(j = 0; j < get_input_layer_size_dueling_categorical_dqn(online_net); j++){
            state[j] = r2();
            state2[j] = r2();
        }
        add_experience(r,state, state2,action,reward,nonterminal);
        train_rainbow(r,terminal);
    }
    
    free_rainbow(r);
    free_dueling_categorical_dqn(online_net);
    free_dueling_categorical_dqn(target_net);
    for(i = 0; i < batch_size; i++){
        free_dueling_categorical_dqn_without_learning_parameters(online_net_wlp[i]);
        free_dueling_categorical_dqn_without_learning_parameters(target_net_wlp[i]);
    }
    free(online_net_wlp);
    free(target_net_wlp);
    
}
