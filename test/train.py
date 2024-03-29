import pyllab
import numpy as np

def load_data():
    f = open('./data/train.bin','rb')
    s = f.read().decode('utf-8')
    f.close()
    l_in = []
    l_out = []
    for i in range(50000):
        inputs = []
        outputs = [0,0,0,0,0,0,0,0,0,0]
        for j in range(0,28*28+1):
            if j < 28*28:
                inputs.append(int(s[i*(28*28+1)+j:i*(28*28+1)+j+1]))
            else:
                outputs[int(s[i*(28*28+1)+j:i*(28*28+1)+j+1])] = 1
        l_in.append(inputs)
        l_out.append(outputs)
    return np.array(l_in, dtype='float'), np.array(l_out,dtype="float")



batch_size = 10
epochs = 10
d = pyllab.get_dict_from_model_setup_file("./model/model_011.txt")
model = pyllab.Model(d = d)
#model.set_training_edge_popup(0.5)
model.make_multi_thread(batch_size)
model.set_model_error(pyllab.PY_FOCAL_LOSS,model.get_output_dimension_from_model(),gamma=2)
inputs, outputs = load_data()
train = pyllab.Training(lr = 0.001, momentum = 0.9,batch_size = batch_size,gradient_descent_flag = pyllab.PY_ADAM,current_beta1 = pyllab.PY_BETA1_ADAM,current_beta2 = pyllab.PY_BETA2_ADAM, regularization = pyllab.PY_NO_REGULARIZATION,total_number_weights = 0, lambda_value = 0, lr_decay_flag = pyllab.PY_LR_NO_DECAY,timestep_threshold = 0,lr_minimum = 0,lr_maximum = 1,decay = 0)
for i in range(epochs):
    model.save(i)
    for j in range(0,inputs.shape[0],batch_size):
        model.ff_error_bp_opt_multi_thread(1, 28,28, inputs[j:j+batch_size], outputs[j:j+batch_size], model.get_output_dimension_from_model())
        model.sum_models_partial_derivatives()
        train.update_model(model)
        train.update_parameters()
        model.reset()


