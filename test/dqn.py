import numpy as np
import gym
from matplotlib import pyplot as plt
import pyllab
from gym import wrappers

class DQNAgent:
    online_net = None
    target_net = None
    memory_buffer = None
    max_memory_buffer = None
    training = None
    lr = None
    gamma = None
    exploration_proba = None
    exploration_proba_decay = None
    batch_size = None
    n_actions = None
    
    def __init__(self, state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, n_actions, mode = False):
        self.online_net = pyllab.duelingCategoricalDQN(filename = filename,input_size = state_size,action_size = action_size, n_atoms = n_atoms, v_min = v_min, v_max = v_max, mode = mode)
        self.target_net = pyllab.py_copy_dueling_categorical_dqn(self.online_net)
        self.online_net.make_multi_thread(batch_size)
        self.target_net.make_multi_thread(batch_size)
        self.lr = 0.0001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.training = pyllab.Training(lr = self.lr, momentum = 0.9,batch_size = batch_size,gradient_descent_flag = pyllab.PY_ADAM,current_beta1 = pyllab.PY_BETA1_ADAM,current_beta2 = pyllab.PY_BETA2_ADAM, regularization = pyllab.PY_NO_REGULARIZATION,total_number_weights = 0, lambda_value = 0, lr_decay_flag = pyllab.PY_LR_NO_DECAY,timestep_threshold = 0,lr_minimum = 0,lr_maximum = 1,decay = 0)
        self.memory_buffer= list()
        self.max_memory_buffer = 20000
        self.batch_size = batch_size
        self.n_actions = n_actions
        
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        action = self.online_net.get_best_action(current_state,self.online_net.get_input_size())
        self.online_net.reset()
        return action
    def compute_real_action(self,current_state):
        action = self.online_net.get_best_action(current_state,self.online_net.get_input_size())
        self.online_net.reset()
        return action
    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
    
    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        states_t = []
        states_t_1 = []
        rewards = []
        actions = []
        nonterminals = []
        # We iterate over the selected experiences
        for experience in batch_sample:
            states_t.append(experience["current_state"][0])
            states_t_1.append(experience["next_state"][0])

            actions.append(experience["action"])
            if experience["done"] == False:
                nonterminals.append(1)
                rewards.append(experience["reward"])
            #nonterminals.append(1)
            else:
                rewards.append(-10)
                nonterminals.append(0)
        actions = np.array(actions)
        rewards = np.array(rewards)
        states_t = np.array(states_t)
        states_t_1 = np.array(states_t_1)
        nonterminals = np.array(nonterminals)
        self.online_net.train(states_t, states_t_1, rewards, actions, nonterminals, self.target_net, self.gamma)
        self.online_net.sum_dueling_categorical_dqn_partial_derivatives()
        self.online_net.adaptive_clip(0.01,1e-3);
        self.training.update_dqn_categorical_dqn(self.online_net)
        self.training.update_parameters()
        pyllab.py_slow_paste_dueling_categorical_dqn(self.online_net,self.target_net,0.8)
        self.online_net.reset()
        self.target_net.reset_all()
        
    def save(self, number_of_file, directory):
        self.online_net.save(number_of_file, directory)

pyllab.get_randomness()
# We create our gym environment 
#env = gym.make("CartPole-v1")
env = gym.make("CartPole-v1")
# We get the shape of a state and the actions space size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# Number of episodes to run
n_episodes = 500
# Max iterations per epiode
max_iteration_ep = 1000
n_atoms = 51
v_min = -10.0
v_max = 10.0
filename = "./model/model_027.txt"
batch_size = 32
# We define our agent
agent = DQNAgent(state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, action_size)
total_steps = 0


def make_video(ag):
    env  = gym.make("CartPole-v1",render_mode="human")
    rewards = 0
    steps = 0
    done = False
    state = env.reset()[0]
    state = np.array([state])
    while not done:
        action = ag.compute_real_action(state)
        state = env.step(action)
        state = state[0]
        reward = state[1]
        done = state[2]
        state = np.array([state])            
        steps += 1
        rewards += reward
        env.render()
    print(rewards)
    env.close()

# We iterate over episodes
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    
    current_state = env.reset()[0]
    current_state = np.array([current_state])
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state = env.step(action)
        next_state = next_state[0]
        reward = next_state[1]
        done = next_state[2]
        next_state = np.array([next_state])
        
        # We sotre each experience in the memory buffer
        agent.store_episode(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            agent.update_exploration_probability()
            break
        current_state = next_state
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    if total_steps >= batch_size:
        #print("train")
        agent.train()
    if e%100== 0 or e == n_episodes-1:
        #agent.save(e,'./')
        make_video(agent)
        #filename = str(e)+'.bin'
        #agent2 = DQNAgent(state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, action_size, mode = True)
        #make_video(agent2)




