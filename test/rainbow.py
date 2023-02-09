import numpy as np
import gym
import pyllab
from gym import wrappers
import matplotlib as plt
import time

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
    
    def __init__(self, state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, n_actions, mode = False, rain = True):
        self.filename =filename
        self.input_size = state_size
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.mode = mode
        self.online_net = pyllab.duelingCategoricalDQN(filename = filename,input_size = state_size,action_size = action_size, n_atoms = n_atoms, v_min = v_min, v_max = v_max, mode = mode)
        self.inference_net = None
        self.target_net = pyllab.py_copy_dueling_categorical_dqn(self.online_net)
        self.online_net.make_multi_thread(batch_size)
        self.target_net.make_multi_thread(batch_size)
        if rain:
            self.rainbow = pyllab.Rainbow(online_net = self.online_net,target_net = self.target_net, threads = batch_size, batch_size = batch_size, sampling_flag = pyllab.PY_RANKED_SAMPLING, n_step_rewards = 3, stop_epsilon_greedy = 510000, epsilon = 0.000001, min_epsilon = 0.000001)
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        return self.rainbow.get_action(current_state)
    def compute_real_action(self,current_state):
        action = self.inference_net.get_best_action(current_state,self.online_net.get_input_size())
        self.inference_net.reset()
        return action
    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm

    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionary to store them
        self.rainbow.add_experience(current_state,next_state,action,reward,done)
    
    def set_inference_net(self):
        self.inference_net = pyllab.duelingCategoricalDQN(filename = self.filename,input_size = self.input_size,action_size = self.action_size, n_atoms = self.n_atoms, v_min = self.v_min, v_max = self.v_max, mode = self.mode)
        pyllab.py_paste_dueling_categorical_dqn(self.online_net,self.inference_net)
        self.inference_net.eliminate_noise_in_layers()

    def save(self, number_of_file, directory):
        self.online_net.save(number_of_file, directory)

pyllab.get_randomness()
# We create our gym environment 
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
batch_size = 128
# We define our agent
agent = DQNAgent(state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, action_size, rain = True)
total_steps = 0


def make_video(ag, directory):
    env_to_wrap = gym.make("CartPole-v1")
    env = wrappers.Monitor(env_to_wrap, directory, force=True)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state = np.array([state])
    while not done:
        action = ag.compute_real_action(state)
        state, reward, done, _ = env.step(action)
        state = np.array([state])            
        steps += 1
        rewards += reward
    print(rewards)
    env.close()
    env_to_wrap.close()
    return rewards

# We iterate over episodes
l = []
directory = 'video'
k = 1
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    
    current_state = env.reset()
    current_state = np.array([current_state])
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10
        next_state = np.array([next_state])
        # We sotre each experience in the memory buffer
        agent.store_episode(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            break
        current_state = next_state
    
    if e%100== 0 or e == n_episodes-1:
        agent.set_inference_net()
        l.append(make_video(agent,directory+str(k)))
        k+=1
print(l)

