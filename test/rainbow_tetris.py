import numpy as np
import gym
from matplotlib import pyplot as plt
import pyllab
from gym import wrappers
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

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
        self.online_net = pyllab.duelingCategoricalDQN(filename = filename,input_size = state_size,action_size = action_size, n_atoms = n_atoms, v_min = v_min, v_max = v_max, mode = mode)
        self.target_net = pyllab.py_copy_dueling_categorical_dqn(self.online_net)
        self.online_net.make_multi_thread(batch_size)
        self.target_net.make_multi_thread(batch_size)
        if rain:
            self.rainbow = pyllab.Rainbow(online_net = self.online_net,target_net = self.target_net, max_buffer_size = 10000, sampling_flag = pyllab.PY_REWARD_SAMPLING, threads = batch_size, batch_size = batch_size, stop_epsilon_greedy =10000,adaptive_clipping_flag = 0, diversity_driven_q_functions = batch_size)
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        return self.rainbow.get_action(current_state)
    def compute_real_action(self,current_state):
        action = self.online_net.get_best_action(current_state,self.online_net.get_input_size())
        self.online_net.reset()
        return action
    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm

    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done, train_flag = False):
        #We use a dictionnary to store them
        self.rainbow.add_experience(current_state,next_state,action,reward,done, train_flag)

    def save(self, number_of_file, directory):
        self.online_net.save(number_of_file, directory)

pyllab.get_randomness()
# We create our gym environment 
env = gym.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
#env = gym.make("CartPole-v1")
#env = gym.make("CartPole-v1")
# We get the shape of a state and the actions space size

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Number of episodes to run
n_episodes = 1000
# Max iterations per epiode
max_iteration_ep = 5000
n_atoms = 51
v_min = -10.0
v_max = 10.0
filename = "./model/model_030.txt"
batch_size = 64
# We define our agent
agent = DQNAgent(state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, action_size, rain = True)
total_steps = 0


def make_video(ag):
    env = gym.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)
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
        env.render()
    env.close()
    env_to_wrap.close()
    return rewards

# We iterate over episodes
l = []
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    
    current_state = env.reset()
    current_state = np.array([current_state])
    
    for step in range(max_iteration_ep):
        print("e: "+str(e)+", step: "+str(step))
        total_steps = total_steps + 1
        # the agent computes the action to perform
        cs = current_state.flatten()
        #print(current_state.shape)
        #exit(0)
        #print(step)
        #print(cs.shape)
        action = agent.compute_action(cs)
        
        #print(step+1)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state, reward, done, _ = env.step(action)
        #env.render()
        next_state = np.array([next_state])
        ns = next_state.flatten()
        # We sotre each experience in the memory buffer
        if(step > 0 and step%20 == 0):
            agent.store_episode(cs, action, reward, ns, done, train_flag = True)
        else:
            agent.store_episode(cs, action, reward, ns, done, train_flag = False)
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            break
        current_state = next_state
    
    if e%100== 0 or e == n_episodes-1:
        agent.save(e,'./')

l2 = []
for i in range(len(l)):
    l2.append(i+1)
plt.scatter(l2, l)
plt.savefig("rainbow.png")
