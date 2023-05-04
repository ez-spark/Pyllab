import numpy as np
import gym
import pyllab
from gym import wrappers
import matplotlib as plt
import time
import cv2
from PIL import Image

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
        qr_dqn = True
        self.online_net = pyllab.duelingCategoricalDQN(filename = filename,qr_dqn = qr_dqn, input_size = state_size,action_size = action_size, n_atoms = n_atoms, v_min = v_min, v_max = v_max, mode = mode)
        self.inference_net = None
        self.target_net = pyllab.py_copy_dueling_categorical_dqn(self.online_net)
        self.online_net.make_multi_thread(batch_size)
        self.target_net.make_multi_thread(batch_size)
        if rain:
            self.rainbow = pyllab.Rainbow(online_net = self.online_net,target_net = self.target_net, threads = batch_size, batch_size = batch_size, diversity_driven_q_functions = 10*batch_size, epsilon = 0.000001)
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        return self.rainbow.get_action(current_state)
    def compute_real_action(self,current_state):
        action = self.online_net.get_best_action(current_state,self.online_net.get_input_size())
        self.online_net.reset()
        return action

    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done, train_flag):
        self.rainbow.add_experience(current_state,next_state,action,reward,done, train_flag = train_flag)
        if done:
            self.rainbow.update_exploration_probability()
    
    def set_inference_net(self):
        self.inference_net = pyllab.py_copy_dueling_categorical_dqn(self.online_net)
        self.inference_net.eliminate_noise_in_layers()

    def save(self, number_of_file):
        self.online_net.save_from_number(number_of_file)

pyllab.get_randomness()
# We create our gym environment 
env = gym.make("LunarLander-v2")
# We get the shape of a state and the actions space size

state_size = 8
action_size = 4

# Number of episodes to run
n_episodes = 1000
# Max iterations per epiode
max_iteration_ep = 500
n_atoms = 51
v_min = -10.0
v_max = 10.0
filename = "./model/model_041.txt"
batch_size = 128
# We define our agent
agent = DQNAgent(state_size, action_size, n_atoms, v_min, v_max, filename, batch_size, action_size, rain = True)
total_steps = 0


def make_video(ag, directory):
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    #env = gym.make("CartPole-v1")
    rewards = 0
    steps = 0
    done = False
    state = env.reset()[0]
    state = np.array([state])
    l = []
    shape = env.render().shape
    for i in range(max_iteration_ep):
        l.append(env.render())
        action = ag.compute_real_action(state)
        t = env.step(action)
        state = t[0]
        reward = t[1]
        done = t[2] 
        state = np.array([state])            
        steps += 1
        rewards += reward
        if done:
            break

    out = cv2.VideoWriter('output'+directory+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (shape[1],shape[0]))

    for img in l:
        out.write(img)

    out.release()
    print(rewards)
    env.close()

    
    
    
    #env_to_wrap.close()
    return rewards

# We iterate over episodes
l = []
directory = 'video'
k = 1
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    print("Episode: "+str(e)+"/"+str(n_episodes))
    current_state = env.reset()[0]
    current_state = np.array([current_state])
    done = False
    for step in range(max_iteration_ep):
        total_steps += 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        t = env.step(action)
        next_state = t[0]
        reward = t[1]
        done = t[2] 
        next_state = np.array([next_state])
        # We sotre each experience in the memory buffer
        train_flag = False
        if reward < -20:
            reward = -20
            
        if reward > 20:
            reward = 20
            
        if total_steps%20 == 0 or done:
            train_flag = True
        if not done and step == max_iteration_ep-1:
            reward = -20
        agent.store_episode(current_state, action, reward, next_state, done, train_flag)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            break
        current_state = next_state
    
    if e%20== 0 or e == n_episodes-1:
        agent.set_inference_net()
        #agent.save(e);
        l.append(make_video(agent,str(k)))
        k+=1
print(l)

