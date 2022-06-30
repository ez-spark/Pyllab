import numpy as np
import gym
import pyllab
from gym import wrappers
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
env = gym.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
action_size = env.action_space.n
pyllab.get_randomness()
neat = pyllab.Neat(state_size,action_size)
threads = 10
max_iterations = 1000

def convert_array(arr):
    l = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j][0] == 0 and arr[i][j][1] == 0 and arr[i][j][2] == 0:
                l.append(0)
            else:
                l.append(1)
    return np.array(l, dtype=np.float32)
    
def make_video(genome):
    env = gym.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state = np.array([state])
    while not done:
        action = genome.ff(state)
        state, reward, done, _ = env.step(action)
        state = np.array([state])            
        steps += 1
        rewards += reward
        env.render()
    env.close()
    env_to_wrap.close()
    return rewards
    
for i in range(pyllab.PY_GENERATIONS):
    number_genomes = neat.get_number_of_genomes()
    neat.reset_fitnesses()
    env = []
    rewards = []
    states = []
    dones = []
    for j in range(number_genomes):
        temp_env = gym.make('TetrisA-v0')
        env.append(JoypadSpace(temp_env, MOVEMENT))
        states.append(convert_array(env[j].reset()))
        rewards.append(0)
        dones.append(False)
    for j in range(0,number_genomes,threads):
        n_ff = min(number_genomes-j,threads)
        print("Genomes: "+str(j)+"/"+str(j+n_ff))
        for z in range(max_iterations):
            print("Iteration: "+str(z))
            l = []
            indices = []
            for k in range(j,j+n_ff):
                if not dones[k]:
                    l.append(states[k])
                    indices.append(k)
            out = neat.ff_ith_genomes(l,indices,n_ff)
            for k in range(j,j+n_ff):
                if not dones[k]:
                    m = -1
                    index = -1
                    for w in range(action_size):
                        if out[k-j][w] > m:
                            m = out[k-j][w]
                            index = w
                    next_state, reward, done, _ = env[k].step(index)
                    states[k] = convert_array(next_state)
                    rewards[k]+=reward
                    dones[k] = done
        for k in range(j,j+n_ff):
            neat.increment_fitness_of_genome_ith(k,rewards[k])
    neat.generation_run()
    if(i%pyllab.PY_SAVING == 0 or i == pyllab.PY_GENERATIONS -1):
        global_innovation_number_connections = neat.get_global_innovation_number_connections()
        global_innovation_number_nodes = neat.get_global_innovation_number_nodes()
        genome = pyllab.Genome(str(i)+'.bin',global_innovation_number_nodes,global_innovation_number_connections,state_size,action_size)
        make_video(genome)
        
        






