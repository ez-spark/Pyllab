import gym
import numpy as np
import pyllab
import time

def displayImage(image):
    plt.imshow(image)
    plt.show()
env = gym.make("LunarLander-v2", render_mode="human")
rewards = 0
#env = wrappers.Monitor(env_to_wrap, directory, force=True)
for gam in range(0,600,100):
    filename = "./ll_rainbow/"+str(gam)+'.bin'
    net = pyllab.duelingCategoricalDQN(filename = filename,input_size = 8,action_size = 4, qr_dqn = True,  n_atoms = 51, v_min = -10, v_max = 10, mode = True)
    print("network: "+filename)
    steps = 0
    done = False
    state = env.reset()[0]
    state = np.array([state])
    l = []
    renders = []
    for step in range(1000):
        img = env.render()
        #action = int(input("insert action: "))
        action = net.get_best_action(state,8)
        net.reset()
        t = env.step(action)
        state = t[0]
        reward = t[1]
        done = t[2] 
        state = np.array([state])            
        steps += 1
        rewards += reward
        if done:
            break
        print(reward)
        print(rewards)
        #time.sleep(0.1)
