from codecs import ignore_errors
import gym
from gym.spaces import Discrete, Box
import numpy as np
from scipy.linalg import expm

# Recreation of the Mathematic File
def U(a):
    #PauliMatrix Def
    i = np.array([[0,1], [1,0]])

    #MatrixExp where j is non real
    return expm(-1j*a*np.pi*i)

# Other Function that was used
def vv(theta, phi):
    return np.array([np.cos(theta / 2), np.sin(theta/2) * np.exp(-1j*phi)])

class QubitEnv(gym.Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Box(low=np.array([-1.]), high=np.array([1.]))
        self.observation_space = Discrete(2) #If 0 =- 1
        self.state = np.array([0,0])
    #END

    def step(self, action):

        #Create the Matrix using our action
        ww = np.matmul(U(action), vv(0,0))

        #Mutliply it by our vector state to get probability
        prob = np.power(np.abs(ww), 2)[0]

        #Get random real to get the stocastic reward
        if (np.random.ranf() < prob):
            reward = -1
        else:
            reward = 1  

        #Our env is always done after on iteration
        done = True
        info = {}

        #Return
        return self.state, reward, done, info
    #END

    #We don't need to render env
    def render(self):
        pass

    #State never changes but incase it does
    def reset(self):
        self.state = np.array([0,0])
        return self.state
    #END