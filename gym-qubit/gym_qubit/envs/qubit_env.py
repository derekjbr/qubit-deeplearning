from cmath import pi
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
            #self.action_space = Box(low=np.array([-1.]), high=np.array([1.]))
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0,0,-1]),high=np.array([2*pi, pi,1])) 

        #The state of our system. It depends of theta, phi, and spin
        self.spin = np.random.randint(0,10) / 10
        self.phi = 0
        self.theta = 0
        self.episodes = 0

        self.state = None
    #END

    def step(self, action):
        self.spin += (action - 1) / 10

        #Create the Matrix using our action
        ww = np.matmul(U(self.spin), vv(self.theta,self.phi))

        #Mutliply it by our vector state to get probability
        prob = np.power(np.abs(ww), 2)[0]

        #Get random real to get the stocastic reward
        if (np.random.ranf() < prob):
            reward = -1
        else:
            reward = 1  

        #Our env is always done after on iteration
        done = bool (
            self.episodes == 30
            or self.spin == 0.5
        )

        self.state = (self.spin, self.theta, self.phi)
        
        info = {}
        self.episodes += 1

        #Return
        return np.array(self.state,dtype=np.float32), reward, done, {}
    #END

    #We don't need to render env
    def render(self):
        pass

    #State never changes but incase it does
    def reset(self):
        self.spin = np.random.randint(0,10) / 10
        self.state = (self.spin, self.theta, self.phi)
        self.episodes = 0

        return np.array(self.state,dtype=np.float32)
    #END
 