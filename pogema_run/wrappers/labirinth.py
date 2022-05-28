import gym
import numpy as np
import cv2 as cv

class Wrapper(gym.Wrapper):
    
    
    '''
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=bool)
    '''
    
    def __init__(self, env, headless = True):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(0.0, 1.0, shape=(3, 11, 11), dtype='float32')
        ))
        self.action_space = gym.spaces.MultiDiscrete([5])
        #print(self.action_space)
        #self.observation_space =  gym.spaces.Box(0.0, 1.0, shape=(3, 11, 11), dtype='float32')
        self.env = env
        self.headless = headless
        self.prev_pos = [0, 0]
        self.prev_prev_pos = [0, 0]
    
    def step(self, action):
        self.steps += 1
        #action = np.int0(action)
        actions = []
        for i in range(self.env.config.num_agents):
            actions.append(np.random.randint(0, 5))
        actions[0] = int(action)
        next_state, reward, done, info = self.env.step(actions)
        #next_state =next_state[0]
        next_state = {'observation': next_state[0]}
        
        #rew = -np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0])) + self.delta - self.steps * 0.1
        #print(rew)
        rew = reward[0] * 20
        if self.prev_pos[0] == self.env.get_agents_xy()[0][0] and self.prev_pos[1] == self.env.get_agents_xy()[0][1]:
            rew -= 0.1
        elif self.prev_prev_pos[0] == self.env.get_agents_xy()[0][0] and self.prev_prev_pos[1] == self.env.get_agents_xy()[0][1]:
            rew -= 0.1
        self.prev_prev_pos = self.prev_pos.copy()
        self.prev_pos = np.array(self.env.get_agents_xy()[0])
        for i in range(len(info)):
            info[i]['is_success'] = reward[i]
        if not self.headless:
            self.render()
        return next_state, rew, done[0], info[0]

    def render(self):
        self.check_reset()
        image = np.zeros((len(self.grid.obstacles), len(self.grid.obstacles[0]),3), np.uint8)
        for i in range(len(self.grid.obstacles)):
            for j in range(len(self.grid.obstacles)):
                if self.grid.obstacles[i][j]:
                    image[i][j] = [255, 0, 0]
        image[self.grid.positions_xy[0][0]][self.grid.positions_xy[0][1]] = [0, 0, 255]
        image[self.grid.finishes_xy[0][0]][self.grid.finishes_xy[0][1]] = [0, 255, 0]
        for i in self.grid.positions_xy[1:]:
            image[i[0]][i[1]] = [0, 50, 150]
        for i in self.grid.finishes_xy[1:]:
            image[i[0]][i[1]] = [0, 150, 50]
        
        cv.imshow("field", cv.resize(image, (400, 400), interpolation = cv.INTER_AREA))
        #cv.imshow("field", image)
        cv.waitKey(1)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.steps = 0
        self.delta = np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0]))
        return {'observation': state[0]}