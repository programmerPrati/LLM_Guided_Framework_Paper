import numpy as np
import random

class GridWorld:
    def __init__(self, GRID_SIZE, padding):
        self.padding = padding
        self.grid_size = GRID_SIZE - padding - 1 # -1 is for some reason
        self.reset()

    def reset(self):
        self.isReady = 0 # ready to push means both robots have reached if it is 1, but box is pucshed at 2 not 1
        self.noResetF = 0 # target assigning code will switch it to one to avoid certain operations in mdist

        self.box = [random.randint(self.padding, self.grid_size), random.randint(self.padding, self.grid_size)]
        self.goal = [random.randint(self.padding, self.grid_size), random.randint(self.padding, self.grid_size)]

        self.robot_a = [random.randint(self.padding, self.grid_size), random.randint(self.padding, self.grid_size)]
        self.robot_b = [random.randint(self.padding, self.grid_size), random.randint(self.padding, self.grid_size)]
        self.target_a = [0,0] # dummy targets, to be updated
        self.target_b = [0,0]
        self.switch = False # if switched, it will be 1 or true
        self.robot_a_orient = None # initial orientations, to be updated when robots on target and pushing
        self.robot_b_orient = None

        self.obstacles = [(5, 7), (5, 8), (5, 9), (11, 14), (12, 14), (13, 14)]
        # self.obs1 = (10,10)
        # self.obs1 = (11, 10)

        return self._get_state()

    def _get_state(self):
        return {
            "robot_a": tuple(float(x) for x in self.robot_a),
            "robot_b": tuple(float(x) for x in self.robot_b),
            "box": tuple(float(x) for x in self.box),
            "goal": tuple(float(x) for x in self.goal),
            "target_a": tuple(float(x) for x in self.target_a),
            "target_b": tuple(float(x) for x in self.target_b),
            "noResetF": int(self.noResetF),
            "switch": self.switch,
            "robot_a_orient": self.robot_a_orient,
            "robot_b_orient": self.robot_b_orient,
            "obstacles": self.obstacles,
        }

    def step(self, expert_action):

        action_a = expert_action["robot_a"]
        action_b = expert_action["robot_b"]

        # update the targets being passed through
        self.target_a = expert_action["target_a"]  # assign target_a
        self.target_b = expert_action["target_b"]  # assign target_b

        # We only want to draw the orientation when both on target and have waited there for 1 cycle
        if self.isReady >= 2: # initial orientations, to be updated when robots on target and pushing
            self.robot_a_orient = expert_action["robot_a_orient"]
            self.robot_b_orient = expert_action["robot_b_orient"]
        else:
            self.robot_a_orient = None  # keeping them none avoids drawing them in the visualizer
            self.robot_b_orient = None


        if self._near(self.robot_a, self.target_a):
            action_a = [action_a[0] / 2, action_a[1] / 2]

        if self._near(self.robot_b, self.target_b):
            action_b = [action_b[0] / 2, action_b[1] / 2]

        # print("robot a: : ", self.robot_a)
        # print("robot b: : ", self.robot_b)
        # print("target a", self.target_a)
        # print("target b", self.target_b)
        #
        # print("Based on above: action a = ta - ra: ", action_a)
        # print("Based on above: action b = tb - rb: ", action_b)
        # print("")


        def move(pos, action):
            pos[0] += action[0]
            pos[1] += action[1]
            return pos

        self.robot_a = move(self.robot_a, action_a)
        self.robot_b = move(self.robot_b, action_b)


        # print("robot a: after move taken : ", self.robot_a)
        # print("robot b: after move taken : ", self.robot_b)

        # if both robots are near their targets, move box toward goal
        if self._near(self.robot_a, self.target_a) and self._near(self.robot_b, self.target_b):
            if(self.isReady >=2):
                dx = np.sign(self.goal[0] - self.box[0])
                dy = np.sign(self.goal[1] - self.box[1])
                self.box[0] += dx * 1
                self.box[1] += dy * 1
                self.isReady = 0
            else:
                self.isReady += 1
        reward = -1  # default step penalty

        if tuple(map(int, self.box)) == tuple(self.goal):
            reward = 10

        done = reward == 10
        return self._get_state(), reward, done

    def _near(self, p1, p2):
        return abs(p1[0] - p2[0]) < 1 and abs(p1[1] - p2[1]) < 1
