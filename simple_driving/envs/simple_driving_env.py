import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.Box(
        low=np.array([-40, -40, 0, -math.pi, -40, -40], dtype=np.float32),
        high=np.array([40, 40, 60, math.pi, 40, 40], dtype=np.float32)
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
            self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 10
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.obstacle = None
        self.obstacle_pos = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.keep_goal = True
        self.reset()
        self._envStepCounter = 0

    def step(self, action):
        reward = 0
        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)

        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, _ = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()
            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        dx, dy, dist_to_goal, angle_to_goal, ob_dx, ob_dy = car_ob
        dist_to_obstacle = math.sqrt(ob_dx**2 + ob_dy**2)

        is_forward = action[0] > 0
        is_backward = action[0] < 0
        angle_reward = math.cos(angle_to_goal) if is_forward else -math.cos(angle_to_goal) if is_backward else 0

        progress_reward = self.prev_dist_to_goal - dist_to_goal
        if progress_reward > 0:
            reward = progress_reward * 5 + angle_reward
        elif progress_reward < 0:
            reward = progress_reward * 10 + angle_reward
        else:
            reward = -1 + angle_reward

        if dist_to_obstacle < 0.8:
            reward -= (1.0 - dist_to_obstacle) * 2
            print("⚠️ Collision risk with obstacle!")

        if dist_to_obstacle < 0.05:
            print("💥 Collision with obstacle!")
            reward -= 50

        if action[0] == 0:
            reward -= 0.5

        if self._envStepCounter > 1200 and dist_to_goal > self.prev_dist_to_goal:
            print("🚨 Stuck — forcing reset")
            self.done = True

        self.prev_dist_to_goal = dist_to_goal

       
            


        if dist_to_goal < 1.5 and not self.reached_goal:
            reward += 50
            print("🚗 Reached goal!")
            self.done = True
            self.reached_goal = True

        print(f"[Step {self._envStepCounter}]")
        print(f"    Car pos:   {np.round(carpos[:2], 2)}")
        print(f"    Goal pos:  {np.round(goalpos[:2], 2)}")
        print(f"    Dist:      {dist_to_goal:.2f}")
        print(f"    Reward:    {reward:.3f}")
        print(f"    Action:    {action}")

        return np.array(car_ob, dtype=np.float32), reward, self.done, {"reached_goal": self.reached_goal}

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False
        self.goal_object = Goal(self._p, self.goal)

        while True:
            ob_x = self.np_random.uniform(-4, 4)
            ob_y = self.np_random.uniform(-4, 4)
            carpos = self.car.get_observation()
            dist_to_car = math.sqrt((ob_x - carpos[0])**2 + (ob_y - carpos[1])**2)
            if dist_to_car > 2.5:
                break

        self.obstacle_id = self._p.loadURDF("simple_driving/resources/obstacle.urdf",
                                            basePosition=[ob_x, ob_y, 0])
        carpos = self.car.get_observation()
        self.prev_dist_to_goal = math.sqrt((carpos[0] - self.goal[0])**2 + (carpos[1] - self.goal[1])**2)
        return np.array(self.getExtendedObservation(), dtype=np.float32)

    def getExtendedObservation(self):
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, _ = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        dx, dy = goalPosInCar[0], goalPosInCar[1]
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        ob_dx, ob_dy = 0.0, 0.0
        if hasattr(self, "obstacle_id"):
            obstacle_pos, _ = self._p.getBasePositionAndOrientation(self.obstacle_id)
            obstacle_rel, _ = self._p.multiplyTransforms(invCarPos, invCarOrn, obstacle_pos, [0, 0, 0, 1])
            ob_dx, ob_dy = obstacle_rel[0], obstacle_rel[1]

        return [dx, dy, dist, angle, ob_dx, ob_dy]


    def _termination(self):
        return self._envStepCounter > 2000

    def render(self, mode='human'):
        return np.array([])

    def close(self):
        self._p.disconnect()
