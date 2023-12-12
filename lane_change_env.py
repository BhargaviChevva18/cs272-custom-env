import gymnasium
import ray
from gymnasium import ObservationWrapper, RewardWrapper, ActionWrapper
from ray import tune

from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from gymnasium.spaces import Space
import numpy as np
import pygame
import random
import sys
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray import tune
from ray.rllib.algorithms import ppo
from ray import air
from ray.rllib.algorithms.ppo import PPOConfig

class LaneChangeEnv(gymnasium.Env):
    def __init__(self, seed = None, optional = None):
        super(LaneChangeEnv, self).__init__()

        # Constants
        self.LANE_WIDTH = 40
        self.VEHICLE_WIDTH = 30
        self.VEHICLE_HEIGHT = 50
        self.VEHICLE_SPEED = 2
        self.LANE_CHANGE_SPEED = 1
        self.NUM_LANES = 3
        self.SCREEN_WIDTH = self.NUM_LANES * self.LANE_WIDTH
        self.SCREEN_HEIGHT = 800
        self.FPS = 60
        self.OBSTACLE_WIDTH = 30
        self.OBSTACLE_HEIGHT = 50
        self.OBSTACLE_SPEED = 2

        # Action space (left, stay, right)
        self.action_space = Discrete(3)

        self.current_step = 0
        self.max_episode_steps = 8000

        self.current_reward = 0
        self.max_reward = 25000

        # Observation space (vehicle_x, vehicle_y, obstacle_x, obstacle_y)
        obs_space_low = np.array([0, 0, 0, 0], dtype=np.float32)
        obs_space_high = np.array([ self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT], dtype=np.float32)

        self.observation_space = Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)

        # Initialize Pygame
        pygame.init()

        # Initialize the screen
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Lane Change Simulation")

        # Clock for controlling the frame rate
        self.clock = pygame.time.Clock()

        # Set the font for rendering text
        self.font = pygame.font.Font(None, 36)

        # Initialize state variables
        self.vehicle_x = None
        self.vehicle_y = None
        self.obstacles = None
        self.target_lane = None
        self.steps = None
        self.lane_change_in_progress = False

        # Reset the environment
        self.reset()

    def reset(self, *, seed= None, options=None):
        super().reset(seed = seed)
        self.vehicle_x = self.SCREEN_WIDTH // 2 - self.VEHICLE_WIDTH // 2
        self.vehicle_y = self.SCREEN_HEIGHT - self.VEHICLE_HEIGHT
        self.target_lane = 1  # Start in the middle lane
        self.obstacles = [[random.randint(0, self.NUM_LANES - 1) * self.LANE_WIDTH + (self.LANE_WIDTH - self.OBSTACLE_WIDTH) // 2, 0]]
        self.steps = 0
        self.lane_change_in_progress = False

        observation = np.array([self.vehicle_x, self.vehicle_y, self.obstacles[0][0], self.obstacles[0][1]], dtype = np.float32)
        if not self.observation_space.contains(observation):
            raise ValueError("Initial observation is not within the valid range.", observation)
        self.current_step = 0
        self.current_reward = 0
        return observation, {}

    def step(self, action):
        self.current_step += 1

        # Maximum steps reached, hence ending the episode
        if self.current_step >= self.max_episode_steps:
            done = True
            obs = np.array([0,0,0,0], dtype = np.float32)
            return obs, 0, True, True, {}

        # Gradually move the vehicle towards the center of the selected lane
        desired_x = self.target_lane * self.LANE_WIDTH + self.LANE_WIDTH // 2 - self.VEHICLE_WIDTH // 2
        self.vehicle_x += self.LANE_CHANGE_SPEED * np.sign(desired_x - self.vehicle_x)

        # If a lane change is in progress, wait until it is completed
        if self.lane_change_in_progress and desired_x == self.vehicle_x:
            self.lane_change_in_progress = False
            # action = 1  # Stay in the current lane
        else:
            self.lane_change_in_progress = True
            action = 1

        # Move the vehicle left or right based on the chosen action
        if action == 0 and self.target_lane > 0:
            self.target_lane -= 1
        elif action == 2 and self.target_lane < self.NUM_LANES - 1:
            self.target_lane += 1

        # Move obstacles and generate new ones
        for obstacle in self.obstacles:
            obstacle[1] += self.OBSTACLE_SPEED
            if obstacle[1] > self.SCREEN_HEIGHT:
                obstacle[0] = random.randint(0, self.NUM_LANES - 1) * self.LANE_WIDTH + (self.LANE_WIDTH - self.OBSTACLE_WIDTH) // 2
                obstacle[1] = 0

        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if (
                self.vehicle_x < obstacle[0] + self.OBSTACLE_WIDTH
                and self.vehicle_x + self.VEHICLE_WIDTH > obstacle[0]
                and self.vehicle_y < obstacle[1] + self.OBSTACLE_HEIGHT
                and self.vehicle_y + self.VEHICLE_HEIGHT > obstacle[1]
            ):
                reward = -100
                done = True
                break
        else:
            reward = 1
            done = False

        # Increment step counter
        self.steps += 1

        # bonus reward
        if self.steps % 1000 == 0:
            reward = 10

        self.current_reward += reward
        if(self.current_reward>=self.max_reward):
            done = True

        # Return the next state, reward, and done flag
        next_state = np.array([self.vehicle_x, self.vehicle_y, self.obstacles[0][0], self.obstacles[0][1]], dtype = np.float32)
        self.render()
        return next_state, reward, done, False, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the background
        self.screen.fill((0, 0, 0))  # Black color for the road

        # Draw the dotted lane divisions
        for lane in range(1, self.NUM_LANES):
            lane_start = lane * self.LANE_WIDTH
            pygame.draw.line(self.screen, (255, 255, 255), (lane_start, 0), (lane_start, self.SCREEN_HEIGHT), 1)

        # Draw the vehicle
        pygame.draw.rect(self.screen, (0, 128, 0), (self.vehicle_x, self.vehicle_y, self.VEHICLE_WIDTH, self.VEHICLE_HEIGHT))

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), (obstacle[0], obstacle[1], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT))

        # Update the display
        pygame.display.flip()

        # Set the frame rate
        self.clock.tick(self.FPS)
        



def train_car_env():
    tune.register_env("CarLaneEnv", lambda config: LaneChangeEnv())

    ray.init()

    config = {
        "env": "CarLaneEnv",
        "num_gpus": 0,
        "num_workers": 4,
        "framework": "tf",
        "lr": 0.3,
        "gamma": 0.99,
        "entropy_coeff": 0.01,
        "train_batch_size": 4000,
        "rollout_fragment_length": 200,
        "sgd_minibatch_size": 128,
        "clip_param": 0.2,
        "kl_coeff": 0.5,
        "kl_target": 0.01,
    }

    stop_criteria = {
        "training_iteration": 100,
        "episode_reward_mean": 1000,
    }
    
    results = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(stop=stop_criteria),
        param_space=config,
    ).fit()

    ray.shutdown()

train_car_env()

