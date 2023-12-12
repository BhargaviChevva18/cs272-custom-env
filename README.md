# Autonomous Driving Scenario
## Project Summary
<!-- Around 200 Words -->
<!-- Cover (1) What problem you are solving, (2) Who will use this RL module and be happy with the learning, and (3) a brief description of the results -->
Recent advancements in Artificial Intelligence have enabled major advancements in developing autonomous driving cars. Training these cars in real-world environments is a challenge due to the high costs and inherent risks involved. Our project aims to create a simulated environment for self-driving cars to train in a virtual environment. We created an elemental yet crucial version of the real-world environment in the form of a three-lane highway with our car agent moving in the forward direction. The goal of our project is to train the car agent to avoid obstacles by shifting across the three lanes. For simplicity, our obstacles are currently stationary and the velocity of our car agent is constant.
![env](https://github.com/BhargaviChevva18/cs272-custom-env/assets/112223228/8313783d-4936-4b72-99b3-c6141d5d57ea)


## State Space
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

The environment consists of a three-lane highway with the car agent positioned at the bottom of the window. 
The dimensions of the window are: [120, 800]
The state space of the environment involves:

| No. | Feature                      | Min | Max |
|-----|------------------------------|-----|------|
| 1.  |X coordinate of the car agent |  0  | 120  |
| 2.  |Y coordinate of the car agent | 750 | 750  |
| 3.  |X coordinate of the obstacle  |  0  | 120  |
| 4.  |Y coordinate of the obstacle  |  0  | 800  |

The values at the right denote the possible values for each component of state. The value of the y coordinate of the car agent is constant because we visualize the highway through the lens of the car, i.e., looks like with each passing time step, the environment moves towards the car and the car remains in the same position. 

## Action Space
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
In order to avoid obstacles, the car agent can shift across the three lanes of the highway. Therefore, the possible actions of the agent are :

| Action | Description |
|--------|-----------------------------------|
|    0   |  shift to the immediate left lane |
|    1   |     remain in the same lane       |
|    2   | shift to the immediate right lane |

## Rewards
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
In order to encourage our car agent to avoid obstacles, we gave it a +1 reward for every time step it avoids an obstacle. Car agent also receives a +!0 bonus if it is successfully able to avoid obstacles in the past 1000 steps. The agent receives a negative reward of -1 if it hits an obstacle and the game terminates.

## Starting State
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
Initially, the car agent starts at the bottom center of the window. The coordinated for the starting position are [45, 750]

## Episode End
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
Each episode terminates if the car agent either hits an obstacle or achieves a maximum reward of 25000 


## RL Algorithm 
We experimented with our Car Driving Simulation Environment using Ray RLLib's Proximal Policy Optimization(PPO) algorithm. PPO is an on-policy algorithm that can work with both discrete and continuous environments. The hyperparameters we used for training PPO algorithm on our Car Environment are as follows:
        {"num_gpus": 0,
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
        "kl_target": 0.01}



## Results

